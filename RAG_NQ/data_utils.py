"""NQ data iteration + context-window construction with controllable gold position."""
import gzip
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple


def sample_position_ratio(
    zone_lo: float,
    zone_hi: float,
    qid: str,
    sa_start: int,
    sa_end: int,
    zone_name: str,
    seed: int,
) -> float:
    """Pick a target ratio uniformly in [zone_lo, zone_hi] for a specific record.
    Deterministic: same (qid, sa, zone, seed) → same ratio across reruns."""
    key = f"{seed}|{qid}|{sa_start}|{sa_end}|{zone_name}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    rng = random.Random(int(h[:16], 16))
    return rng.uniform(zone_lo, zone_hi)


def iter_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_nq_examples(input_path: str) -> Iterator[Dict[str, Any]]:
    path = Path(input_path)
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl.gz"))
    for fp in files:
        logging.info(f"Reading file: {fp}")
        for ex in iter_jsonl_gz(fp):
            yield ex


def get_question_id(example: Dict[str, Any]) -> str:
    return str(example.get("example_id") or example.get("question_id") or example.get("id"))


def get_question_text(example: Dict[str, Any]) -> str:
    if "question_text" in example:
        return example["question_text"]
    if "question" in example:
        return example["question"]
    return " ".join(example.get("question_tokens", []))


def get_doc_tokens(example: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for t in example.get("document_tokens", []):
        if isinstance(t, dict):
            out.append(t.get("token", ""))
        else:
            out.append(str(t))
    return out


def is_valid_span(span: Dict[str, Any]) -> bool:
    return (isinstance(span, dict)
            and span.get("start_token", -1) >= 0
            and span.get("end_token", -1) > span.get("start_token", -1))


def extract_answer_text(doc_tokens: List[str], start: int, end: int) -> str:
    return " ".join(doc_tokens[start:end]).strip()


def _full_prompt_token_count(tokenizer, system_prompt: str, user_template: str,
                              question: str, context_text: str,
                              use_chat_template: bool) -> int:
    """Length of the FULL prompt (system + chat-template + user) in decoder tokens."""
    user_content = (user_template
                    .replace("{context}", context_text)
                    .replace("{question}", question))
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_content})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return len(tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:
            pass
    full = (system_prompt + "\n\n" if system_prompt else "") + user_content + "\nAnswer:"
    return len(tokenizer(full, add_special_tokens=True).input_ids)


def build_context_at_position(
    tokenizer,
    system_prompt: str,
    user_template: str,
    question: str,
    doc_tokens: List[str],
    sa_start: int,
    sa_end: int,
    target_window_size: int,
    target_pos_ratio: float,
    max_input_tokens: int,
    use_chat_template: bool = True,
) -> Dict[str, Any]:
    """Slice the document such that:
      * the gold span [sa_start, sa_end) is fully contained in the slice,
      * the gold center sits at fractional offset `target_pos_ratio`
        of the slice (in word-tokens — start/middle/end zones),
      * the encoded full prompt (system + chat template + user) fits into
        target_window_size decoder tokens, capped by max_input_tokens.
    For target_window_size=-1 the budget equals max_input_tokens (full-doc mode).
    """
    doc_len = len(doc_tokens)
    sa_start = max(0, min(sa_start, doc_len))
    sa_end = max(sa_start, min(sa_end, doc_len))

    if target_window_size == -1:
        budget = max_input_tokens
    else:
        budget = (min(target_window_size, max_input_tokens)
                  if max_input_tokens else target_window_size)

    # Binary search over slice length in word-tokens. The largest slice that
    # still fits the prompt budget wins.
    lo = sa_end - sa_start
    hi = doc_len
    best_s, best_e = sa_start, sa_end

    # Always check the answer-only slice first as a floor.
    ctx_text = " ".join(doc_tokens[sa_start:sa_end])
    n_tokens = _full_prompt_token_count(
        tokenizer, system_prompt, user_template, question, ctx_text, use_chat_template
    )
    if n_tokens > budget:
        # Even the bare answer doesn't fit — return it anyway; caller will truncate.
        return {
            "context_tokens": doc_tokens[sa_start:sa_end],
            "context_start_token": sa_start,
            "context_end_token": sa_end,
        }

    while lo <= hi:
        mid = (lo + hi) // 2
        if mid <= 0:
            break
        gold_center = (sa_start + sa_end) / 2.0
        s = int(round(gold_center - mid * target_pos_ratio))
        s = max(0, min(s, doc_len - mid))
        e = s + mid
        # Snap window so gold span is fully inside.
        if s > sa_start:
            s = sa_start
            e = s + mid
        if e < sa_end:
            e = sa_end
            s = max(0, e - mid)
        s = max(0, s)
        e = min(doc_len, e)
        if s > sa_start or e < sa_end:
            hi = mid - 1
            continue

        ctx_text = " ".join(doc_tokens[s:e])
        n_tokens = _full_prompt_token_count(
            tokenizer, system_prompt, user_template, question, ctx_text, use_chat_template
        )
        if n_tokens <= budget:
            best_s, best_e = s, e
            lo = mid + 1
        else:
            hi = mid - 1

    return {
        "context_tokens": doc_tokens[best_s:best_e],
        "context_start_token": best_s,
        "context_end_token": best_e,
    }


def iterate_oracle_examples(
    input_path: str,
    limit_questions: int = None,
) -> Iterator[Tuple[str, str, List[str], int, int, int, int, str]]:
    """Yield (qid, question, doc_tokens, long_start, long_end,
    sa_start, sa_end, gold_text) for every annotated short answer.
    A single qid can appear multiple times if there are multiple short answers."""
    seen: set = set()
    for ex in iter_nq_examples(input_path):
        qid = get_question_id(ex)
        if limit_questions is not None and qid not in seen and len(seen) >= limit_questions:
            break
        q = get_question_text(ex)
        toks = get_doc_tokens(ex)
        for ann in ex.get("annotations", []) or []:
            la = ann.get("long_answer", {}) or {}
            if not is_valid_span(la):
                continue
            for sa in ann.get("short_answers", []) or []:
                if not is_valid_span(sa):
                    continue
                gold = extract_answer_text(toks, sa["start_token"], sa["end_token"])
                seen.add(qid)
                yield (qid, q, toks,
                       la["start_token"], la["end_token"],
                       sa["start_token"], sa["end_token"],
                       gold)
