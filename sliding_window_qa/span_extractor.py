"""
Stage 2: span extraction.

Берёт top-K чанков, отранжированных cross-encoder'ом, и применяет к
каждой паре (question, chunk_text) модель QA
(AutoModelForQuestionAnswering). Возвращает лучший span (текст + скор)
по всем K чанкам.

Скор span'а = start_logit + end_logit (стандартная SQuAD-схема).
"""

import gc
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


@dataclass
class SpanPrediction:
    text: str
    score: float
    chunk_id: int
    word_start: int      # в шкале слов чанка
    word_end: int        # включительно
    n_chunk_words: int


def load_span_extractor(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """
    Загружает QA-модель + токенайзер. Поддерживается локальная папка
    с весами (`from_pretrained(<path>)`) и HF id.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def extract_spans_for_top_k(
    model,
    tokenizer,
    question: str,
    top_chunks: List[dict],          # каждая запись: {"chunk_id", "text", ...}
    device: torch.device,
    max_seq_length: int,
    max_span_words: int,
    batch_size: int,
    n_best_logits: int = 20,
) -> List[SpanPrediction]:
    """
    Возвращает для одного вопроса список SpanPrediction'ов длиной до K
    (по одному лучшему span на чанк). Сортировка не делается — её
    производит вызывающая сторона.
    """

    if not top_chunks:
        return []

    questions = [question] * len(top_chunks)
    chunk_word_lists = [c["text"].split() for c in top_chunks]
    chunk_ids = [c["chunk_id"] for c in top_chunks]

    results: List[SpanPrediction] = []

    i = 0
    n = len(top_chunks)
    while i < n:
        bs = min(batch_size, n - i)
        q_b = questions[i : i + bs]
        c_b = chunk_word_lists[i : i + bs]
        cid_b = chunk_ids[i : i + bs]

        try:
            enc = tokenizer(
                [q.split() for q in q_b],
                c_b,
                is_split_into_words=True,
                max_length=max_seq_length,
                truncation="only_second",
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**inputs)
            start_logits = outputs.start_logits.float().cpu()
            end_logits = outputs.end_logits.float().cpu()

            for j in range(bs):
                seq_ids = enc.sequence_ids(j)
                word_ids = enc.word_ids(j)
                s_l = start_logits[j]
                e_l = end_logits[j]

                top_s = torch.topk(s_l, k=min(n_best_logits, len(s_l))).indices.tolist()
                top_e = torch.topk(e_l, k=min(n_best_logits, len(e_l))).indices.tolist()

                best_score: Optional[float] = None
                best_span = None  # (sw, ew)

                for si in top_s:
                    if seq_ids[si] != 1:
                        continue
                    sw = word_ids[si]
                    if sw is None:
                        continue
                    for ei in top_e:
                        if ei < si:
                            continue
                        if seq_ids[ei] != 1:
                            continue
                        ew = word_ids[ei]
                        if ew is None or ew < sw:
                            continue
                        if ew - sw + 1 > max_span_words:
                            continue
                        sc = float(s_l[si] + e_l[ei])
                        if best_score is None or sc > best_score:
                            best_score = sc
                            best_span = (sw, ew)

                words_j = c_b[j]
                if best_span is None:
                    results.append(
                        SpanPrediction(
                            text="",
                            score=float("-inf"),
                            chunk_id=cid_b[j],
                            word_start=-1,
                            word_end=-1,
                            n_chunk_words=len(words_j),
                        )
                    )
                else:
                    sw, ew = best_span
                    text = " ".join(words_j[sw : ew + 1]).strip()
                    results.append(
                        SpanPrediction(
                            text=text,
                            score=best_score,
                            chunk_id=cid_b[j],
                            word_start=sw,
                            word_end=ew,
                            n_chunk_words=len(words_j),
                        )
                    )

            i += bs

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if bs == 1:
                logging.error(
                    f"Span OOM at bs=1 (seq_len={max_seq_length}); empty prediction."
                )
                results.append(
                    SpanPrediction(
                        text="",
                        score=float("-inf"),
                        chunk_id=cid_b[0],
                        word_start=-1,
                        word_end=-1,
                        n_chunk_words=len(c_b[0]),
                    )
                )
                i += 1
            else:
                new_bs = max(1, bs // 2)
                logging.warning(f"Span OOM at bs={bs}, retry with {new_bs}")
                batch_size = new_bs

    return results


def pick_best_span(predictions: List[SpanPrediction]) -> Optional[SpanPrediction]:
    if not predictions:
        return None
    finite = [p for p in predictions if p.score != float("-inf")]
    pool = finite if finite else predictions
    return max(pool, key=lambda p: p.score)
