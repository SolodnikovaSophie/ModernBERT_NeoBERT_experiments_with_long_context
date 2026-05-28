"""EM, F1, Token Recall, BERTScore, BLEURT — with bootstrap CIs.

Basic metrics (EM/F1/Recall) are computed during the generation phase.
Semantic metrics (BERTScore, BLEURT) are computed in a separate phase
(after the decoder has been freed) to avoid VRAM contention.
"""
import collections
import gc
import json
import logging
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# ============================================================
# String normalization (SQuAD)
# ============================================================

def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        excl = set(string.punctuation)
        return "".join(ch for ch in t if ch not in excl)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split() if s else []


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    g, p = get_tokens(gold), get_tokens(pred)
    if not g or not p:
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pre, rec = n / len(p), n / len(g)
    return 2 * pre * rec / (pre + rec)


def compute_token_recall(gold: str, pred: str) -> float:
    g, p = get_tokens(gold), get_tokens(pred)
    if not g:
        return float(not p)
    common = collections.Counter(g) & collections.Counter(p)
    return sum(common.values()) / len(g)


def bootstrap_ci(values: List[float], n_samples: int = 1000,
                 confidence_level: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_samples))
    alpha = 1.0 - confidence_level
    return (means[int((alpha / 2) * n_samples)],
            means[int((1 - alpha / 2) * n_samples) - 1])


def best_basic_scores_per_question(predictions: List[str],
                                   gold_answers: List[str]) -> Dict[str, Any]:
    """Max over all (pred, gold) pairs. Returns the (pred, gold) that won
    plus the three scalar scores."""
    best = {"best_prediction": "", "best_gold": "",
            "exact_match": 0.0, "f1": 0.0, "token_recall": 0.0}
    for pred in predictions:
        for gold in gold_answers:
            ex = compute_exact(gold, pred)
            f1 = compute_f1(gold, pred)
            rc = compute_token_recall(gold, pred)
            if (f1, ex, rc) > (best["f1"], best["exact_match"], best["token_recall"]):
                best = {"best_prediction": pred, "best_gold": gold,
                        "exact_match": float(ex), "f1": f1, "token_recall": rc}
    return best


def aggregate_basic_metrics(per_question: Dict[str, Dict[str, List[str]]],
                            selected: List[str],
                            bootstrap_samples: int = 1000,
                            confidence_level: float = 0.95,
                            seed: int = 42) -> Dict[str, Any]:
    em_v, f1_v, rec_v = [], [], []
    best_preds, refs_per_q, qids_ordered = [], [], []
    for qid, payload in per_question.items():
        preds = payload["predictions"]
        golds = list(set(payload["gold_answers"]))
        b = best_basic_scores_per_question(preds, golds)
        em_v.append(b["exact_match"])
        f1_v.append(b["f1"])
        rec_v.append(b["token_recall"])
        best_preds.append(b["best_prediction"] or (preds[0] if preds else ""))
        refs_per_q.append(golds)
        qids_ordered.append(qid)

    out: Dict[str, Any] = {"count": len(qids_ordered)}
    if "em" in selected:
        out["EM"] = (sum(em_v) / len(em_v) * 100) if em_v else 0.0
        lo, hi = bootstrap_ci(em_v, bootstrap_samples, confidence_level, seed)
        out["EM_CI"] = [lo * 100, hi * 100]
    if "f1" in selected:
        out["F1"] = (sum(f1_v) / len(f1_v) * 100) if f1_v else 0.0
        lo, hi = bootstrap_ci(f1_v, bootstrap_samples, confidence_level, seed)
        out["F1_CI"] = [lo * 100, hi * 100]
    if "recall" in selected:
        out["Recall"] = (sum(rec_v) / len(rec_v) * 100) if rec_v else 0.0
        lo, hi = bootstrap_ci(rec_v, bootstrap_samples, confidence_level, seed)
        out["Recall_CI"] = [lo * 100, hi * 100]
    return out, best_preds, refs_per_q, qids_ordered


# ============================================================
# Semantic scorers — loaded lazily, reused across runs
# ============================================================

class BERTScorer:
    def __init__(self, model_type: str = "roberta-large", device: str = "cuda"):
        from bert_score import BERTScorer as _BS
        import torch
        self._torch = torch
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        # roberta-large emits a harmless "pooler.dense not initialized" message —
        # BERTScore doesn't use pooler, so the warning is pure noise.
        try:
            from transformers.utils import logging as hf_logging
            prev = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            prev = None
        try:
            self.scorer = _BS(model_type=model_type, lang="en",
                              device=self.device, rescale_with_baseline=False)
        finally:
            if prev is not None:
                try:
                    from transformers.utils import logging as hf_logging
                    hf_logging.set_verbosity(prev)
                except Exception:
                    pass

    def score(self, predictions: List[str],
              references_per_pred: List[List[str]],
              batch_size: int = 32) -> List[float]:
        flat_p, flat_r, ranges = [], [], []
        cur = 0
        for pred, refs in zip(predictions, references_per_pred):
            if not refs:
                ranges.append((cur, cur))
                continue
            n = len(refs)
            flat_p.extend([pred or ""] * n)
            flat_r.extend(refs)
            ranges.append((cur, cur + n))
            cur += n
        if not flat_p:
            return [0.0] * len(predictions)
        P, R, F = self.scorer.score(flat_p, flat_r, batch_size=batch_size, verbose=False)
        f_list = F.tolist()
        out: List[float] = []
        for s, e in ranges:
            out.append(max(f_list[s:e]) if s < e else 0.0)
        return out

    def free(self):
        try:
            del self.scorer
        except Exception:
            pass
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _resolve_bleurt_sp_tokenizer_cls():
    """`BleurtSPTokenizer` may live in different submodules across versions of
    bleurt-pytorch. Try a few paths and return the first hit."""
    candidates = [
        ("bleurt_pytorch", "BleurtSPTokenizer"),
        ("bleurt_pytorch.bleurt.tokenization_bleurt_sp", "BleurtSPTokenizer"),
        ("bleurt_pytorch.tokenization_bleurt_sp", "BleurtSPTokenizer"),
        ("bleurt_pytorch.bleurt", "BleurtSPTokenizer"),
    ]
    for mod_path, name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    return None


class BLEURTScorer:
    def __init__(self, model_path: str = "lucadiliello/BLEURT-20-D12",
                 device: str = "cuda", max_length: int = 512):
        # BLEURT-20 family uses RemBERT/SentencePiece → BleurtSPTokenizer.
        # Older BLEURT checkpoints use BERT/WordPiece → BleurtTokenizer.
        from bleurt_pytorch import (BleurtConfig,
                                    BleurtForSequenceClassification,
                                    BleurtTokenizer)
        import torch
        BleurtSPTokenizer = _resolve_bleurt_sp_tokenizer_cls()

        log = logging.getLogger(__name__)
        self._torch = torch
        self.max_length = max_length
        self.config = BleurtConfig.from_pretrained(model_path)

        prefer_sp = ("bleurt-20" in model_path.lower()
                     and BleurtSPTokenizer is not None)
        candidates = ([BleurtSPTokenizer, BleurtTokenizer] if prefer_sp
                      else [BleurtTokenizer, BleurtSPTokenizer] if BleurtSPTokenizer
                      else [BleurtTokenizer])

        # Force the right class — if `from_pretrained` would emit a
        # "tokenizer class mismatch" warning, that means we picked the wrong
        # parent. We reach into HF logging to silence the cosmetic warning
        # only; the actual tokenizer class is set explicitly here.
        try:
            from transformers.utils import logging as hf_logging
            prev_verbosity = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            prev_verbosity = None

        last_err = None
        self.tokenizer = None
        try:
            for tok_cls in candidates:
                if tok_cls is None:
                    continue
                try:
                    self.tokenizer = tok_cls.from_pretrained(model_path)
                    log.info(f"BLEURT tokenizer: {tok_cls.__name__} "
                             f"(found via {tok_cls.__module__})")
                    break
                except Exception as e:
                    log.debug(f"BLEURT tokenizer {tok_cls.__name__} failed: {e}")
                    last_err = e
        finally:
            if prev_verbosity is not None:
                try:
                    from transformers.utils import logging as hf_logging
                    hf_logging.set_verbosity(prev_verbosity)
                except Exception:
                    pass

        if self.tokenizer is None:
            raise RuntimeError(
                f"Cannot load BLEURT tokenizer from {model_path}: {last_err}")
        if prefer_sp and not type(self.tokenizer).__name__.startswith("BleurtSP"):
            log.warning(
                f"BLEURT-20 expects BleurtSPTokenizer but loaded "
                f"{type(self.tokenizer).__name__}. Install/upgrade "
                f"bleurt-pytorch (it should expose BleurtSPTokenizer). "
                f"BLEURT scores may be incorrect.")

        self.model = BleurtForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda"
                                   if (device == "cuda" and torch.cuda.is_available())
                                   else "cpu")
        self.model.to(self.device).eval()

    def score(self, predictions: List[str],
              references_per_pred: List[List[str]],
              batch_size: int = 16) -> List[float]:
        flat_p, flat_r, ranges = [], [], []
        cur = 0
        for pred, refs in zip(predictions, references_per_pred):
            if not refs:
                ranges.append((cur, cur))
                continue
            n = len(refs)
            flat_p.extend([pred or ""] * n)
            flat_r.extend(refs)
            ranges.append((cur, cur + n))
            cur += n
        if not flat_p:
            return [0.0] * len(predictions)
        scores: List[float] = []
        with self._torch.no_grad():
            for i in range(0, len(flat_p), batch_size):
                bp = flat_p[i:i + batch_size]
                br = flat_r[i:i + batch_size]
                enc = self.tokenizer(br, bp, padding="longest",
                                     truncation=True, max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
                out = self.model(**enc)
                scores.extend(out.logits.flatten().cpu().tolist())
        result: List[float] = []
        for s, e in ranges:
            result.append(max(scores[s:e]) if s < e else 0.0)
        return result

    def free(self):
        try:
            self.model.cpu()
            del self.model
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


# ============================================================
# Phase-2 driver: walk predictions.json files and add semantic metrics
# ============================================================

def _load_predictions(pred_path: Path) -> Dict[str, Dict[str, List[str]]]:
    with open(pred_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    per_q: Dict[str, Dict[str, List[str]]] = collections.defaultdict(
        lambda: {"predictions": [], "gold_answers": []})
    for r in records:
        qid = r.get("question_id")
        if qid is None:
            continue
        per_q[qid]["predictions"].append(r.get("prediction", "") or "")
        per_q[qid]["gold_answers"].append(r.get("gold_answer", "") or "")
    return per_q


def add_semantic_metrics_to_dir(
    metrics_path: Path,
    pred_path: Path,
    *,
    bertscore: Optional[BERTScorer] = None,
    bleurt: Optional[BLEURTScorer] = None,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)

    if not metrics_path.exists() or not pred_path.exists():
        log.warning(f"skip: metrics or predictions missing in {metrics_path.parent}")
        return {}

    with open(metrics_path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    per_q = _load_predictions(pred_path)
    if not per_q:
        return existing

    qids = list(per_q.keys())
    best_preds, refs_per_q = [], []
    for qid in qids:
        preds = per_q[qid]["predictions"]
        golds = list(set(per_q[qid]["gold_answers"]))
        b = best_basic_scores_per_question(preds, golds)
        best_preds.append(b["best_prediction"] or (preds[0] if preds else ""))
        refs_per_q.append(golds)

    if bertscore is not None:
        try:
            bs = bertscore.score(best_preds, refs_per_q)
            existing["BERTScore"] = (sum(bs) / len(bs) * 100) if bs else 0.0
            lo, hi = bootstrap_ci(bs, bootstrap_samples, confidence_level, seed)
            existing["BERTScore_CI"] = [lo * 100, hi * 100]
        except Exception as e:
            log.exception(f"BERTScore failed for {pred_path}: {e}")
            existing["BERTScore"] = None

    if bleurt is not None:
        try:
            bl = bleurt.score(best_preds, refs_per_q)
            existing["BLEURT"] = (sum(bl) / len(bl)) if bl else 0.0
            lo, hi = bootstrap_ci(bl, bootstrap_samples, confidence_level, seed)
            existing["BLEURT_CI"] = [lo, hi]
        except Exception as e:
            log.exception(f"BLEURT failed for {pred_path}: {e}")
            existing["BLEURT"] = None

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)
    return existing
