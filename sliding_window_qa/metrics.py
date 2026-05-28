"""
Метрики:

1. Reranking (по запросу):
   - MRR  — 1/rank позиции первого чанка с label=1
            (если положительных нет — вклад 0).
   - Recall@K для K ∈ {1, 3, 5}
            — 1, если в топ-K есть хотя бы один чанк с label=1, иначе 0.

   Затем усреднение по уникальным вопросам.

2. Span-метрики (между текстом топ-1 чанка и gold short answer'ом):
   - Exact Match
   - F1
   - Token Recall

3. Bootstrap CI (перцентильный) по списку per-query значений.
"""

import collections
import random
import re
import string
from typing import Dict, List, Tuple, Iterable


# =============================================================
# Текстовые утилиты (как в SQuAD-eval)
# =============================================================

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split() if s else []


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    gt, pt = get_tokens(gold), get_tokens(pred)
    if not gt or not pt:
        return float(gt == pt)
    common = collections.Counter(gt) & collections.Counter(pt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pt), same / len(gt)
    return 2 * p * r / (p + r)


def compute_token_recall(gold: str, pred: str) -> float:
    gt, pt = get_tokens(gold), get_tokens(pred)
    if not gt:
        return float(not pt)
    common = collections.Counter(gt) & collections.Counter(pt)
    return sum(common.values()) / len(gt)


# =============================================================
# Per-query ranking
# =============================================================

def rank_chunks(chunk_records: List[Dict]) -> List[Dict]:
    """
    Сортировка по убыванию score. На вход — список dict с ключами
    'score', 'label', 'chunk_id', 'text'.
    """
    return sorted(chunk_records, key=lambda x: x["score"], reverse=True)


def reciprocal_rank(ranked: List[Dict]) -> float:
    for i, ch in enumerate(ranked, start=1):
        if ch["label"] == 1:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: List[Dict], k: int) -> float:
    top = ranked[:k]
    return float(any(ch["label"] == 1 for ch in top))


def best_span_metrics_for_query(
    ranked: List[Dict],
    gold_texts: List[str],
) -> Dict[str, float]:
    """
    «Coarse» span-метрики: сравниваем ТЕКСТ ТОП-1 чанка с лучшим gold.
    Это верхняя граница «ответ есть где-то в топ-1 чанке».
    """
    if not ranked or not gold_texts:
        return {"exact_match": 0.0, "f1": 0.0, "token_recall": 0.0, "top1_text": ""}

    top1_text = ranked[0]["text"]
    best = {"exact_match": 0.0, "f1": 0.0, "token_recall": 0.0}
    for g in gold_texts:
        em = compute_exact(g, top1_text)
        f1 = compute_f1(g, top1_text)
        rc = compute_token_recall(g, top1_text)
        if (f1, em, rc) > (best["f1"], best["exact_match"], best["token_recall"]):
            best = {"exact_match": float(em), "f1": float(f1), "token_recall": float(rc)}
    best["top1_text"] = top1_text
    return best


def extracted_span_metrics_for_query(
    predicted_span_text: str,
    gold_texts: List[str],
) -> Dict[str, float]:
    """
    Честные span-метрики: сравниваем строку, извлечённую QA-моделью
    (stage-2), с лучшим gold-ответом.
    """
    if not gold_texts:
        return {"exact_match": 0.0, "f1": 0.0, "token_recall": 0.0}

    best = {"exact_match": 0.0, "f1": 0.0, "token_recall": 0.0}
    for g in gold_texts:
        em = compute_exact(g, predicted_span_text)
        f1 = compute_f1(g, predicted_span_text)
        rc = compute_token_recall(g, predicted_span_text)
        if (f1, em, rc) > (best["f1"], best["exact_match"], best["token_recall"]):
            best = {"exact_match": float(em), "f1": float(f1), "token_recall": float(rc)}
    return best


# =============================================================
# Bootstrap CI
# =============================================================

def bootstrap_ci(
    values: List[float],
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_samples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    alpha = 1.0 - confidence_level
    lo = means[int((alpha / 2) * n_samples)]
    hi = means[max(0, int((1 - alpha / 2) * n_samples) - 1)]
    return lo, hi


# =============================================================
# Aggregation across queries
# =============================================================

def aggregate(
    per_query: Dict[str, Dict[str, float]],
    keys: Iterable[str],
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Усредняет указанные ключи по per_query, считает bootstrap-CI.
    """
    if not per_query:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [v[key] for v in per_query.values() if key in v]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        lo, hi = bootstrap_ci(vals, bootstrap_samples, confidence_level, seed)
        out[key] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}

    return out
