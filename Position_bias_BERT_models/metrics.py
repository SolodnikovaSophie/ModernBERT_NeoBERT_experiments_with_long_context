import collections
import random
import re
import string
from typing import List, Tuple, Dict, Any
from rouge_score import rouge_scorer

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
    gold_toks, pred_toks = get_tokens(gold), get_tokens(pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision, recall = num_same / len(pred_toks), num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def compute_token_recall(gold: str, pred: str) -> float:
    gold_toks, pred_toks = get_tokens(gold), get_tokens(pred)
    if not gold_toks:
        return float(not pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    return sum(common.values()) / len(gold_toks)

def compute_rouge_l(gold: str, pred: str) -> float:
    if not gold or not pred: return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(gold, pred)['rougeL'].fmeasure

def bootstrap_ci(values: List[float], n_samples: int = 1000, confidence_level: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    if not values: return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum([values[rng.randrange(n)] for _ in range(n)]) / n for _ in range(n_samples))
    alpha = 1.0 - confidence_level
    return means[int((alpha / 2) * n_samples)], means[int((1 - alpha / 2) * n_samples) - 1]

def get_best_score_for_question(predictions: List[str], gold_answers: List[str]) -> Dict[str, Any]:
    best = {"best_prediction": "", "best_gold_answer": "", "exact_match": 0.0, "f1": 0.0, "token_recall": 0.0, "rougeL": 0.0}
    for pred in predictions:
        for gold in gold_answers:
            exact = compute_exact(gold, pred)
            f1 = compute_f1(gold, pred)
            recall = compute_token_recall(gold, pred)
            rl = compute_rouge_l(gold, pred)
            if (f1, exact, rl) > (best["f1"], best["exact_match"], best["rougeL"]):
                best = {"best_prediction": pred, "best_gold_answer": gold, "exact_match": exact, "f1": f1, "token_recall": recall, "rougeL": rl}
    return best