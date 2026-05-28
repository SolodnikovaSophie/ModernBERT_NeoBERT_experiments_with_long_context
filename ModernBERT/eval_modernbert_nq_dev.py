import os
import json
import gzip
import glob
import random
import re
import string
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple, Optional
from collections import Counter

import yaml
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    Trainer,
)

# ============================================================
# Config
# ============================================================


@dataclass
class EvalConfig:
    model_name: str
    dev_file: str
    output_dir: str

    max_seq_length: int
    max_query_length: int
    doc_stride: int

    n_best_size: int
    max_answer_length: int

    per_device_eval_batch_size: int
    dataloader_num_workers: int

    seed: int = 42
    bootstrap_samples: int = 1000
    allow_no_answer: bool = True


def load_config(path: str) -> EvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Берем только нужные поля из train-config
    return EvalConfig(
        model_name=data["model_name"],
        dev_file=data["validation_file"],
        output_dir=data["output_dir"],
        max_seq_length=data["max_seq_length"],
        max_query_length=data["max_query_length"],
        doc_stride=data["doc_stride"],
        n_best_size=data["n_best_size"],
        max_answer_length=data["max_answer_length"],
        per_device_eval_batch_size=data["per_device_eval_batch_size"],
        dataloader_num_workers=data["dataloader_num_workers"],
        seed=data.get("seed", 42),
        bootstrap_samples=data.get("bootstrap_samples", 1000),
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, items: List[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# I/O
# ============================================================


def iter_jsonl_or_gz(path: str) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"

    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_nq_input_files(path: str, split: Optional[str] = None) -> List[str]:
    normalized = os.path.expanduser(path)

    if os.path.isfile(normalized):
        return [normalized]

    if os.path.isdir(normalized):
        split_key = (split or "").lower()
        split_patterns = {
            "dev": ["nq-dev-*.jsonl.gz", "nq-dev-*.jsonl", "*.jsonl.gz", "*.jsonl"],
            "validation": [
                "nq-validation-*.jsonl.gz",
                "nq-validation-*.jsonl",
                "*.jsonl.gz",
                "*.jsonl",
            ],
        }
        patterns = split_patterns.get(split_key, ["*.jsonl.gz", "*.jsonl"])

        matched: List[str] = []
        for pattern in patterns:
            matched.extend(glob.glob(os.path.join(normalized, pattern)))

        files = sorted({os.path.normpath(p) for p in matched if os.path.isfile(p)})
        if not files:
            raise FileNotFoundError(f"No input files found in: {normalized}")
        return files

    if any(ch in normalized for ch in ["*", "?", "["]):
        files = sorted(
            {os.path.normpath(p) for p in glob.glob(normalized) if os.path.isfile(p)}
        )
        if files:
            return files

    raise FileNotFoundError(f"Input path does not exist or is not readable: {path}")


# ============================================================
# DEV preprocessing
# ============================================================


def get_doc_tokens(example: Dict[str, Any]) -> List[str]:
    return [t["token"] for t in example["document_tokens"]]


def normalize_nq_eval_examples(
    example: Dict[str, Any],
    source_file: Optional[str] = None,
    source_input: Optional[str] = None,
    allow_no_answer: bool = True,
    max_seq_length: int = 2048,
) -> List[Dict[str, Any]]:
    """
    Из одного исходного DEV-примера создает несколько eval-инстансов.
    Поддерживает режим no_answer: если ответа нет, генерирует пример с пустым gold_spans
    и ссылкой на токен [CLS] (train_span = (0, 0)).
    """
    results: List[Dict[str, Any]] = []

    try:
        doc_tokens = get_doc_tokens(example)

        question_tokens = example.get("question_tokens")
        if not question_tokens:
            question_tokens = example["question_text"].split()

        original_example_id = str(example["example_id"])

        for ann_idx, ann in enumerate(example["annotations"]):
            long_answer = ann.get("long_answer", {})
            short_answers = ann.get("short_answers", [])
            yes_no_answer = ann.get("yes_no_answer", "NONE")

            # В задачах извлечения ответа (Extractive QA) мы обычно пропускаем yes/no вопросы
            if yes_no_answer != "NONE":
                continue

            # Если мы тестируем только наличие ответа, пропускаем примеры без short_answers
            if not allow_no_answer and not short_answers:
                continue

            long_start = int(long_answer.get("start_token", -1))
            long_end = int(long_answer.get("end_token", -1))

            # =======================================================
            # СЛУЧАЙ 1: Нет размеченного длинного ответа (long_answer)
            # =======================================================
            if long_start < 0 or long_end <= long_start:
                if allow_no_answer:
                    doc_length = len(doc_tokens)

                    # Если статья короче окна, берем ее целиком
                    if doc_length <= max_seq_length:
                        random_context = doc_tokens
                    # Иначе выбираем случайный кусок из документа размером с окно
                    else:
                        start_idx = random.randint(0, doc_length - max_seq_length)
                        random_context = doc_tokens[
                            start_idx : start_idx + max_seq_length
                        ]

                    results.append(
                        {
                            "example_id": f"{original_example_id}__ann{ann_idx}",
                            "original_example_id": original_example_id,
                            "annotation_idx": ann_idx,
                            "question_tokens": question_tokens,
                            "context_tokens": random_context,
                            "train_span": (
                                0,
                                0,
                            ),  # Указывает на токен [CLS] (No Answer)
                            "gold_spans": [],  # Пустой список правильных ответов
                            "source_file": source_file,
                            "source_file_name": (
                                os.path.basename(source_file) if source_file else None
                            ),
                            "source_input": source_input,
                        }
                    )
                continue

            # =======================================================
            # СЛУЧАЙ 2: Есть длинный ответ, обрабатываем короткие
            # =======================================================
            context_tokens = doc_tokens[long_start:long_end]
            if not context_tokens:
                continue

            gold_spans_local: List[Tuple[int, int]] = []
            for sa in short_answers:
                s = int(sa["start_token"]) - long_start
                e = int(sa["end_token"]) - long_start - 1
                if 0 <= s <= e < len(context_tokens):
                    gold_spans_local.append((s, e))

            # Если коротких ответов нет (или они криво размечены и вышли за границы)
            if not gold_spans_local:
                if allow_no_answer:
                    results.append(
                        {
                            "example_id": f"{original_example_id}__ann{ann_idx}",
                            "original_example_id": original_example_id,
                            "annotation_idx": ann_idx,
                            "question_tokens": question_tokens,
                            "context_tokens": context_tokens,
                            "train_span": (
                                0,
                                0,
                            ),  # Указывает на токен [CLS] (No Answer)
                            "gold_spans": [],
                            "source_file": source_file,
                            "source_file_name": (
                                os.path.basename(source_file) if source_file else None
                            ),
                            "source_input": source_input,
                        }
                    )
                continue

            # =======================================================
            # СЛУЧАЙ 3: Стандартный пример с нормальным ответом
            # =======================================================
            results.append(
                {
                    "example_id": f"{original_example_id}__ann{ann_idx}",
                    "original_example_id": original_example_id,
                    "annotation_idx": ann_idx,
                    "question_tokens": question_tokens,
                    "context_tokens": context_tokens,
                    "train_span": gold_spans_local[0],  # Для лосса берется первый спан
                    "gold_spans": gold_spans_local,  # Все спаны для подсчета метрик
                    "source_file": source_file,
                    "source_file_name": (
                        os.path.basename(source_file) if source_file else None
                    ),
                    "source_input": source_input,
                }
            )

    except (KeyError, IndexError, TypeError, ValueError):

        return []

    return results


def load_nq_eval_examples(
    path: str,
    cfg: EvalConfig,
    split: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if files is None:
        files = resolve_nq_input_files(path, split=split)

    kept: List[Dict[str, Any]] = []
    total_raw = 0

    for fp in files:
        for ex in iter_jsonl_or_gz(fp):
            total_raw += 1
            kept.extend(
                normalize_nq_eval_examples(
                    example=ex,
                    source_file=fp,
                    source_input=path,
                    allow_no_answer=cfg.allow_no_answer,  # <-- Берем из конфига
                    max_seq_length=cfg.max_seq_length,  # <-- Берем из конфига
                )
            )

    print(
        f"[DEV] files={len(files)}, raw_examples={total_raw}, eval_instances={len(kept)}"
    )
    return kept


# ============================================================
# Tokenization
# ============================================================


def _locate_answer_tokens(
    word_ids: List[Optional[int]],
    sequence_ids: List[Optional[int]],
    gold_start: int,
    gold_end: int,
) -> Optional[Tuple[int, int]]:
    feature_min_word = None
    feature_max_word = None
    token_start = None
    token_end = None

    for idx, (sid, wid) in enumerate(zip(sequence_ids, word_ids)):
        if sid != 1 or wid is None:
            continue

        if feature_min_word is None:
            feature_min_word = wid
        feature_max_word = wid

        if wid == gold_start and token_start is None:
            token_start = idx

    if feature_min_word is None or feature_max_word is None:
        return None

    if gold_start < feature_min_word or gold_end > feature_max_word:
        return None

    for idx in range(len(word_ids) - 1, -1, -1):
        if sequence_ids[idx] == 1 and word_ids[idx] == gold_end:
            token_end = idx
            break

    if token_start is None or token_end is None or token_end < token_start:
        return None

    return token_start, token_end


def build_eval_features(
    examples: List[Dict[str, Any]],
    tokenizer,
    cfg: EvalConfig,
) -> Tuple[Dataset, List[Dict[str, Any]]]:
    questions = [ex["question_tokens"][: cfg.max_query_length] for ex in examples]
    contexts = [ex["context_tokens"] for ex in examples]

    tokenized = tokenizer(
        questions,
        contexts,
        is_split_into_words=True,
        truncation="only_second",
        max_length=cfg.max_seq_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        padding=False,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    start_positions: List[int] = []
    end_positions: List[int] = []
    features_meta: List[Dict[str, Any]] = []

    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]
        ex = examples[sample_idx]

        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        word_ids = tokenized.word_ids(batch_index=i)
        sequence_ids = tokenized.sequence_ids(i)

        context_mask = [
            1 if sid == 1 and wid is not None else 0
            for sid, wid in zip(sequence_ids, word_ids)
        ]
        safe_word_ids = [-1 if w is None else int(w) for w in word_ids]

        features_meta.append(
            {
                "example_id": ex["example_id"],
                "original_example_id": ex["original_example_id"],
                "annotation_idx": ex["annotation_idx"],
                "word_ids": safe_word_ids,
                "context_mask": context_mask,
            }
        )

        gold_start, gold_end = ex["train_span"]
        answer_pos = _locate_answer_tokens(word_ids, sequence_ids, gold_start, gold_end)

        if answer_pos is None:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            token_start, token_end = answer_pos
            start_positions.append(token_start)
            end_positions.append(token_end)

    dataset = Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "start_positions": start_positions,
            "end_positions": end_positions,
        }
    )
    return dataset, features_meta


# ============================================================
# Metrics
# ============================================================


def span_to_text(span: Optional[Tuple[int, int]], context_tokens: List[str]) -> str:
    if span is None:
        return ""

    s, e = span
    if s < 0 or e < s or e >= len(context_tokens):
        return ""

    return " ".join(context_tokens[s : e + 1]).strip()


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def squad_exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def squad_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_recall_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level recall после SQuAD-style normalization.
    Recall = overlap / number_of_gold_tokens
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    return overlap / len(gold_tokens)


def squad_metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: List[str]
) -> float:
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def top_k_indices(logits: np.ndarray, k: int) -> List[int]:
    if len(logits) <= k:
        return np.argsort(logits)[::-1].tolist()
    idx = np.argpartition(logits, -k)[-k:]
    return idx[np.argsort(logits[idx])[::-1]].tolist()


def postprocess_predictions(
    examples: List[Dict[str, Any]],
    features_meta: List[Dict[str, Any]],
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    cfg: EvalConfig,
) -> Tuple[Dict[str, Optional[Tuple[int, int]]], List[Dict[str, Any]]]:
    start_logits, end_logits = raw_predictions
    best_predictions: Dict[str, Dict[str, Any]] = {}

    for feature_idx, meta in enumerate(features_meta):
        ex_id = meta["example_id"]
        word_ids = meta["word_ids"]
        context_mask = meta["context_mask"]

        s_logits = start_logits[feature_idx]
        e_logits = end_logits[feature_idx]

        start_indexes = top_k_indices(s_logits, cfg.n_best_size)
        end_indexes = top_k_indices(e_logits, cfg.n_best_size)

        for s_idx in start_indexes:
            for e_idx in end_indexes:
                if s_idx >= len(word_ids) or e_idx >= len(word_ids):
                    continue
                if context_mask[s_idx] == 0 or context_mask[e_idx] == 0:
                    continue
                if e_idx < s_idx:
                    continue

                s_word = word_ids[s_idx]
                e_word = word_ids[e_idx]

                if s_word < 0 or e_word < 0:
                    continue
                if e_word < s_word:
                    continue

                answer_len = e_word - s_word + 1
                if answer_len > cfg.max_answer_length:
                    continue

                span = (s_word, e_word)
                score = float(s_logits[s_idx] + e_logits[e_idx])

                prev = best_predictions.get(ex_id)
                if prev is None or score > prev["score"]:
                    best_predictions[ex_id] = {
                        "score": score,
                        "span": span,
                    }

    final_predictions: Dict[str, Optional[Tuple[int, int]]] = {}
    pred_details: List[Dict[str, Any]] = []

    for ex in examples:
        ex_id = ex["example_id"]
        best = best_predictions.get(ex_id)
        context_tokens = ex["context_tokens"]

        pred_span = None if best is None else best["span"]
        score = None if best is None else best["score"]
        pred_text = span_to_text(pred_span, context_tokens)

        final_predictions[ex_id] = pred_span
        pred_details.append(
            {
                "example_id": ex_id,
                "original_example_id": ex["original_example_id"],
                "annotation_idx": ex["annotation_idx"],
                "pred_span": pred_span,
                "pred_text": pred_text,
                "score": score,
                "gold_spans": ex["gold_spans"],
            }
        )

    return final_predictions, pred_details


def evaluate_predictions(
    examples: List[Dict[str, Any]],
    predictions: Dict[str, Optional[Tuple[int, int]]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped_scores: Dict[str, Dict[str, float]] = {}
    per_instance: List[Dict[str, Any]] = []

    for ex in examples:
        ex_id = ex["example_id"]
        original_example_id = ex["original_example_id"]

        pred_span = predictions.get(ex_id)
        context_tokens = ex["context_tokens"]
        gold_spans = ex["gold_spans"]

        gold_texts = [
            span_to_text(gold_span, context_tokens) for gold_span in gold_spans
        ]
        if not gold_texts:
            gold_texts = [""]

        pred_text = span_to_text(pred_span, context_tokens)

        max_em = squad_metric_max_over_ground_truths(
            squad_exact_match_score,
            pred_text,
            gold_texts,
        )
        max_f1 = squad_metric_max_over_ground_truths(
            squad_f1_score,
            pred_text,
            gold_texts,
        )
        max_recall = squad_metric_max_over_ground_truths(
            squad_recall_score,
            pred_text,
            gold_texts,
        )

        prev = grouped_scores.get(original_example_id)
        if prev is None:
            grouped_scores[original_example_id] = {
                "exact_match": max_em,
                "f1": max_f1,
                "token_recall": max_recall,
            }
        else:
            grouped_scores[original_example_id]["exact_match"] = max(
                grouped_scores[original_example_id]["exact_match"],
                max_em,
            )
            grouped_scores[original_example_id]["f1"] = max(
                grouped_scores[original_example_id]["f1"],
                max_f1,
            )
            grouped_scores[original_example_id]["token_recall"] = max(
                grouped_scores[original_example_id]["token_recall"],
                max_recall,
            )

        per_instance.append(
            {
                "example_id": ex_id,
                "original_example_id": original_example_id,
                "annotation_idx": ex["annotation_idx"],
                "pred_span": pred_span,
                "pred_text": pred_text,
                "gold_spans": gold_spans,
                "gold_texts": gold_texts,
                "exact_match": max_em,
                "f1": max_f1,
                "token_recall": max_recall,
            }
        )

    per_question = [
        {
            "original_example_id": qid,
            "exact_match": vals["exact_match"],
            "f1": vals["f1"],
            "token_recall": vals["token_recall"],
        }
        for qid, vals in grouped_scores.items()
    ]

    em_scores = [v["exact_match"] for v in grouped_scores.values()]
    f1_scores = [v["f1"] for v in grouped_scores.values()]
    recall_scores = [v["token_recall"] for v in grouped_scores.values()]

    metrics = {
        "exact_match": (float(np.mean(em_scores)) * 100.0) if em_scores else 0.0,
        "f1": (float(np.mean(f1_scores)) * 100.0) if f1_scores else 0.0,
        "token_recall": (
            (float(np.mean(recall_scores)) * 100.0) if recall_scores else 0.0
        ),
        "n_examples": len(grouped_scores),
        "n_eval_instances": len(examples),
    }
    return metrics, per_instance, per_question


# ============================================================
# Bootstrap confidence intervals
# ============================================================


def bootstrap_confidence_intervals(
    per_question: List[Dict[str, Any]],
    n_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap CI по исходным вопросам.
    """
    rng = np.random.default_rng(seed)

    em_values = np.array([row["exact_match"] for row in per_question], dtype=np.float64)
    f1_values = np.array([row["f1"] for row in per_question], dtype=np.float64)
    recall_values = np.array(
        [row["token_recall"] for row in per_question], dtype=np.float64
    )

    n = len(per_question)
    if n == 0:
        return {
            "bootstrap_samples": n_samples,
            "confidence_level": 0.95,
            "exact_match_ci": [0.0, 0.0],
            "f1_ci": [0.0, 0.0],
            "token_recall_ci": [0.0, 0.0],
        }

    em_boot = []
    f1_boot = []
    recall_boot = []

    for _ in range(n_samples):
        sample_idx = rng.integers(0, n, size=n)
        em_boot.append(float(np.mean(em_values[sample_idx])) * 100.0)
        f1_boot.append(float(np.mean(f1_values[sample_idx])) * 100.0)
        recall_boot.append(float(np.mean(recall_values[sample_idx])) * 100.0)

    em_ci = [float(np.percentile(em_boot, 2.5)), float(np.percentile(em_boot, 97.5))]
    f1_ci = [float(np.percentile(f1_boot, 2.5)), float(np.percentile(f1_boot, 97.5))]
    recall_ci = [
        float(np.percentile(recall_boot, 2.5)),
        float(np.percentile(recall_boot, 97.5)),
    ]

    return {
        "bootstrap_samples": n_samples,
        "confidence_level": 0.95,
        "exact_match_ci": em_ci,
        "f1_ci": f1_ci,
        "token_recall_ci": recall_ci,
    }


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Путь к YAML-конфигу")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Путь к обученной модели или checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Куда сохранить результаты eval"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    output_dir = args.output_dir or os.path.join(cfg.output_dir, "dev_eval")
    ensure_dir(output_dir)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)

    dev_files = resolve_nq_input_files(cfg.dev_file, split="dev")
    save_json(os.path.join(output_dir, "dev_input_files.json"), dev_files)

    print(f"Loading DEV examples (allow_no_answer={cfg.allow_no_answer})...")

    eval_examples = load_nq_eval_examples(
        path=cfg.dev_file,
        cfg=cfg,
        split="dev",
        files=dev_files,
    )

    print("Tokenizing DEV set...")
    eval_dataset, eval_features_meta = build_eval_features(
        eval_examples, tokenizer, cfg
    )

    print(f"DEV eval instances: {len(eval_examples)}")
    print(f"DEV features: {len(eval_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
    )

    print("Running prediction on DEV...")
    pred_output = trainer.predict(
        eval_dataset,
        metric_key_prefix="dev",
    )

    predictions, pred_details = postprocess_predictions(
        examples=eval_examples,
        features_meta=eval_features_meta,
        raw_predictions=pred_output.predictions,
        cfg=cfg,
    )

    metrics, per_instance, per_question = evaluate_predictions(
        eval_examples, predictions
    )
    ci = bootstrap_confidence_intervals(
        per_question=per_question,
        n_samples=cfg.bootstrap_samples,
        seed=cfg.seed,
    )

    metrics_with_ci = {**metrics, **ci}

    save_json(os.path.join(output_dir, "dev_metrics.json"), metrics_with_ci)
    save_json(
        os.path.join(output_dir, "dev_predictions.json"),
        {k: list(v) if v is not None else None for k, v in predictions.items()},
    )
    save_json(os.path.join(output_dir, "dev_pred_details.json"), pred_details)
    save_json(os.path.join(output_dir, "dev_per_instance.json"), per_instance)
    save_jsonl(os.path.join(output_dir, "dev_per_instance.jsonl"), per_instance)
    save_json(os.path.join(output_dir, "dev_per_question.json"), per_question)
    save_jsonl(os.path.join(output_dir, "dev_per_question.jsonl"), per_question)

    print("Done.")
    print(json.dumps(metrics_with_ci, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
