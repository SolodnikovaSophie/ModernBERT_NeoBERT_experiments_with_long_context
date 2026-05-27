#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Обучение ModernBERT модели для extractive QA на Natural Questions.
Поддерживает SQuAD 2.0 формат (ответ может отсутствовать в отрывке).

Оптимизации для GPU-сервера / H100:
- progress bar при чтении файлов;
- логирование GPU/CUDA перед обучением;
- bf16/tf32-friendly режим;
- fused AdamW optimizer;
- pin_memory / persistent_workers для DataLoader;
- быстрый evaluate(): один predict-pass вместо evaluate + predict;
- SQuAD 2.0 null-score postprocessing.
"""

from __future__ import annotations

import argparse
import gzip
import glob
import json
import os
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import yaml
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# ============================================================
# Config
# ============================================================


@dataclass
class Config:
    seed: int
    model_name: str
    train_file: str
    validation_file: str
    output_dir: str

    allow_unanswerable: bool  # <-- Новый параметр для экспериментов

    max_seq_length: int
    max_query_length: int
    doc_stride: int

    n_best_size: int
    max_answer_length: int

    learning_rate: float
    weight_decay: float
    num_train_epochs: int

    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float

    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int

    bf16: bool
    fp16: bool
    gradient_checkpointing: bool

    dataloader_num_workers: int
    report_to: str

    optim: str = "adamw_torch_fused"
    tf32: bool = True


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.detach().cpu().tolist()
    return obj


def log_gpu_info() -> None:
    print("=" * 80)
    print("Runtime / GPU info")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device name: {torch.cuda.get_device_name(current_device)}")
    print("=" * 80)


# ============================================================
# I/O & Preprocessing
# ============================================================


def iter_jsonl_or_gz(path: str) -> Iterator[Dict[str, Any]]:
    """
    Потоково читает jsonl / jsonl.gz файл.
    Добавлена защита от битых архивов (EOFError, BadGzipFile).
    """
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"

    try:
        with opener(path, mode, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except (EOFError, gzip.BadGzipFile) as e:
        print(f"\n[WARNING] Пропущен поврежденный архив: {path}")
        print(f"[WARNING] Ошибка: {e}\n")
    except Exception as e:
        print(f"\n[ERROR] Ошибка чтения файла {path}: {e}\n")


def resolve_nq_input_files(path: str, split: Optional[str] = None) -> List[str]:
    normalized = os.path.expanduser(path)
    if os.path.isfile(normalized):
        return [normalized]
    if os.path.isdir(normalized):
        patterns = ["*.jsonl.gz", "*.jsonl"]
        matched: List[str] = []
        for pattern in patterns:
            matched.extend(glob.glob(os.path.join(normalized, pattern)))
        files = sorted({os.path.normpath(p) for p in matched if os.path.isfile(p)})
        if files:
            return files
    if any(ch in normalized for ch in ["*", "?", "["]):
        return sorted(
            {os.path.normpath(p) for p in glob.glob(normalized) if os.path.isfile(p)}
        )
    raise FileNotFoundError(f"Input path does not exist: {path}")


def get_doc_tokens(example: Dict[str, Any]) -> List[str]:
    return [t["token"] for t in example["document_tokens"]]


def normalize_nq_example(
    example: Dict[str, Any],
    allow_unanswerable: bool,
    source_file: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        ann = example["annotations"][0]
        long_answer = ann["long_answer"]
        short_answers = ann["short_answers"]

        long_start = int(long_answer["start_token"])
        long_end = int(long_answer["end_token"])

        doc_tokens = get_doc_tokens(example)
        context_tokens = doc_tokens[long_start:long_end]
        if not context_tokens:
            return None

        gold_spans_local: List[Tuple[int, int]] = []
        for sa in short_answers:
            s = int(sa["start_token"]) - long_start
            e = int(sa["end_token"]) - long_start - 1
            if 0 <= s <= e < len(context_tokens):
                gold_spans_local.append((s, e))

        # Логика SQuAD 2.0
        if not gold_spans_local:
            if not allow_unanswerable:
                return None
            train_span = (-1, -1)
        else:
            first_short = gold_spans_local[0]
            train_span = first_short

        question_tokens = example.get("question_tokens")
        if not question_tokens:
            question_tokens = example["question_text"].split()

        return {
            "example_id": str(example["example_id"]),
            "question_tokens": question_tokens,
            "context_tokens": context_tokens,
            "train_span": train_span,
            "gold_spans": gold_spans_local,
            "source_file_name": os.path.basename(source_file) if source_file else None,
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def load_nq_examples(
    path: str,
    allow_unanswerable: bool,
    split: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if files is None:
        files = resolve_nq_input_files(path, split=split)

    kept: List[Dict[str, Any]] = []
    split_name = split or "DATA"

    for fp in tqdm(files, desc=f"Loading {split_name} files", unit="file"):
        for ex in iter_jsonl_or_gz(fp):
            norm = normalize_nq_example(ex, allow_unanswerable, source_file=fp)
            if norm is not None:
                kept.append(norm)
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
    # Если ответ отсутствует (unanswerable)
    if gold_start == -1 and gold_end == -1:
        return None

    feature_min_word, feature_max_word = None, None
    token_start, token_end = None, None

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


def build_train_features(
    examples: List[Dict[str, Any]], tokenizer, cfg: Config
) -> Dataset:
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
    start_positions, end_positions = [], []

    for i in tqdm(range(len(tokenized["input_ids"])), desc="Building train features"):
        sample_idx = sample_mapping[i]
        ex = examples[sample_idx]
        cls_index = tokenized["input_ids"][i].index(tokenizer.cls_token_id)
        word_ids = tokenized.word_ids(batch_index=i)
        sequence_ids = tokenized.sequence_ids(i)

        gold_start, gold_end = ex["train_span"]
        answer_pos = _locate_answer_tokens(word_ids, sequence_ids, gold_start, gold_end)

        # Здесь реализован SQuAD 2.0 трюк: если нет ответа, указываем на [CLS]
        if answer_pos is None:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_positions.append(answer_pos[0])
            end_positions.append(answer_pos[1])

    return Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "start_positions": start_positions,
            "end_positions": end_positions,
        }
    )


def build_eval_features(
    examples: List[Dict[str, Any]], tokenizer, cfg: Config
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
    features_meta = []

    for i in tqdm(range(len(tokenized["input_ids"])), desc="Building eval features"):
        sample_idx = sample_mapping[i]
        word_ids = tokenized.word_ids(batch_index=i)
        sequence_ids = tokenized.sequence_ids(i)

        context_mask = [
            1 if sid == 1 and wid is not None else 0
            for sid, wid in zip(sequence_ids, word_ids)
        ]
        safe_word_ids = [-1 if w is None else int(w) for w in word_ids]

        features_meta.append(
            {
                "example_id": examples[sample_idx]["example_id"],
                "word_ids": safe_word_ids,
                "context_mask": context_mask,
            }
        )

    dataset = Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
    )
    return dataset, features_meta


# ============================================================
# Metrics & Postprocessing
# ============================================================


def span_to_text(span: Optional[Tuple[int, int]], context_tokens: List[str]) -> str:
    if span is None:
        return ""
    s, e = span
    if s < 0 or e < s or e >= len(context_tokens):
        return ""
    return " ".join(context_tokens[s : e + 1]).strip()


def normalize_answer(text: str) -> str:
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        return "".join(ch for ch in value if ch not in set(string.punctuation))

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


class MetricsLoggingCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.logs_dir = os.path.join(output_dir, "training_logs")
        self.metrics_jsonl = os.path.join(self.logs_dir, "metrics_log.jsonl")

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ensure_dir(self.logs_dir)
            with open(self.metrics_jsonl, "w", encoding="utf-8"):
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            entry = {
                "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "global_step": int(state.global_step),
                "logs": to_jsonable(logs),
            }
            append_jsonl(self.metrics_jsonl, entry)


class NQSpanTrainer(Trainer):
    def __init__(
        self, *args, eval_examples=None, eval_features_meta=None, cfg=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.eval_features_meta = eval_features_meta
        self.cfg = cfg

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        pred_output = self.predict(
            eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        start_logits, end_logits = pred_output.predictions

        # --- SQuAD 2.0 Postprocessing ---
        example_scores = (
            {}
        )  # example_id -> {"best_span": span, "best_score": score, "min_null_score": score}

        for i, meta in enumerate(self.eval_features_meta):
            ex_id = meta["example_id"]
            if ex_id not in example_scores:
                example_scores[ex_id] = {
                    "best_span": None,
                    "best_score": -1e30,
                    "min_null_score": 1e30,
                }

            s_logits, e_logits = start_logits[i], end_logits[i]
            word_ids, context_mask = meta["word_ids"], meta["context_mask"]

            # Null score для текущей фичи (вероятность того, что ответ в [CLS] токене)
            feature_null_score = s_logits[0] + e_logits[0]
            if feature_null_score < example_scores[ex_id]["min_null_score"]:
                example_scores[ex_id]["min_null_score"] = feature_null_score

            for s_idx in top_k_indices(s_logits, self.cfg.n_best_size):
                if s_idx >= len(context_mask) or context_mask[s_idx] == 0:
                    continue
                for e_idx in range(
                    s_idx, min(s_idx + self.cfg.max_answer_length, len(e_logits))
                ):
                    if e_idx >= len(context_mask) or context_mask[e_idx] == 0:
                        continue
                    if (
                        word_ids[s_idx] == -1
                        or word_ids[e_idx] == -1
                        or word_ids[e_idx] < word_ids[s_idx]
                    ):
                        continue

                    score = s_logits[s_idx] + e_logits[e_idx]
                    if score > example_scores[ex_id]["best_score"]:
                        example_scores[ex_id]["best_score"] = score
                        example_scores[ex_id]["best_span"] = (
                            word_ids[s_idx],
                            word_ids[e_idx],
                        )

        # Применяем Null Score порог
        predictions = {}
        for ex_id, scores in example_scores.items():
            if scores["min_null_score"] >= scores["best_score"]:
                predictions[ex_id] = None  # Ответа нет
            else:
                predictions[ex_id] = scores["best_span"]

        # --- Metrics Calculation ---
        em_scores, f1_scores = [], []
        for ex in self.eval_examples:
            pred_span = predictions.get(ex["example_id"])
            context_tokens, gold_spans = ex["context_tokens"], ex["gold_spans"]

            gold_texts = [span_to_text(gs, context_tokens) for gs in gold_spans]
            if not gold_texts:
                gold_texts = [""]  # Важно для корректного расчета пустого ответа

            pred_text = span_to_text(pred_span, context_tokens)

            em_scores.append(
                squad_metric_max_over_ground_truths(
                    squad_exact_match_score, pred_text, gold_texts
                )
            )
            f1_scores.append(
                squad_metric_max_over_ground_truths(
                    squad_f1_score, pred_text, gold_texts
                )
            )

        metrics = dict(pred_output.metrics)
        metrics[f"{metric_key_prefix}_exact_match"] = (
            float(np.mean(em_scores)) * 100.0 if em_scores else 0.0
        )
        metrics[f"{metric_key_prefix}_f1"] = (
            float(np.mean(f1_scores)) * 100.0 if f1_scores else 0.0
        )

        # Save lightweight eval summary
        step = int(self.state.global_step)
        out_dir = os.path.join(self.args.output_dir, "eval_outputs")
        ensure_dir(out_dir)
        save_json(
            os.path.join(out_dir, f"{metric_key_prefix}_metrics_step_{step}.json"),
            to_jsonable(metrics),
        )
        self.log(metrics)
        return metrics


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg.output_dir)
    save_json(os.path.join(cfg.output_dir, "used_config.json"), vars(cfg))

    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
    log_gpu_info()

    print("Loading tokenizer and ModernBERT AutoModelForQuestionAnswering...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_files = resolve_nq_input_files(cfg.train_file, split="train")
    eval_files = resolve_nq_input_files(cfg.validation_file, split="dev")

    print(f"Loading train examples (allow_unanswerable={cfg.allow_unanswerable})...")
    train_examples = load_nq_examples(
        cfg.train_file, cfg.allow_unanswerable, split="train", files=train_files
    )

    print(
        f"Loading validation examples (allow_unanswerable={cfg.allow_unanswerable})..."
    )
    eval_examples = load_nq_examples(
        cfg.validation_file, cfg.allow_unanswerable, split="dev", files=eval_files
    )

    print("Tokenizing features...")
    train_dataset = build_train_features(train_examples, tokenizer, cfg)
    eval_dataset, eval_features_meta = build_eval_features(
        eval_examples, tokenizer, cfg
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        tf32=cfg.tf32,
        optim=cfg.optim,
        report_to=cfg.report_to,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=cfg.dataloader_num_workers > 0,
    )

    trainer = NQSpanTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        eval_examples=eval_examples,
        eval_features_meta=eval_features_meta,
        cfg=cfg,
        callbacks=[MetricsLoggingCallback(cfg.output_dir)],
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    final_metrics = trainer.evaluate()

    print("Saving model...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    save_json(os.path.join(cfg.output_dir, "final_eval_metrics.json"), final_metrics)

    print("Done!")


if __name__ == "__main__":
    main()
