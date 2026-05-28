#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Обучение NeoBERT / encoder-only модели для extractive QA на Natural Questions.

Версия оптимизирована под уже очищенный span-only NQ-датасет:
- HTML уже удален;
- YES/NO уже удалены;
- примеры без long answer уже удалены;
- примеры без short answer уже удалены.

Оптимизации для GPU-сервера / H100:
- progress bar при чтении файлов;
- логирование GPU/CUDA перед обучением;
- bf16/tf32-friendly режим;
- fused AdamW optimizer;
- pin_memory / persistent_workers для DataLoader;
- быстрый evaluate(): один predict-pass вместо evaluate + predict;
- сохранение только легких eval metrics JSON/JSONL;
- без тяжелых predictions/per_example JSON на каждом eval;
- без substep JSONL логирования.
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
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

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

    # H100 / Trainer options.
    # Эти поля необязательные: старый YAML без них продолжит работать.
    optim: str = "adamw_torch_fused"
    tf32: bool = True


def load_config(path: str) -> Config:
    """Загружает конфигурацию обучения из YAML-файла."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def ensure_dir(path: str) -> None:
    """Создает папку, если она не существует."""
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    """Сохраняет объект в JSON."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any) -> None:
    """Дописывает объект в JSONL-файл одной строкой."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def to_jsonable(obj: Any) -> Any:
    """Рекурсивно приводит объект к JSON-совместимому виду."""
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
    """
    Печатает информацию о CUDA/GPU-устройствах.

    Полезно перед обучением на сервере:
    - видно, какую GPU получил job;
    - видно, доступен ли bf16;
    - видно, сколько всего GPU обнаружено;
    - можно заметить, если обучение случайно запустилось на CPU.
    """
    print("=" * 80)
    print("Runtime / GPU info")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device available. Training will run on CPU.")
        print("=" * 80)
        return

    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    for device_idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_idx)
        total_gb = props.total_memory / 1024**3
        print(
            f"GPU {device_idx}: {props.name} | "
            f"capability={props.major}.{props.minor} | "
            f"total_memory={total_gb:.2f} GB"
        )

    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {current_device}")
    print(f"Current CUDA device name: {torch.cuda.get_device_name(current_device)}")
    print("=" * 80)


# ============================================================
# I/O
# ============================================================


def iter_jsonl_or_gz(path: str) -> Iterator[Dict[str, Any]]:
    """
    Потоково читает jsonl / jsonl.gz файл.

    Это быстрее и экономнее по памяти, чем сначала читать весь файл в список.
    """
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"

    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_nq_input_files(path: str, split: Optional[str] = None) -> List[str]:
    """
    Преобразует входной путь в список файлов датасета NQ.

    Поддерживает:
    - путь к одному файлу;
    - директорию с шардированными файлами;
    - glob-паттерн.
    """
    normalized = os.path.expanduser(path)

    if os.path.isfile(normalized):
        return [normalized]

    if os.path.isdir(normalized):
        split_key = (split or "").lower()
        split_patterns = {
            "train": ["nq-train-*.jsonl.gz", "nq-train-*.jsonl"],
            "dev": ["nq-dev-*.jsonl.gz", "nq-dev-*.jsonl"],
            "validation": [
                "nq-dev-*.jsonl.gz",
                "nq-dev-*.jsonl",
                "nq-validation-*.jsonl.gz",
                "nq-validation-*.jsonl",
            ],
        }
        generic_patterns = ["*.jsonl.gz", "*.jsonl"]
        patterns = split_patterns.get(split_key, generic_patterns)

        matched: List[str] = []
        for pattern in patterns:
            matched.extend(glob.glob(os.path.join(normalized, pattern)))

        if not matched and patterns != generic_patterns:
            for pattern in generic_patterns:
                matched.extend(glob.glob(os.path.join(normalized, pattern)))

        files = sorted({os.path.normpath(p) for p in matched if os.path.isfile(p)})
        if not files:
            raise FileNotFoundError(
                f"No input files found in directory: {normalized} "
                f"(split={split_key or 'any'})"
            )
        return files

    if any(ch in normalized for ch in ["*", "?", "["]):
        files = sorted(
            {os.path.normpath(p) for p in glob.glob(normalized) if os.path.isfile(p)}
        )
        if files:
            return files

    raise FileNotFoundError(f"Input path does not exist or is not readable: {path}")


# ============================================================
# NQ preprocessing for already-cleaned span-only data
# ============================================================


def get_doc_tokens(example: Dict[str, Any]) -> List[str]:
    """
    Датасет уже очищен, поэтому просто берем token у каждого document_token.

    Никакой повторной фильтрации html_token не делаем.
    """
    return [t["token"] for t in example["document_tokens"]]


def normalize_nq_example(
    example: Dict[str, Any],
    source_file: Optional[str] = None,
    source_input: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Нормализует уже очищенный пример.

    Логика:
    - берет первую аннотацию;
    - берет long answer как context;
    - переводит short answer из глобальных координат документа
      в локальные координаты внутри long answer;
    - сохраняет все gold spans в локальных координатах.

    Ожидается, что пример уже очищен:
    - annotations есть;
    - yes_no_answer = NONE;
    - long_answer валиден;
    - short_answers не пустой.
    """
    try:
        ann = example["annotations"][0]
        long_answer = ann["long_answer"]
        short_answers = ann["short_answers"]

        long_start = int(long_answer["start_token"])
        long_end = int(long_answer["end_token"])  # exclusive

        doc_tokens = get_doc_tokens(example)
        context_tokens = doc_tokens[long_start:long_end]
        if not context_tokens:
            return None

        first_short = short_answers[0]
        short_start_global = int(first_short["start_token"])
        short_end_global_exclusive = int(first_short["end_token"])

        short_start_local = short_start_global - long_start
        short_end_local_inclusive = short_end_global_exclusive - long_start - 1

        if short_start_local < 0 or short_end_local_inclusive < short_start_local:
            return None
        if short_end_local_inclusive >= len(context_tokens):
            return None

        gold_spans_local: List[Tuple[int, int]] = []
        for sa in short_answers:
            s = int(sa["start_token"]) - long_start
            e = int(sa["end_token"]) - long_start - 1
            if 0 <= s <= e < len(context_tokens):
                gold_spans_local.append((s, e))

        if not gold_spans_local:
            return None

        question_tokens = example.get("question_tokens")
        if not question_tokens:
            question_tokens = example["question_text"].split()

        return {
            "example_id": str(example["example_id"]),
            "question_tokens": question_tokens,
            "context_tokens": context_tokens,
            "train_span": (short_start_local, short_end_local_inclusive),
            "gold_spans": gold_spans_local,
            "source_file": source_file,
            "source_file_name": os.path.basename(source_file) if source_file else None,
            "source_input": source_input,
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def load_nq_examples(
    path: str,
    split: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает примеры из уже очищенного NQ-датасета.

    Добавлен progress bar по файлам и текущая статистика:
    - сколько файлов уже прочитано;
    - сколько всего примеров просмотрено;
    - сколько примеров оставлено;
    - сколько примеров отброшено как invalid.
    """
    if files is None:
        files = resolve_nq_input_files(path, split=split)

    kept: List[Dict[str, Any]] = []
    total = 0
    invalid = 0
    split_name = split or "DATA"

    print(f"[{split_name}] input path: {path}")
    print(f"[{split_name}] files found: {len(files)}")

    for fp in tqdm(files, desc=f"Loading {split_name} files", unit="file"):
        file_total = 0
        file_kept = 0
        file_invalid = 0

        for ex in iter_jsonl_or_gz(fp):
            total += 1
            file_total += 1

            norm = normalize_nq_example(
                ex,
                source_file=fp,
                source_input=path,
            )

            if norm is None:
                invalid += 1
                file_invalid += 1
                continue

            kept.append(norm)
            file_kept += 1

        tqdm.write(
            f"[{split_name}] {os.path.basename(fp)}: "
            f"total={file_total}, kept={file_kept}, invalid={file_invalid}"
        )

    print(
        f"[{split_name}] DONE: "
        f"files={len(files)}, total={total}, kept={len(kept)}, invalid={invalid}"
    )

    return kept


# ============================================================
# Tokenization helpers
# ============================================================


def _locate_answer_tokens(
    word_ids: List[Optional[int]],
    sequence_ids: List[Optional[int]],
    gold_start: int,
    gold_end: int,
) -> Optional[Tuple[int, int]]:
    """
    Находит token-level start/end позиции ответа в пределах feature-окна.

    Возвращает:
    - (token_start, token_end), если span попал в окно;
    - None, если span не попал в окно или определить его нельзя.
    """
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


# ============================================================
# Tokenization
# ============================================================


def build_train_features(
    examples: List[Dict[str, Any]],
    tokenizer,
    cfg: Config,
) -> Dataset:
    """Строит train features для extractive QA."""
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

    for i in tqdm(
        range(len(tokenized["input_ids"])), desc="Building train labels", unit="feature"
    ):
        sample_idx = sample_mapping[i]
        ex = examples[sample_idx]

        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        word_ids = tokenized.word_ids(batch_index=i)
        sequence_ids = tokenized.sequence_ids(i)

        gold_start, gold_end = ex["train_span"]
        answer_pos = _locate_answer_tokens(word_ids, sequence_ids, gold_start, gold_end)

        if answer_pos is None:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            token_start, token_end = answer_pos
            start_positions.append(token_start)
            end_positions.append(token_end)

    return Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "start_positions": start_positions,
            "end_positions": end_positions,
        }
    )


def build_eval_features(
    examples: List[Dict[str, Any]],
    tokenizer,
    cfg: Config,
) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """Строит eval features и метаданные для постобработки."""
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

    for i in tqdm(
        range(len(tokenized["input_ids"])),
        desc="Building eval labels/meta",
        unit="feature",
    ):
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
    """Преобразует inclusive token span в plain-text ответ."""
    if span is None:
        return ""

    s, e = span
    if s < 0 or e < s or e >= len(context_tokens):
        return ""

    return " ".join(context_tokens[s : e + 1]).strip()


def normalize_answer(text: str) -> str:
    """
    SQuAD-style normalization:
    lowercasing, punctuation/article removal, whitespace cleanup.
    """

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


def squad_metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: List[str]
) -> float:
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def top_k_indices(logits: np.ndarray, k: int) -> List[int]:
    """Возвращает индексы top-k логитов по убыванию."""
    if len(logits) <= k:
        return np.argsort(logits)[::-1].tolist()
    idx = np.argpartition(logits, -k)[-k:]
    return idx[np.argsort(logits[idx])[::-1]].tolist()


import numpy as np


def postprocess_predictions(
    examples,
    features,
    raw_start_logits,
    raw_end_logits,
    tokenizer,
    max_answer_length=30,
    n_best_size=20,
):
    """
    Safe post-processing for span QA (SQuAD-style).
    Fixes:
    - feature-level vs example-level mismatch
    - word_ids out-of-range
    - incorrect boolean masking
    """

    example_id_to_index = {k["id"]: i for i, k in enumerate(examples)}
    features_per_example = {}

    for i, feature in enumerate(features):
        example_index = example_id_to_index[feature["example_id"]]
        features_per_example.setdefault(example_index, []).append(i)

    predictions = []

    for example_index, example in enumerate(examples):
        context = example["context"]
        feature_indices = features_per_example.get(example_index, [])

        best_answer = ""
        best_score = -1e30

        for feature_index in feature_indices:
            start_logits = raw_start_logits[feature_index]
            end_logits = raw_end_logits[feature_index]

            offsets = features[feature_index]["offset_mapping"]
            word_ids = features[feature_index]["word_ids"]

            # convert to numpy safely
            start_logits = np.array(start_logits)
            end_logits = np.array(end_logits)

            # top start indices
            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]

            for start_index in start_indexes:
                if start_index >= len(offsets):
                    continue
                if word_ids[start_index] is None:
                    continue

                for end_index in range(
                    start_index, min(start_index + max_answer_length, len(offsets))
                ):
                    if end_index >= len(offsets):
                        continue
                    if word_ids[end_index] is None:
                        continue

                    # must be same word span
                    if word_ids[start_index] != word_ids[end_index]:
                        continue

                    score = start_logits[start_index] + end_logits[end_index]

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]

                    if start_char is None or end_char is None:
                        continue

                    answer = context[start_char:end_char].strip()

                    if score > best_score:
                        best_score = score
                        best_answer = answer

        predictions.append({"id": example["id"], "prediction_text": best_answer})

    return predictions


def evaluate_predictions_metrics_only(
    examples: List[Dict[str, Any]],
    predictions: Dict[str, Optional[Tuple[int, int]]],
) -> Dict[str, float]:
    """Считает только EM/F1 без сохранения per-example детализации."""
    em_scores = []
    f1_scores = []

    for ex in examples:
        ex_id = ex["example_id"]
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

        em_scores.append(max_em)
        f1_scores.append(max_f1)

    return {
        "exact_match": (float(np.mean(em_scores)) * 100.0) if em_scores else 0.0,
        "f1": (float(np.mean(f1_scores)) * 100.0) if f1_scores else 0.0,
        "n_examples": len(examples),
    }


# ============================================================
# Custom Trainer
# ============================================================


class MetricsLoggingCallback(TrainerCallback):
    """
    Сохраняет основные trainer logs.

    Файлы:
    - training_logs/metrics_log.jsonl:
      потоковый лог train/eval метрик;
    - training_logs/log_history.json:
      полная история Trainer в конце обучения.

    Substep-логирование специально убрано, чтобы не создавать большой JSONL-файл
    при gradient accumulation.
    """

    def __init__(self, output_dir: str):
        self.logs_dir = os.path.join(output_dir, "training_logs")
        self.metrics_jsonl = os.path.join(self.logs_dir, "metrics_log.jsonl")
        self.log_history_json = os.path.join(self.logs_dir, "log_history.json")

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        ensure_dir(self.logs_dir)

        with open(self.metrics_jsonl, "w", encoding="utf-8"):
            pass

        save_json(self.log_history_json, [])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return

        entry = {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "global_step": int(state.global_step),
            "epoch": None if state.epoch is None else float(state.epoch),
            "logs": to_jsonable(logs),
        }

        append_jsonl(self.metrics_jsonl, entry)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        save_json(self.log_history_json, to_jsonable(state.log_history))


class NQSpanTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_examples=None,
        eval_features_meta=None,
        eval_input_path=None,
        eval_input_files=None,
        cfg=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.eval_features_meta = eval_features_meta
        self.eval_input_path = eval_input_path
        self.eval_input_files = eval_input_files or []
        self.cfg = cfg

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        pred_output = self.predict(
            eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        metrics = dict(pred_output.metrics)

        # logits
        start_logits, end_logits = pred_output.predictions

        # ====== POSTPROCESS ======
        predictions = {}

        for i, meta in enumerate(self.eval_features_meta):
            example_id = meta["example_id"]
            word_ids = meta["word_ids"]
            context_mask = meta["context_mask"]

            s_logits = start_logits[i]
            e_logits = end_logits[i]

            best_score = -1e30
            best_span = None

            for start_idx in top_k_indices(s_logits, self.cfg.n_best_size):
                if start_idx >= len(context_mask):
                    continue
                if context_mask[start_idx] == 0:
                    continue

                for end_idx in range(
                    start_idx,
                    min(start_idx + self.cfg.max_answer_length, len(e_logits)),
                ):
                    if end_idx >= len(context_mask):
                        continue
                    if context_mask[end_idx] == 0:
                        continue

                    if word_ids[start_idx] == -1 or word_ids[end_idx] == -1:
                        continue

                    if word_ids[end_idx] < word_ids[start_idx]:
                        continue

                    score = s_logits[start_idx] + e_logits[end_idx]

                    if score > best_score:
                        best_score = score
                        best_span = (word_ids[start_idx], word_ids[end_idx])

            if example_id not in predictions or best_score > predictions[example_id][1]:
                predictions[example_id] = (best_span, best_score)

        predictions = {k: v[0] for k, v in predictions.items()}

        # ====== METRICS ======
        qa_metrics = evaluate_predictions_metrics_only(
            self.eval_examples,
            predictions,
        )

        qa_metrics = {f"{metric_key_prefix}_{k}": v for k, v in qa_metrics.items()}

        metrics.update(qa_metrics)

        # ====== SAVE ======
        step = int(self.state.global_step)
        epoch = None if self.state.epoch is None else float(self.state.epoch)

        metrics_for_file = {
            "global_step": step,
            "epoch": epoch,
            "metrics": to_jsonable(metrics),
        }

        out_dir = os.path.join(self.args.output_dir, "eval_outputs")
        ensure_dir(out_dir)

        save_json(
            os.path.join(out_dir, f"{metric_key_prefix}_metrics_step_{step}.json"),
            metrics_for_file,
        )

        append_jsonl(
            os.path.join(out_dir, f"{metric_key_prefix}_metrics_history.jsonl"),
            metrics_for_file,
        )

        self.log(metrics)

        return metrics


# ============================================================
# NeoBERT QA wrapper
# ============================================================


class NeoBERTForQuestionAnswering(PreTrainedModel):
    """
    Обертка для NeoBERT под extractive QA.

    NeoBERT сам по себе является encoder-only моделью и не имеет
    стандартного AutoModelForQuestionAnswering-класса.

    Поэтому мы берем NeoBERT как backbone:
        input_ids -> hidden_states

    И добавляем сверху QA-head:
        hidden_states -> Linear(hidden_size, 2)

    На выходе получаем:
        start_logits, end_logits
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = 2

        self.neobert = AutoModel.from_pretrained(
            config.name_or_path,
            config=config,
            trust_remote_code=True,
        )

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("NeoBERT config does not contain hidden_size.")

        self.qa_outputs = torch.nn.Linear(hidden_size, self.num_labels)

        initializer_range = getattr(config, "initializer_range", 0.02)
        torch.nn.init.normal_(self.qa_outputs.weight, mean=0.0, std=initializer_range)
        torch.nn.init.zeros_(self.qa_outputs.bias)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.neobert, "gradient_checkpointing_enable"):
            self.neobert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.neobert, "gradient_checkpointing_disable"):
            self.neobert.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        **kwargs,
    ):
        outputs = self.neobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )


def is_neobert_model(model_name: str) -> bool:
    return "neobert" in model_name.lower()


def load_tokenizer_and_model(cfg: Config):
    """
    Загружает tokenizer и модель.

    Для обычных QA-моделей, например ModernBERT:
        AutoModelForQuestionAnswering

    Для NeoBERT:
        AutoModel + вручную добавленная QA-head
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if is_neobert_model(cfg.model_name):
        print("Detected NeoBERT. Loading AutoModel backbone + custom QA head...")

        config = AutoConfig.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )
        config.name_or_path = cfg.model_name

        model = NeoBERTForQuestionAnswering(config)
    else:
        print("Loading standard AutoModelForQuestionAnswering...")

        model = AutoModelForQuestionAnswering.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )

    return tokenizer, model


# ============================================================
# Main
# ============================================================


def main():
    """Точка входа: подготовка данных, обучение модели и финальная оценка."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    ensure_dir(cfg.output_dir)
    save_json(os.path.join(cfg.output_dir, "used_config.json"), vars(cfg))

    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
        torch.cuda.manual_seed_all(cfg.seed)

    log_gpu_info()

    print("Loading tokenizer and model...")
    tokenizer, model = load_tokenizer_and_model(cfg)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_files = resolve_nq_input_files(cfg.train_file, split="train")
    eval_files = resolve_nq_input_files(cfg.validation_file, split="dev")

    save_json(os.path.join(cfg.output_dir, "train_input_files.json"), train_files)
    save_json(os.path.join(cfg.output_dir, "validation_input_files.json"), eval_files)

    print("Loading train examples...")
    train_examples = load_nq_examples(
        cfg.train_file,
        split="train",
        files=train_files,
    )

    print("Loading validation examples...")
    eval_examples = load_nq_examples(
        cfg.validation_file,
        split="dev",
        files=eval_files,
    )

    print("Tokenizing train set...")
    train_dataset = build_train_features(train_examples, tokenizer, cfg)

    print("Tokenizing validation set...")
    eval_dataset, eval_features_meta = build_eval_features(
        eval_examples, tokenizer, cfg
    )

    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(eval_examples)}")
    print(f"Train features: {len(train_dataset)}")
    print(f"Validation features: {len(eval_dataset)}")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

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
        logging_first_step=True,
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
        eval_input_path=cfg.validation_file,
        eval_input_files=eval_files,
        cfg=cfg,
        callbacks=[MetricsLoggingCallback(cfg.output_dir)],
    )

    print("Starting training...")
    trainer.train()

    print("Saving tokenizer and final model...")
    tokenizer.save_pretrained(cfg.output_dir)
    trainer.save_model(cfg.output_dir)

    print("Running final evaluation...")
    final_metrics = trainer.evaluate()
    save_json(os.path.join(cfg.output_dir, "final_eval_metrics.json"), final_metrics)

    print("Done.")
    print(json.dumps(to_jsonable(final_metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
