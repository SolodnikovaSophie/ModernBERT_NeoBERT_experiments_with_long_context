# Этот скрипт обучает модель для extractive question answering на данных Natural Questions.
# Версия упрощена и оптимизирована под уже очищенный датасет:
# - HTML уже удален
# - YES/NO уже удалены
# - примеры без long answer уже удалены
# - примеры без short answer уже удалены

import os
import json
import gzip
import random
import glob
import re
import string
from collections import Counter
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
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

    # === Новые опции: устранение позиционного байаса ===
    # Включить ли сэмплирование окна по бинам позиции ответа в обучении.
    # Если False — используется старое поведение (контекст = long_answer).
    position_balanced_sampling: bool = True
    # Сколько равных бинов по позиции ответа в окне. По умолчанию 4: 0-25, 25-50, 50-75, 75-100.
    num_position_bins: int = 4
    # Веса CE-лосса для каждого бина (длина = num_position_bins). Чем дальше ответ,
    # тем сильнее «штраф». None или пустой список → веса все 1.0 (выключено).
    position_bin_loss_weights: Optional[List[float]] = None
    # Ограничение длины полного документа в словах для обучения, чтобы не тратить
    # память на гигантские документы. None = без ограничения.
    max_doc_words: Optional[int] = None

    # === SQuAD 2.0-режим (no-answer) ===
    # Если True — в обучающей выборке остаются примеры:
    #   * без short_answer (но с long_answer),
    #   * без long_answer вовсе (документы без точного ответа).
    # Для них train_span = (-1, -1) и таргетом является [CLS] (классический SQuAD 2.0 трюк).
    # На eval применяется null-score thresholding.
    # Если False — старое поведение: такие примеры отбрасываются.
    allow_unanswerable: bool = False
    # Доля unanswerable примеров в обучении (в долях от общего количества).
    # None — оставить все unanswerable примеры как есть.
    # Например, 0.3 → отрицательных будет ~30% от итогового train-set.
    unanswerable_ratio: Optional[float] = None


def load_config(path: str) -> Config:
    """Загружает конфигурацию обучения из YAML-файла."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Отфильтруем неизвестные ключи на всякий случай (для совместимости со старыми YAML)
    allowed = set(Config.__dataclass_fields__.keys())
    data = {k: v for k, v in data.items() if k in allowed}
    return Config(**data)


def ensure_dir(path: str) -> None:
    """Создает папку, если она не существует."""
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    """Сохраняет объект в JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any) -> None:
    """Дописывает объект в JSONL-файл одной строкой."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_jsonl(path: str, items: List[Any]) -> None:
    """Сохраняет список объектов в JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    allow_unanswerable: bool = False,
    source_file: Optional[str] = None,
    source_input: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Нормализует пример датасета NQ.

    Возвращает dict со следующими случаями:
    - **answerable**: есть long_answer и short_answer. context_tokens = long_answer,
      train_span/gold_spans в локальных координатах long_answer;
      doc_tokens полный, train_span_global/gold_spans_global в глобальных координатах.
    - **unanswerable, long_answer есть, short_answer нет** (только если allow_unanswerable=True):
      context_tokens = long_answer, train_span = (-1, -1), gold_spans = [];
      doc_tokens полный, train_span_global = (-1, -1), gold_spans_global = [].
    - **unanswerable, long_answer нет** (только если allow_unanswerable=True):
      context_tokens = doc_tokens полный, train_span/gold_spans пустые;
      doc_tokens полный, train_span_global = (-1, -1).

    Возвращает None для невалидных примеров и для unanswerable, когда allow_unanswerable=False.
    """
    try:
        ann = example["annotations"][0]
        long_answer = ann["long_answer"]
        short_answers = ann.get("short_answers", []) or []

        long_start = int(long_answer["start_token"])
        long_end = int(long_answer["end_token"])  # exclusive

        doc_tokens = get_doc_tokens(example)
        if not doc_tokens:
            return None

        question_tokens = example.get("question_tokens")
        if not question_tokens:
            question_tokens = example["question_text"].split()

        long_answer_valid = (
            long_start >= 0 and long_end > long_start and long_end <= len(doc_tokens)
        )

        # --- Кейс 1: long_answer отсутствует / невалиден ---
        if not long_answer_valid:
            if not allow_unanswerable:
                return None
            return {
                "example_id": str(example["example_id"]),
                "question_tokens": question_tokens,
                "context_tokens": doc_tokens,  # для eval контекст = весь документ
                "train_span": (-1, -1),
                "gold_spans": [],
                "doc_tokens": doc_tokens,
                "train_span_global": (-1, -1),
                "gold_spans_global": [],
                "is_impossible": True,
                "source_file": source_file,
                "source_file_name": (
                    os.path.basename(source_file) if source_file else None
                ),
                "source_input": source_input,
            }

        context_tokens = doc_tokens[long_start:long_end]
        if not context_tokens:
            return None

        # --- Кейс 2: long_answer есть, short_answer пустой ---
        if not short_answers:
            if not allow_unanswerable:
                return None
            return {
                "example_id": str(example["example_id"]),
                "question_tokens": question_tokens,
                "context_tokens": context_tokens,
                "train_span": (-1, -1),
                "gold_spans": [],
                "doc_tokens": doc_tokens,
                "train_span_global": (-1, -1),
                "gold_spans_global": [],
                "is_impossible": True,
                "source_file": source_file,
                "source_file_name": (
                    os.path.basename(source_file) if source_file else None
                ),
                "source_input": source_input,
            }

        # --- Кейс 3: обычный answerable пример ---
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

        train_span_global = (short_start_global, short_end_global_exclusive - 1)
        gold_spans_global: List[Tuple[int, int]] = []
        for sa in short_answers:
            s_g = int(sa["start_token"])
            e_g = int(sa["end_token"]) - 1
            if 0 <= s_g <= e_g < len(doc_tokens):
                gold_spans_global.append((s_g, e_g))

        return {
            "example_id": str(example["example_id"]),
            "question_tokens": question_tokens,
            "context_tokens": context_tokens,
            "train_span": (short_start_local, short_end_local_inclusive),
            "gold_spans": gold_spans_local,
            "doc_tokens": doc_tokens,
            "train_span_global": train_span_global,
            "gold_spans_global": gold_spans_global,
            "is_impossible": False,
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
    allow_unanswerable: bool = False,
    unanswerable_ratio: Optional[float] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Загружает примеры из NQ-датасета.

    Если allow_unanswerable=True, в выборку включаются примеры без short_answer
    и без long_answer (для SQuAD 2.0-режима).

    Если unanswerable_ratio задан и allow_unanswerable=True — отрицательные
    примеры даунсэмплируются так, чтобы их доля в итоговом наборе примерно
    равнялась этому значению.
    """
    if files is None:
        files = resolve_nq_input_files(path, split=split)

    kept: List[Dict[str, Any]] = []
    total = 0
    invalid = 0
    answerable_n = 0
    unanswerable_n = 0

    for fp in files:
        for ex in iter_jsonl_or_gz(fp):
            total += 1
            norm = normalize_nq_example(
                ex,
                allow_unanswerable=allow_unanswerable,
                source_file=fp,
                source_input=path,
            )
            if norm is None:
                invalid += 1
                continue
            kept.append(norm)
            if norm.get("is_impossible"):
                unanswerable_n += 1
            else:
                answerable_n += 1

    # Downsampling отрицательных примеров.
    if (
        allow_unanswerable
        and unanswerable_ratio is not None
        and 0.0 < unanswerable_ratio < 1.0
        and unanswerable_n > 0
        and answerable_n > 0
    ):
        target_neg = int(
            round(answerable_n * unanswerable_ratio / (1.0 - unanswerable_ratio))
        )
        if target_neg < unanswerable_n:
            rng = random.Random(seed)
            pos = [e for e in kept if not e.get("is_impossible")]
            neg = [e for e in kept if e.get("is_impossible")]
            rng.shuffle(neg)
            kept = pos + neg[:target_neg]
            rng.shuffle(kept)
            unanswerable_n = target_neg

    print(
        f"[{split or 'DATA'}] files={len(files)}, total={total}, kept={len(kept)} "
        f"(answerable={answerable_n}, unanswerable={unanswerable_n}), invalid={invalid}"
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
    Находит токеновые start/end позиции ответа в пределах конкретного feature-окна.

    Возвращает:
    - (token_start, token_end), если span попал в окно;
    - None, если span не попал в окно или определить его нельзя.
    """
    # Unanswerable: gold_start == gold_end == -1 → таргет = [CLS].
    if gold_start < 0 or gold_end < 0:
        return None

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


def _resolve_position_bin_weights(cfg: Config) -> List[float]:
    """Возвращает список весов лосса по бинам (или все единицы, если выключено)."""
    n = max(1, int(cfg.num_position_bins))
    weights = cfg.position_bin_loss_weights
    if not weights:
        return [1.0] * n
    if len(weights) != n:
        raise ValueError(
            f"position_bin_loss_weights должен иметь длину num_position_bins={n}, "
            f"получено {len(weights)}"
        )
    return [float(w) for w in weights]


def _sample_word_window(
    doc_len: int,
    answer_start: int,
    answer_end: int,
    word_budget: int,
    num_bins: int,
    rng: random.Random,
) -> Tuple[int, int, int]:
    """
    Выбирает окно длиной word_budget (в словах) из документа длиной doc_len,
    стремясь поместить ответ в случайно выбранный бин позиции.

    Возвращает (window_start, window_end_exclusive, bin_index), где bin_index —
    бин, в который фактически попал ответ после клиппинга к границам документа.
    """
    target_bin = rng.randrange(num_bins)
    answer_len = answer_end - answer_start + 1

    if doc_len <= word_budget:
        # Документ короче бюджета окна — окно фиксировано, бин определяется естественно.
        win_start, win_end = 0, doc_len
    else:
        # Хотим, чтобы answer_start попал в относительную позицию p в окне.
        # Берём p равномерно внутри выбранного бина, но оставляя место под сам ответ.
        bin_lo = target_bin / num_bins
        bin_hi = (target_bin + 1) / num_bins
        max_p = max(bin_lo, 1.0 - answer_len / word_budget)
        bin_hi_eff = min(bin_hi, max_p) if max_p > bin_lo else bin_hi
        if bin_hi_eff <= bin_lo:
            p = bin_lo
        else:
            p = rng.uniform(bin_lo, bin_hi_eff)

        # Сдвигаем окно так, чтобы answer_start был на относительной позиции p.
        win_start = int(round(answer_start - p * word_budget))
        win_start = max(0, min(win_start, doc_len - word_budget))
        win_end = win_start + word_budget

        # Если ответ всё-таки не помещается целиком — подвинуть окно так, чтобы поместился.
        if answer_end >= win_end:
            win_end = min(doc_len, answer_end + 1)
            win_start = max(0, win_end - word_budget)
        if answer_start < win_start:
            win_start = answer_start
            win_end = min(doc_len, win_start + word_budget)

    # Фактический бин ответа в окне после клиппинга.
    win_size = max(1, win_end - win_start)
    rel_pos = (answer_start - win_start) / win_size
    actual_bin = min(num_bins - 1, max(0, int(rel_pos * num_bins)))
    return win_start, win_end, actual_bin


def build_train_features(
    examples: List[Dict[str, Any]],
    tokenizer,
    cfg: Config,
) -> Dataset:
    """
    Строит train features для extractive QA.

    Если cfg.position_balanced_sampling=True — для каждого примера выбирается
    окно из полного документа так, чтобы ответ равновероятно оказался в одном
    из num_position_bins бинов позиции внутри окна. Дополнительно для каждого
    feature сохраняется position_weight (вес для CE-лосса), зависящий от бина.

    Если False — используется старое поведение (контекст = long_answer, без overflow-генерации,
    окна нарезаются tokenizer'ом со stride).
    """
    bin_weights = _resolve_position_bin_weights(cfg)
    num_bins = len(bin_weights)

    if not cfg.position_balanced_sampling:
        # --- Старое поведение ---
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

        start_positions, end_positions, position_weights = [], [], []
        for i in range(len(tokenized["input_ids"])):
            sample_idx = sample_mapping[i]
            ex = examples[sample_idx]
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            word_ids = tokenized.word_ids(batch_index=i)
            sequence_ids = tokenized.sequence_ids(i)
            gold_start, gold_end = ex["train_span"]
            answer_pos = _locate_answer_tokens(
                word_ids, sequence_ids, gold_start, gold_end
            )
            if answer_pos is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                token_start, token_end = answer_pos
                start_positions.append(token_start)
                end_positions.append(token_end)
            position_weights.append(1.0)

        return Dataset.from_dict(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "start_positions": start_positions,
                "end_positions": end_positions,
                "position_weight": position_weights,
            }
        )

    # --- Сэмплирование окна из полного документа с балансировкой по бинам ---
    rng = random.Random(cfg.seed)

    # Бюджет окна в словах. Подбираем консервативно, чтобы subword-токенизация
    # с большой вероятностью уместилась в max_seq_length.
    # Опытная оценка: ~1.4 subword токена на слово в среднем для NQ.
    overhead = cfg.max_query_length + 5  # question + спец. токены
    word_budget = max(64, int((cfg.max_seq_length - overhead) / 1.4))

    all_input_ids: List[List[int]] = []
    all_attention: List[List[int]] = []
    start_positions: List[int] = []
    end_positions: List[int] = []
    position_weights: List[float] = []
    bin_counts = [0] * num_bins
    fallback_cls = 0
    unanswerable_count = 0

    for ex in examples:
        doc_tokens: List[str] = ex["doc_tokens"]
        is_impossible = bool(ex.get("is_impossible"))

        if is_impossible:
            # --- Unanswerable: семплируем произвольное окно из документа ---
            if cfg.max_doc_words is not None and len(doc_tokens) > cfg.max_doc_words:
                doc_tokens_use = doc_tokens[: cfg.max_doc_words]
            else:
                doc_tokens_use = doc_tokens

            doc_len = len(doc_tokens_use)
            if doc_len <= word_budget:
                win_start, win_end = 0, doc_len
            else:
                win_start = rng.randint(0, doc_len - word_budget)
                win_end = win_start + word_budget
            context_window = doc_tokens_use[win_start:win_end]

            question = ex["question_tokens"][: cfg.max_query_length]
            tokenized = tokenizer(
                [question],
                [context_window],
                is_split_into_words=True,
                truncation="only_second",
                max_length=cfg.max_seq_length,
                return_attention_mask=True,
                padding=False,
            )
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            all_input_ids.append(input_ids)
            all_attention.append(attention_mask)
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            position_weights.append(1.0)  # вес для unanswerable всегда 1.0
            unanswerable_count += 1
            continue

        # --- Answerable: позиционно-сбалансированное окно ---
        if cfg.max_doc_words is not None and len(doc_tokens) > cfg.max_doc_words:
            ans_s, ans_e = ex["train_span_global"]
            center = (ans_s + ans_e) // 2
            half = cfg.max_doc_words // 2
            ds = max(0, center - half)
            de = min(len(doc_tokens), ds + cfg.max_doc_words)
            ds = max(0, de - cfg.max_doc_words)
            doc_tokens_use = doc_tokens[ds:de]
            ans_s -= ds
            ans_e -= ds
        else:
            doc_tokens_use = doc_tokens
            ans_s, ans_e = ex["train_span_global"]

        if not (0 <= ans_s <= ans_e < len(doc_tokens_use)):
            continue

        win_start, win_end, actual_bin = _sample_word_window(
            doc_len=len(doc_tokens_use),
            answer_start=ans_s,
            answer_end=ans_e,
            word_budget=word_budget,
            num_bins=num_bins,
            rng=rng,
        )
        context_window = doc_tokens_use[win_start:win_end]
        local_gold_start = ans_s - win_start
        local_gold_end = ans_e - win_start

        question = ex["question_tokens"][: cfg.max_query_length]
        tokenized = tokenizer(
            [question],
            [context_window],
            is_split_into_words=True,
            truncation="only_second",
            max_length=cfg.max_seq_length,
            return_attention_mask=True,
            padding=False,
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        word_ids = tokenized.word_ids(batch_index=0)
        sequence_ids = tokenized.sequence_ids(0)

        cls_index = input_ids.index(tokenizer.cls_token_id)
        answer_pos = _locate_answer_tokens(
            word_ids, sequence_ids, local_gold_start, local_gold_end
        )

        if answer_pos is None:
            fallback_cls += 1
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            ts, te = answer_pos
            start_positions.append(ts)
            end_positions.append(te)

        all_input_ids.append(input_ids)
        all_attention.append(attention_mask)
        position_weights.append(float(bin_weights[actual_bin]))
        bin_counts[actual_bin] += 1

    print(
        f"[TRAIN] position-balanced sampling: bins={num_bins}, "
        f"word_budget={word_budget}, distribution per bin = {bin_counts}, "
        f"weights = {bin_weights}, cls_fallback_features={fallback_cls}, "
        f"unanswerable_features={unanswerable_count}"
    )

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "position_weight": position_weights,
        }
    )


def build_eval_features(
    examples: List[Dict[str, Any]],
    tokenizer,
    cfg: Config,
) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """
    Строит eval features и метаданные для постобработки.
    """
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
            "position_weight": [1.0] * len(tokenized["input_ids"]),
        }
    )
    return dataset, features_meta


# ============================================================
# Metrics
# ============================================================


def span_to_text(span: Optional[Tuple[int, int]], context_tokens: List[str]) -> str:
    """Converts an inclusive token span to plain answer text."""
    if span is None:
        return ""

    s, e = span
    if s < 0 or e < s or e >= len(context_tokens):
        return ""

    return " ".join(context_tokens[s : e + 1]).strip()


def normalize_answer(text: str) -> str:
    """
    Official SQuAD normalization:
    lowercasing, punctuation/article removal and extra whitespace cleanup.
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
    if len(logits) <= k:
        return np.argsort(logits)[::-1].tolist()
    idx = np.argpartition(logits, -k)[-k:]
    return idx[np.argsort(logits[idx])[::-1]].tolist()


def postprocess_predictions(
    examples: List[Dict[str, Any]],
    features_meta: List[Dict[str, Any]],
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    cfg: Config,
) -> Tuple[Dict[str, Optional[Tuple[int, int]]], List[Dict[str, Any]]]:
    """
    Преобразует логиты модели в лучшие span-предсказания по примерам.

    Если cfg.allow_unanswerable=True — применяется SQuAD 2.0 null-score thresholding:
    для каждого примера сравнивается лучший «настоящий» span-score с минимальным
    null score (логит [CLS]) по всем feature-окнам. Если null >= best → предсказание None.
    """
    start_logits, end_logits = raw_predictions
    best_predictions: Dict[str, Dict[str, Any]] = {}
    null_scores: Dict[str, float] = {}

    for feature_idx, meta in enumerate(features_meta):
        ex_id = meta["example_id"]
        word_ids = meta["word_ids"]
        context_mask = meta["context_mask"]

        s_logits = start_logits[feature_idx]
        e_logits = end_logits[feature_idx]

        # Null score для текущей фичи (логиты на [CLS] = индекс 0).
        feature_null_score = float(s_logits[0] + e_logits[0])
        prev_null = null_scores.get(ex_id)
        if prev_null is None or feature_null_score < prev_null:
            null_scores[ex_id] = feature_null_score

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
        null_score = null_scores.get(ex_id)

        pred_span = None if best is None else best["span"]
        score = None if best is None else best["score"]

        # SQuAD 2.0: если null score >= лучший span-score, считаем что ответа нет.
        if (
            getattr(cfg, "allow_unanswerable", False)
            and null_score is not None
            and (score is None or null_score >= score)
        ):
            pred_span = None
            # score оставляем как есть (для логирования)

        pred_text = span_to_text(pred_span, context_tokens)

        final_predictions[ex_id] = pred_span
        pred_details.append(
            {
                "example_id": ex_id,
                "pred_span": pred_span,
                "pred_text": pred_text,
                "score": score,
                "null_score": null_score,
                "is_impossible_gold": bool(ex.get("is_impossible", False)),
                "source_file": ex.get("source_file"),
                "source_file_name": ex.get("source_file_name"),
                "source_input": ex.get("source_input"),
                "train_span": ex.get("train_span"),
                "true_span": ex.get("train_span"),
                "gold_spans": ex.get("gold_spans"),
            }
        )

    return final_predictions, pred_details


def evaluate_predictions(
    examples: List[Dict[str, Any]],
    predictions: Dict[str, Optional[Tuple[int, int]]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Считает SQuAD EM/F1 и детализацию по каждому примеру."""
    em_scores = []
    f1_scores = []
    per_example = []

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

        best_em = 0.0
        best_f1 = 0.0
        best_gold_span = None
        best_gold_text = ""

        for idx, gold_text in enumerate(gold_texts):
            em = squad_exact_match_score(pred_text, gold_text)
            f1 = squad_f1_score(pred_text, gold_text)
            if (f1 > best_f1) or (f1 == best_f1 and em > best_em):
                best_f1 = f1
                best_em = em
                best_gold_span = gold_spans[idx] if idx < len(gold_spans) else None
                best_gold_text = gold_text

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

        per_example.append(
            {
                "example_id": ex_id,
                "pred_span": pred_span,
                "pred_text": pred_text,
                "train_span": ex.get("train_span"),
                "true_span": ex.get("train_span"),
                "gold_spans": gold_spans,
                "gold_texts": gold_texts,
                "matched_gold_span": best_gold_span,
                "matched_gold_text": best_gold_text,
                "exact_match": max_em,
                "f1": max_f1,
                "source_file": ex.get("source_file"),
                "source_file_name": ex.get("source_file_name"),
                "source_input": ex.get("source_input"),
            }
        )

    metrics = {
        "exact_match": (float(np.mean(em_scores)) * 100.0) if em_scores else 0.0,
        "f1": (float(np.mean(f1_scores)) * 100.0) if f1_scores else 0.0,
        "n_examples": len(examples),
    }
    return metrics, per_example


# ============================================================
# Custom Trainer
# ============================================================


class MetricsLoggingCallback(TrainerCallback):
    """
    Сохраняет trainer logs.
    Оптимизация: log_history.json перезаписывается только в конце обучения,
    а не на каждом шаге логирования.
    """

    def __init__(self, output_dir: str):
        self.logs_dir = os.path.join(output_dir, "training_logs")
        self.metrics_jsonl = os.path.join(self.logs_dir, "metrics_log.jsonl")
        self.substeps_jsonl = os.path.join(self.logs_dir, "substeps_log.jsonl")
        self.log_history_json = os.path.join(self.logs_dir, "log_history.json")
        self._substep_counter = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ensure_dir(self.logs_dir)
        with open(self.metrics_jsonl, "w", encoding="utf-8"):
            pass
        with open(self.substeps_jsonl, "w", encoding="utf-8"):
            pass
        save_json(self.log_history_json, [])
        self._substep_counter = 0

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

    def on_substep_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self._substep_counter += 1
        grad_acc = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        optimizer = kwargs.get("optimizer")
        lr = None
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            lr = optimizer.param_groups[0].get("lr")

        entry = {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "substep_index_total": self._substep_counter,
            "substep_in_update_step": ((self._substep_counter - 1) % grad_acc) + 1,
            "gradient_accumulation_steps": grad_acc,
            "global_step": int(state.global_step),
            "epoch": None if state.epoch is None else float(state.epoch),
            "learning_rate": None if lr is None else float(lr),
        }
        append_jsonl(self.substeps_jsonl, entry)


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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Position-Weighted Cross-Entropy для span QA.

        Стандартный QA-лосс — среднее по батчу от (CE(start) + CE(end)) / 2.
        Здесь мы считаем то же самое, но per-example, и взвешиваем по position_weight
        (зависит от бина позиции ответа в окне — см. build_train_features).
        """
        weights = inputs.pop("position_weight", None)
        start_positions = inputs.pop("start_positions")
        end_positions = inputs.pop("end_positions")

        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Клиппинг таргетов к длине последовательности (стандартная практика HF QA).
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index - 1)
        end_positions = end_positions.clamp(0, ignored_index - 1)

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        per_example_loss = 0.5 * (start_loss + end_loss)

        if weights is not None:
            weights = weights.to(per_example_loss.dtype).to(per_example_loss.device)
            loss = (per_example_loss * weights).sum() / weights.sum().clamp_min(1e-8)
        else:
            loss = per_example_loss.mean()

        # Подменяем поле loss в выходе модели для совместимости с остальным пайплайном.
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        pred_output = self.predict(
            eval_dataset if eval_dataset is not None else self.eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        predictions, pred_details = postprocess_predictions(
            examples=self.eval_examples,
            features_meta=self.eval_features_meta,
            raw_predictions=pred_output.predictions,
            cfg=self.cfg,
        )

        qa_metrics, per_example = evaluate_predictions(self.eval_examples, predictions)
        qa_metrics = {f"{metric_key_prefix}_{k}": v for k, v in qa_metrics.items()}
        metrics.update(qa_metrics)

        step = self.state.global_step
        out_dir = os.path.join(self.args.output_dir, "eval_outputs")
        ensure_dir(out_dir)

        save_json(
            os.path.join(out_dir, f"{metric_key_prefix}_metrics_step_{step}.json"),
            metrics,
        )
        save_json(
            os.path.join(out_dir, f"{metric_key_prefix}_predictions_step_{step}.json"),
            {k: list(v) if v is not None else None for k, v in predictions.items()},
        )
        save_json(
            os.path.join(out_dir, f"{metric_key_prefix}_pred_details_step_{step}.json"),
            pred_details,
        )
        save_json(
            os.path.join(out_dir, f"{metric_key_prefix}_per_example_step_{step}.json"),
            per_example,
        )
        save_jsonl(
            os.path.join(out_dir, f"{metric_key_prefix}_per_example_step_{step}.jsonl"),
            per_example,
        )

        self.log(metrics)
        return metrics


# ============================================================
# Main
# ============================================================


def main():
    """Точка входа: подготовка данных, обучение модели и финальная оценка."""
    import argparse

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
        torch.cuda.manual_seed_all(cfg.seed)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_files = resolve_nq_input_files(cfg.train_file, split="train")
    eval_files = resolve_nq_input_files(cfg.validation_file, split="dev")
    save_json(os.path.join(cfg.output_dir, "train_input_files.json"), train_files)
    save_json(os.path.join(cfg.output_dir, "validation_input_files.json"), eval_files)

    print(f"Loading train examples (allow_unanswerable={cfg.allow_unanswerable})...")
    train_examples = load_nq_examples(
        cfg.train_file,
        split="train",
        files=train_files,
        allow_unanswerable=cfg.allow_unanswerable,
        unanswerable_ratio=cfg.unanswerable_ratio,
        seed=cfg.seed,
    )

    print(
        f"Loading validation examples (allow_unanswerable={cfg.allow_unanswerable})..."
    )
    eval_examples = load_nq_examples(
        cfg.validation_file,
        split="dev",
        files=eval_files,
        allow_unanswerable=cfg.allow_unanswerable,
        unanswerable_ratio=None,  # на eval отрицательные не даунсэмплируем
        seed=cfg.seed,
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
        report_to=cfg.report_to,
        # ВАЖНО: position_weight не входит в сигнатуру forward, но нужен для compute_loss.
        # Поэтому выключаем автоудаление неиспользуемых колонок.
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_num_workers=cfg.dataloader_num_workers,
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

    print("Running final evaluation...")
    final_metrics = trainer.evaluate()
    save_json(os.path.join(cfg.output_dir, "final_eval_metrics.json"), final_metrics)

    print("Done.")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
