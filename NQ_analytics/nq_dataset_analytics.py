#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Аналитика по датасету Natural Questions из папки с .jsonl.gz файлами.

Режимы работы с аннотациями:
1. --annotation-mode first
   Берется только первая аннотация каждого примера.
   В этом режиме статистика считается по примерам, без графика числа аннотаций.

2. --annotation-mode all
   Обрабатываются все аннотации каждого примера.
   В этом режиме дополнительно считается:
   - количество аннотаций на пример
   - распределение количества аннотаций

Что считает:
1. Количество объектов с YES / NO / NULL / SPAN
2. Количество объектов с long answer / без long answer
3. Распределение длин long answer:
   - в токенах NQ
   - в символах
4. Распределение длин short answer:
   - в токенах NQ
   - в символах
5. Распределение количества short answers на объект анализа
6. Распределение длины вопроса в символах
7. Распределение относительной позиции short answer внутри long answer
8. Отношение длины long answer к длине документа
9. Сводные статистики по всем длинам

Важно:
- В режиме first единица анализа = пример с первой аннотацией
- В режиме all единица анализа = аннотация

Особенность этой версии:
- для long_answer_length_chars.png используется шаг бина 5000
- для long_answer_length_tokens.png используется шаг бина 5000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Конфиг и структуры
# ============================================================

@dataclass
class Config:
    input_dir: str
    output_dir: str
    annotation_mode: str
    long_answer_chars_bin_size: int
    long_answer_tokens_bin_size: int
    bins_short_answer_tokens: int
    bins_short_answer_chars: int
    bins_question_chars: int
    bins_relative_position: int
    bins_ratio: int
    bins_annotation_count: int


@dataclass
class LengthStats:
    count: int
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]


# ============================================================
# Утилиты
# ============================================================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def iter_gz_jsonl_files(input_dir: str) -> List[Path]:
    root = Path(input_dir)
    files = sorted(root.glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"В папке не найдено файлов .jsonl.gz: {input_dir}")
    return files


def iter_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Ошибка JSON в {path} строка {line_no}: {e}")


def has_valid_span(start: Any, end: Any) -> bool:
    try:
        start = int(start)
        end = int(end)
        return start >= 0 and end > start
    except Exception:
        return False


def normalize_yes_no(value: Any) -> str:
    if value is None:
        return "NONE"
    value = str(value).strip().upper()
    if value in {"YES", "NO", "NONE"}:
        return value
    return "NONE"


def extract_token_texts(document_tokens: List[Dict[str, Any]]) -> List[str]:
    texts = []
    for token_info in document_tokens:
        token_text = token_info.get("token", "")
        if token_text is None:
            token_text = ""
        texts.append(str(token_text))
    return texts


def join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens).strip()


def compute_length_stats(values: List[float | int]) -> Dict[str, Any]:
    if not values:
        return asdict(
            LengthStats(
                count=0,
                mean=None,
                median=None,
                std=None,
                min=None,
                max=None,
                p25=None,
                p50=None,
                p75=None,
                p90=None,
                p95=None,
                p99=None,
            )
        )

    arr = np.array(values, dtype=float)
    return asdict(
        LengthStats(
            count=int(arr.size),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )
    )


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_counter_csv(counter: Counter, path: str | Path, key_name: str, value_name: str = "count") -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([key_name, value_name])
        for key, value in sorted(counter.items(), key=lambda x: x[0]):
            writer.writerow([key, value])


def save_rows_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_hist(
    values: List[float | int],
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(10, 6))
    if values:
        plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_hist_fixed_step_limited(
    values: List[float | int],
    step: int,
    max_value: int,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(12, 6))

    if values:
        clipped_values = [v for v in values if v <= max_value]
        tail_count = len(values) - len(clipped_values)

        upper = int(math.ceil(max_value / step) * step)
        if upper == 0:
            upper = step

        bin_edges = np.arange(0, upper + step, step)
        plt.hist(clipped_values, bins=bin_edges)

        # Показываем не все подписи, чтобы они не слипались
        tick_step = max(step, 2000)
        tick_positions = np.arange(0, upper + tick_step, tick_step)
        plt.xticks(tick_positions, rotation=45, ha="right")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bar_from_counter(
    counter: Counter,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
    sort_numeric: bool = True,
) -> None:
    plt.figure(figsize=(10, 6))
    if counter:
        items = list(counter.items())
        if sort_numeric:
            items = sorted(items, key=lambda x: x[0])
        else:
            items = sorted(items, key=lambda x: str(x[0]))

        xs = [str(k) for k, _ in items]
        ys = [v for _, v in items]
        plt.bar(xs, ys)
        plt.xticks(rotation=45, ha="right")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# Логика работы с аннотациями
# ============================================================

def classify_short_answer_type(annotation: Optional[Dict[str, Any]]) -> str:
    if annotation is None:
        return "NULL"

    yes_no = normalize_yes_no(annotation.get("yes_no_answer", "NONE"))
    short_answers = annotation.get("short_answers", []) or []

    if yes_no == "YES":
        return "YES"
    if yes_no == "NO":
        return "NO"
    if short_answers:
        return "SPAN"
    return "NULL"


def has_long_answer(annotation: Optional[Dict[str, Any]]) -> bool:
    if annotation is None:
        return False
    long_answer = annotation.get("long_answer", {}) or {}
    return has_valid_span(long_answer.get("start_token", -1), long_answer.get("end_token", -1))


def get_long_answer_span(annotation: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if annotation is None:
        return None

    long_answer = annotation.get("long_answer", {}) or {}
    start = long_answer.get("start_token", -1)
    end = long_answer.get("end_token", -1)

    if has_valid_span(start, end):
        return int(start), int(end)
    return None


def get_valid_short_answer_spans(annotation: Optional[Dict[str, Any]]) -> List[Tuple[int, int]]:
    if annotation is None:
        return []

    spans = []
    for short_answer in annotation.get("short_answers", []) or []:
        start = short_answer.get("start_token", -1)
        end = short_answer.get("end_token", -1)
        if has_valid_span(start, end):
            spans.append((int(start), int(end)))
    return spans


# ============================================================
# Основная аналитика
# ============================================================

class NQDatasetAnalytics:
    def __init__(self, config: Config) -> None:
        self.config = config

        self.files_total = 0
        self.examples_total = 0
        self.analysis_objects_total = 0

        self.answer_type_counter = Counter()
        self.long_answer_presence_counter = Counter()
        self.short_answers_per_example_counter = Counter()

        self.annotation_count_per_example: List[int] = []
        self.annotation_count_counter = Counter()

        self.long_answer_lengths_tokens: List[int] = []
        self.long_answer_lengths_chars: List[int] = []

        self.short_answer_lengths_tokens: List[int] = []
        self.short_answer_lengths_chars: List[int] = []

        self.short_answers_per_example: List[int] = []
        self.question_lengths_chars: List[int] = []
        self.relative_short_answer_positions: List[float] = []
        self.long_answer_to_document_ratios: List[float] = []

        self.example_rows: List[Dict[str, Any]] = []

    def process_folder(self) -> None:
        files = iter_gz_jsonl_files(self.config.input_dir)
        self.files_total = len(files)

        for file_path in files:
            print(f"[INFO] Обрабатываю {file_path.name}")
            for example in iter_jsonl_gz(file_path):
                self.process_example(example, source_file=file_path.name)

    def process_example(self, example: Dict[str, Any], source_file: str) -> None:
        self.examples_total += 1

        example_id = example.get("example_id")
        question_text = str(example.get("question_text", "") or "")
        document_tokens_raw = example.get("document_tokens", []) or []
        document_tokens = extract_token_texts(document_tokens_raw)
        document_len_tokens = len(document_tokens)

        annotations = example.get("annotations", []) or []

        if self.config.annotation_mode == "all":
            ann_count = len(annotations)
            self.annotation_count_per_example.append(ann_count)
            self.annotation_count_counter[ann_count] += 1
            annotations_to_process = annotations
        else:
            annotations_to_process = [annotations[0]] if annotations else [None]

        for ann_idx, annotation in enumerate(annotations_to_process):
            self.analysis_objects_total += 1

            answer_type = classify_short_answer_type(annotation)
            self.answer_type_counter[answer_type] += 1

            has_la = has_long_answer(annotation)
            self.long_answer_presence_counter["has_long_answer" if has_la else "no_long_answer"] += 1

            question_len_chars = len(question_text)
            self.question_lengths_chars.append(question_len_chars)

            valid_short_spans = get_valid_short_answer_spans(annotation)
            short_answers_count = len(valid_short_spans)
            self.short_answers_per_example.append(short_answers_count)
            self.short_answers_per_example_counter[short_answers_count] += 1

            long_answer_len_tokens = None
            long_answer_len_chars = None

            if has_la:
                long_answer_span = get_long_answer_span(annotation)
                if long_answer_span is not None:
                    long_start, long_end = long_answer_span
                    long_answer_tokens = document_tokens[long_start:long_end]
                    long_answer_text = join_tokens(long_answer_tokens)

                    long_answer_len_tokens = len(long_answer_tokens)
                    long_answer_len_chars = len(long_answer_text)

                    self.long_answer_lengths_tokens.append(long_answer_len_tokens)
                    self.long_answer_lengths_chars.append(long_answer_len_chars)

                    if document_len_tokens > 0:
                        self.long_answer_to_document_ratios.append(long_answer_len_tokens / document_len_tokens)

                    for short_start, short_end in valid_short_spans:
                        if short_start >= long_start and short_end <= long_end:
                            short_tokens = document_tokens[short_start:short_end]
                            short_text = join_tokens(short_tokens)

                            short_len_tokens = len(short_tokens)
                            short_len_chars = len(short_text)

                            self.short_answer_lengths_tokens.append(short_len_tokens)
                            self.short_answer_lengths_chars.append(short_len_chars)

                            if long_answer_len_tokens > 0:
                                relative_position = (short_start - long_start) / long_answer_len_tokens
                                self.relative_short_answer_positions.append(relative_position)

            row = {
                "source_file": source_file,
                "example_id": example_id,
                "annotation_index": ann_idx if self.config.annotation_mode == "all" else 0,
                "answer_type": answer_type,
                "has_long_answer": has_la,
                "document_len_tokens_nq": document_len_tokens,
                "question_len_chars": question_len_chars,
                "long_answer_len_tokens_nq": long_answer_len_tokens,
                "long_answer_len_chars": long_answer_len_chars,
                "short_answers_count": short_answers_count,
            }

            if self.config.annotation_mode == "all":
                row["annotations_in_example"] = len(annotations)

            self.example_rows.append(row)

    def build_summary(self) -> Dict[str, Any]:
        denominator = self.analysis_objects_total if self.analysis_objects_total else 0

        answer_type_table = []
        for label in ["YES", "NO", "NULL", "SPAN"]:
            count = int(self.answer_type_counter.get(label, 0))
            share = count / denominator if denominator else None
            answer_type_table.append(
                {
                    "type": label,
                    "count": count,
                    "share": share,
                }
            )

        long_answer_table = []
        for label in ["has_long_answer", "no_long_answer"]:
            count = int(self.long_answer_presence_counter.get(label, 0))
            share = count / denominator if denominator else None
            long_answer_table.append(
                {
                    "type": label,
                    "count": count,
                    "share": share,
                }
            )

        summary = {
            "config": asdict(self.config),
            "files_total": self.files_total,
            "examples_total": self.examples_total,
            "analysis_objects_total": self.analysis_objects_total,
            "analysis_unit": "annotation" if self.config.annotation_mode == "all" else "example_first_annotation",
            "answer_type_counts": dict(self.answer_type_counter),
            "answer_type_table": answer_type_table,
            "long_answer_presence_counts": dict(self.long_answer_presence_counter),
            "long_answer_presence_table": long_answer_table,
            "short_answers_per_example_counts": dict(self.short_answers_per_example_counter),
            "stats": {
                "long_answer_len_tokens_nq": compute_length_stats(self.long_answer_lengths_tokens),
                "long_answer_len_chars": compute_length_stats(self.long_answer_lengths_chars),
                "short_answer_len_tokens_nq": compute_length_stats(self.short_answer_lengths_tokens),
                "short_answer_len_chars": compute_length_stats(self.short_answer_lengths_chars),
                "question_len_chars": compute_length_stats(self.question_lengths_chars),
                "relative_short_answer_position": compute_length_stats(self.relative_short_answer_positions),
                "long_answer_to_document_ratio": compute_length_stats(self.long_answer_to_document_ratios),
            },
        }

        if self.config.annotation_mode == "all":
            summary["annotation_count_per_example_counts"] = dict(self.annotation_count_counter)
            summary["annotation_count_per_example_stats"] = compute_length_stats(self.annotation_count_per_example)

        return summary

    def save_outputs(self) -> None:
        output_dir = Path(self.config.output_dir)
        tables_dir = output_dir / "tables"
        plots_dir = output_dir / "plots"

        ensure_dir(output_dir)
        ensure_dir(tables_dir)
        ensure_dir(plots_dir)

        summary = self.build_summary()
        save_json(summary, output_dir / "summary.json")

        save_rows_csv(summary["answer_type_table"], tables_dir / "answer_type_table.csv")
        save_rows_csv(summary["long_answer_presence_table"], tables_dir / "long_answer_presence_table.csv")
        save_counter_csv(
            self.short_answers_per_example_counter,
            tables_dir / "short_answers_per_analysis_object.csv",
            "short_answers_count",
        )
        save_rows_csv(self.example_rows, tables_dir / "example_level_stats.csv")

        if self.config.annotation_mode == "all":
            save_counter_csv(
                self.annotation_count_counter,
                tables_dir / "annotation_count_per_example.csv",
                "annotation_count",
            )

        plot_bar_from_counter(
            self.answer_type_counter,
            title="Распределение типов short answer",
            xlabel="Тип ответа",
            ylabel="Количество объектов анализа",
            output_path=plots_dir / "answer_type_distribution.png",
            sort_numeric=False,
        )

        plot_bar_from_counter(
            self.long_answer_presence_counter,
            title="Наличие long answer",
            xlabel="Категория",
            ylabel="Количество объектов анализа",
            output_path=plots_dir / "long_answer_presence_distribution.png",
            sort_numeric=False,
        )

        plot_bar_from_counter(
            self.short_answers_per_example_counter,
            title="Количество short answers на объект анализа",
            xlabel="Число short answers",
            ylabel="Количество объектов анализа",
            output_path=plots_dir / "short_answers_per_analysis_object_distribution.png",
            sort_numeric=True,
        )

        if self.config.annotation_mode == "all":
            plot_bar_from_counter(
                self.annotation_count_counter,
                title="Распределение количества аннотаций на пример",
                xlabel="Число аннотаций",
                ylabel="Количество примеров",
                output_path=plots_dir / "annotation_count_per_example_distribution.png",
                sort_numeric=True,
            )

        # fixed step = 5000 (или то, что передано параметром)
        plot_hist_fixed_step_limited(
            self.long_answer_lengths_tokens,
            step=300,
            max_value=20000,
            title="Распределение длины long answer (токены NQ)",
            xlabel="Длина long answer, токены",
            ylabel="Количество объектов анализа",
            output_path=plots_dir / "long_answer_length_tokens.png",
        )

        plot_hist_fixed_step_limited(
            self.long_answer_lengths_tokens,
            step=300,
            max_value=20000,
            title="Распределение длины long answer (токены NQ)",
            xlabel="Длина long answer, токены",
            ylabel="Количество объектов анализа",
            output_path=plots_dir / "long_answer_length_tokens.png",
        )

        plot_hist(
            self.short_answer_lengths_tokens,
            bins=self.config.bins_short_answer_tokens,
            title="Распределение длины short answer (токены NQ)",
            xlabel="Длина short answer, токены",
            ylabel="Количество short answers",
            output_path=plots_dir / "short_answer_length_tokens.png",
        )

        plot_hist(
            self.short_answer_lengths_chars,
            bins=self.config.bins_short_answer_chars,
            title="Распределение длины short answer (символы)",
            xlabel="Длина short answer, символы",
            ylabel="Количество short answers",
            output_path=plots_dir / "short_answer_length_chars.png",
        )

        plot_hist(
            self.question_lengths_chars,
            bins=self.config.bins_question_chars,
            title="Распределение длины вопроса (символы)",
            xlabel="Длина вопроса, символы",
            ylabel="Количество объектов анализа",
            output_path=plots_dir / "question_length_chars.png",
        )

        if self.relative_short_answer_positions:
            plot_hist(
                self.relative_short_answer_positions,
                bins=self.config.bins_relative_position,
                title="Относительная позиция short answer внутри long answer",
                xlabel="Относительная позиция [0, 1]",
                ylabel="Количество short answers",
                output_path=plots_dir / "relative_short_answer_position.png",
            )

        if self.long_answer_to_document_ratios:
            plot_hist(
                self.long_answer_to_document_ratios,
                bins=self.config.bins_ratio,
                title="Отношение длины long answer к длине документа",
                xlabel="long_answer_len / document_len",
                ylabel="Количество объектов анализа",
                output_path=plots_dir / "long_answer_to_document_ratio.png",
            )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Аналитика по датасету NQ из папки с .jsonl.gz"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Папка с файлами .jsonl.gz",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Папка для сохранения результатов",
    )
    parser.add_argument(
        "--annotation-mode",
        choices=["first", "all"],
        default="first",
        help="Режим обработки аннотаций: first или all",
    )
    parser.add_argument(
        "--long-answer-chars-bin-size",
        type=int,
        default=300,
        help="Шаг бинов для long_answer_length_chars.png",
    )
    parser.add_argument(
        "--long-answer-tokens-bin-size",
        type=int,
        default=300,
        help="Шаг бинов для long_answer_length_tokens.png",
    )
    parser.add_argument("--bins-short-answer-tokens", type=int, default=30)
    parser.add_argument("--bins-short-answer-chars", type=int, default=30)
    parser.add_argument("--bins-question-chars", type=int, default=30)
    parser.add_argument("--bins-relative-position", type=int, default=20)
    parser.add_argument("--bins-ratio", type=int, default=30)
    parser.add_argument("--bins-annotation-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        annotation_mode=args.annotation_mode,
        long_answer_chars_bin_size=args.long_answer_chars_bin_size,
        long_answer_tokens_bin_size=args.long_answer_tokens_bin_size,
        bins_short_answer_tokens=args.bins_short_answer_tokens,
        bins_short_answer_chars=args.bins_short_answer_chars,
        bins_question_chars=args.bins_question_chars,
        bins_relative_position=args.bins_relative_position,
        bins_ratio=args.bins_ratio,
        bins_annotation_count=args.bins_annotation_count,
    )

    ensure_dir(config.output_dir)

    analytics = NQDatasetAnalytics(config)
    analytics.process_folder()
    analytics.save_outputs()

    print("[DONE] Аналитика завершена.")
    print(f"[DONE] Результаты сохранены в: {config.output_dir}")
    print(f"[DONE] Режим аннотаций: {config.annotation_mode}")


if __name__ == "__main__":
    main()