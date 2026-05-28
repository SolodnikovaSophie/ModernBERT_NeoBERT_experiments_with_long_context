"""
Загрузка Natural Questions (dev) + утилиты для работы с документами,
аннотациями и токенами.

Фильтрация: пропускаем варианты аннотаций, у которых нет одновременно
валидного long_answer И валидного short_answer.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional


def iter_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_nq_examples(input_path: str) -> Iterator[Dict[str, Any]]:
    """
    Принимает либо путь к одному файлу *.jsonl.gz,
    либо директорию (в которой будут отсортированы и прочитаны все *.jsonl.gz).
    """

    path = Path(input_path)
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl.gz"))

    if not files:
        raise FileNotFoundError(f"No NQ files found at: {input_path}")

    for file_path in files:
        logging.info(f"Reading file: {file_path}")
        for example in iter_jsonl_gz(file_path):
            yield example


def get_question_id(example: Dict[str, Any]) -> str:
    return str(
        example.get("example_id")
        or example.get("question_id")
        or example.get("id")
    )


def get_question_text(example: Dict[str, Any]) -> str:
    return example.get(
        "question_text",
        example.get("question", " ".join(example.get("question_tokens", []))),
    )


def get_doc_tokens(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Возвращает список dict'ов: {"token": str, "html_token": bool}.
    Сохраняем разметку html_token, чтобы при сборке текста чанков
    исключать html-теги.
    """

    out = []
    for t in example.get("document_tokens", []):
        if isinstance(t, dict):
            out.append({"token": str(t.get("token", "")), "html_token": bool(t.get("html_token", False))})
        else:
            out.append({"token": str(t), "html_token": False})
    return out


def is_valid_span(span: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(span, dict):
        return False
    s = span.get("start_token", -1)
    e = span.get("end_token", -1)
    return s >= 0 and e > s


def extract_text_from_tokens(
    doc_tokens: List[Dict[str, Any]],
    start: int,
    end: int,
    skip_html: bool = True,
) -> str:
    parts = []
    for tok in doc_tokens[start:end]:
        if skip_html and tok["html_token"]:
            continue
        parts.append(tok["token"])
    return " ".join(parts).strip()


def filter_valid_annotations(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Оставляем только аннотации, у которых одновременно:
      - валидный long_answer,
      - есть хотя бы один валидный short_answer.
    Прочее (yes/no, no-answer) пропускаем.
    """

    valid = []
    for ann in example.get("annotations", []):
        if not is_valid_span(ann.get("long_answer")):
            continue
        shorts = [s for s in (ann.get("short_answers") or []) if is_valid_span(s)]
        if not shorts:
            continue
        valid.append({"long_answer": ann["long_answer"], "short_answers": shorts})
    return valid


def collect_gold_spans(annotations: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    """
    Возвращает список всех валидных short-answer span'ов для вопроса.
    """

    spans = []
    for ann in annotations:
        for sa in ann["short_answers"]:
            spans.append({"start_token": sa["start_token"], "end_token": sa["end_token"]})
    return spans
