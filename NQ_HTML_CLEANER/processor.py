from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator

from tqdm import tqdm

from remap import (
    build_token_remap,
    remap_span,
    validate_answer_span,
)


def get_example_id(example: Dict[str, Any]) -> str:
    """
    Возвращает строковый идентификатор примера.

    Приоритет выбора:
        1. example_id
        2. id
        3. document_title
        4. "unknown"

    Аргументы:
        example (Dict[str, Any]): запись датасета.

    Возвращает:
        str: идентификатор примера.
    """
    return str(
        example.get("example_id")
        or example.get("id")
        or example.get("document_title")
        or "unknown"
    )


def log_jsonl(logger_obj, payload: Dict[str, Any]) -> None:
    """
    Записывает словарь в JSONL-логгер, если логгер задан.

    Аргументы:
        logger_obj: объект JSONL-логгера или None.
        payload (Dict[str, Any]): данные для записи.
    """
    if logger_obj is not None:
        logger_obj.write(payload)


def drop_unused_fields(example: Dict[str, Any]) -> None:
    """
    Удаляет из примера поля, не используемые в очищенной версии датасета.

    Аргументы:
        example (Dict[str, Any]): запись датасета.
    """
    example.pop("document_html", None)
    example.pop("long_answer_candidates", None)


def clean_span_fields(span: Dict[str, Any]) -> None:
    """
    Удаляет из span-представления поля, связанные с байтовыми смещениями.

    После удаления HTML-токенов байтовые позиции более не соответствуют новой
    токенизации, поэтому такие поля удаляются.

    Аргументы:
        span (Dict[str, Any]): словарь со span-описанием.
    """
    span.pop("start_byte", None)
    span.pop("end_byte", None)
    span.pop("text", None)


def has_yes_no_answer(example: Dict[str, Any]) -> bool:
    """
    Проверяет, содержит ли пример ответы типа YES или NO.

    Аргументы:
        example (Dict[str, Any]): запись датасета.

    Возвращает:
        bool: True, если найден YES/NO ответ, иначе False.
    """
    for ann in example.get("annotations", []):
        if ann.get("yes_no_answer", "NONE") in {"YES", "NO"}:
            return True
    return False


def has_valid_long_answer(example: Dict[str, Any]) -> bool:
    """
    Проверяет, содержит ли пример хотя бы один корректный long answer.

    Аргументы:
        example (Dict[str, Any]): запись датасета.

    Возвращает:
        bool: True, если найден корректный long answer, иначе False.
    """
    for ann in example.get("annotations", []):
        long_answer = ann.get("long_answer", {})
        if not isinstance(long_answer, dict):
            continue

        start_token = long_answer.get("start_token", -1)
        end_token = long_answer.get("end_token", -1)

        if start_token >= 0 and end_token >= 0 and end_token > start_token:
            return True

    return False


def has_valid_short_answer(example: Dict[str, Any]) -> bool:
    """
    Проверяет, содержит ли пример хотя бы один корректный short answer.

    Аргументы:
        example (Dict[str, Any]): запись датасета.

    Возвращает:
        bool: True, если найден корректный short answer, иначе False.
    """
    for ann in example.get("annotations", []):
        short_answers = ann.get("short_answers", [])

        if not isinstance(short_answers, list):
            continue

        for sa in short_answers:
            if not isinstance(sa, dict):
                continue

            start_token = sa.get("start_token", -1)
            end_token = sa.get("end_token", -1)

            if start_token >= 0 and end_token > start_token:
                return True

    return False


def remap_annotation_span(
    *,
    span_obj: Dict[str, Any],
    span_type: str,
    old_to_new,
    clean_len: int,
    stats: Dict[str, int],
    validation_enabled: bool = True,
) -> None:
    """
    Пересчитывает индексы одного span после удаления HTML-токенов.

    Функция:
        - пересчитывает start_token и end_token;
        - удаляет устаревшие байтовые поля;
        - обновляет статистику;
        - при включённой валидации учитывает ошибки корректности span.

    Аргументы:
        span_obj (Dict[str, Any]): объект span для пересчёта.
        span_type (str): тип span для статистики.
        old_to_new: отображение старых индексов токенов в новые.
        clean_len (int): длина очищенного списка токенов.
        stats (Dict[str, int]): словарь агрегированной статистики.
        validation_enabled (bool): флаг включения валидации.
    """
    old_start = span_obj.get("start_token")
    old_end = span_obj.get("end_token")

    new_start, new_end, status = remap_span(old_start, old_end, old_to_new)

    span_obj["start_token"] = new_start
    span_obj["end_token"] = new_end
    clean_span_fields(span_obj)

    if status == "empty_after_html_removal":
        if span_type == "annotation_long_answer":
            stats["annotation_long_answer_empty_after_html"] += 1
        elif span_type == "annotation_short_answer":
            stats["annotation_short_answer_empty_after_html"] += 1

    if validation_enabled:
        errors = validate_answer_span(new_start, new_end, clean_len)
        if errors:
            stats["validation_errors"] += 1


def process_example(
    example: Dict[str, Any],
    stats: Dict[str, int],
    token_stats_logger=None,
    validation_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Обрабатывает один пример датасета.

    Этапы обработки:
        - проверка ограничений на тип примера;
        - удаление неиспользуемых полей;
        - удаление HTML-токенов;
        - пересчёт индексов long answer и short answer;
        - обновление статистики.

    Аргументы:
        example (Dict[str, Any]): исходный пример.
        stats (Dict[str, int]): словарь общей статистики.
        token_stats_logger: JSONL-логгер статистики токенов.
        validation_enabled (bool): флаг включения валидации.

    Возвращает:
        Dict[str, Any]: обработанный пример.

    Исключения:
        ValueError: если пример не удовлетворяет условиям отбора.
        KeyError: если отсутствует поле document_tokens.
    """
    example_id = get_example_id(example)

    if has_yes_no_answer(example):
        raise ValueError("example_has_yes_no_answer")

    if not has_valid_long_answer(example):
        raise ValueError("example_has_no_valid_long_answer")

    if not has_valid_short_answer(example):
        raise ValueError("example_has_no_valid_short_answer")

    drop_unused_fields(example)

    if "document_tokens" not in example:
        raise KeyError("Field 'document_tokens' not found")

    original_tokens = example["document_tokens"]
    remap_result = build_token_remap(original_tokens)

    clean_tokens = remap_result.clean_tokens
    old_to_new = remap_result.old_to_new
    clean_len = len(clean_tokens)

    example["document_tokens"] = clean_tokens

    stats["tokens_before_total"] += remap_result.tokens_before
    stats["tokens_after_total"] += remap_result.tokens_after
    stats["html_tokens_removed_total"] += remap_result.removed_html_count

    log_jsonl(token_stats_logger, {
        "example_id": example_id,
        "tokens_before": remap_result.tokens_before,
        "tokens_after": remap_result.tokens_after,
        "html_tokens_removed": remap_result.removed_html_count,
    })

    for ann in example.get("annotations", []):
        long_answer = ann.get("long_answer")

        if isinstance(long_answer, dict):
            remap_annotation_span(
                span_obj=long_answer,
                span_type="annotation_long_answer",
                old_to_new=old_to_new,
                clean_len=clean_len,
                stats=stats,
                validation_enabled=validation_enabled,
            )
            stats["long_answers_remapped"] += 1

        short_answers = ann.get("short_answers", [])
        if isinstance(short_answers, list):
            for short_answer in short_answers:
                if not isinstance(short_answer, dict):
                    continue

                remap_annotation_span(
                    span_obj=short_answer,
                    span_type="annotation_short_answer",
                    old_to_new=old_to_new,
                    clean_len=clean_len,
                    stats=stats,
                    validation_enabled=validation_enabled,
                )
                stats["short_answers_remapped"] += 1

    stats["examples_processed"] += 1
    return example


def process_records(
    records: Iterable[Dict[str, Any]],
    stats: Dict[str, int],
    token_stats_logger=None,
    validation_enabled: bool = True,
    tqdm_desc: str = "Processing",
) -> Iterator[Dict[str, Any]]:
    """
    Последовательно обрабатывает набор примеров датасета.

    Ошибки обработки отдельных примеров не прерывают общий процесс:
    такие примеры учитываются в статистике как failed и пропускаются.

    Аргументы:
        records (Iterable[Dict[str, Any]]): входные примеры.
        stats (Dict[str, int]): словарь общей статистики.
        token_stats_logger: JSONL-логгер статистики токенов.
        validation_enabled (bool): флаг включения валидации.
        tqdm_desc (str): подпись progress bar.

    Возвращает:
        Iterator[Dict[str, Any]]: поток успешно обработанных примеров.
    """
    for example in tqdm(records, desc=tqdm_desc, unit="example"):
        stats["examples_total"] += 1
        try:
            yield process_example(
                example=example,
                stats=stats,
                token_stats_logger=token_stats_logger,
                validation_enabled=validation_enabled,
            )
        except Exception:
            stats["examples_failed"] += 1