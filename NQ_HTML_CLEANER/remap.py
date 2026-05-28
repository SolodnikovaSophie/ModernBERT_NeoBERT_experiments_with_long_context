from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RemapResult:
    """
    Хранит результат очистки токенов и отображение индексов.
    """
    clean_tokens: List[Dict[str, Any]]
    old_to_new: Dict[int, int]
    tokens_before: int
    tokens_after: int
    removed_html_count: int


def build_token_remap(tokens: List[Dict[str, Any]]) -> RemapResult:
    """
    Удаляет HTML-токены из document_tokens и строит отображение old_idx -> new_idx
    только для текстовых токенов.

    Аргументы:
        tokens (List[Dict[str, Any]]): исходный список токенов.

    Возвращает:
        RemapResult: результат очистки токенов и пересчёта индексов.
    """
    clean_tokens: List[Dict[str, Any]] = []
    old_to_new: Dict[int, int] = {}

    tokens_before = len(tokens)
    removed_html_count = 0

    for old_idx, token_obj in enumerate(tokens):
        if token_obj.get("html_token", False):
            removed_html_count += 1
            continue

        new_idx = len(clean_tokens)
        old_to_new[old_idx] = new_idx

        clean_tokens.append({
            "token": token_obj.get("token", ""),
            "html_token": False
        })

    return RemapResult(
        clean_tokens=clean_tokens,
        old_to_new=old_to_new,
        tokens_before=tokens_before,
        tokens_after=len(clean_tokens),
        removed_html_count=removed_html_count,
    )


def remap_span(
    start_token: Optional[int],
    end_token: Optional[int],
    old_to_new: Dict[int, int],
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Пересчитывает span [start_token, end_token) в новую систему индексов после
    удаления HTML-токенов.

    Аргументы:
        start_token (Optional[int]): начальная позиция span.
        end_token (Optional[int]): конечная позиция span.
        old_to_new (Dict[int, int]): отображение старых индексов в новые.

    Возвращает:
        Tuple[Optional[int], Optional[int], str]:
            новые границы span и статус пересчёта.
    """
    if start_token is None or end_token is None:
        return None, None, "null_span"

    if start_token == -1 and end_token == -1:
        return -1, -1, "no_answer"

    if start_token < 0 or end_token < 0 or end_token < start_token:
        return None, None, "invalid_range"

    surviving = [old_to_new[i] for i in range(start_token, end_token) if i in old_to_new]

    if not surviving:
        return None, None, "empty_after_html_removal"

    new_start = surviving[0]
    new_end = surviving[-1] + 1

    old_len = end_token - start_token
    new_len = len(surviving)

    if old_len == new_len and new_end - new_start == old_len:
        status = "unchanged"
    else:
        status = "shifted_or_compacted"

    return new_start, new_end, status


def extract_span_text(
    tokens: List[Dict[str, Any]],
    start_token: Optional[int],
    end_token: Optional[int],
    skip_html: bool = False,
) -> str:
    """
    Извлекает текстовое содержимое span по токенам.

    Если skip_html=True, HTML-токены пропускаются.

    Аргументы:
        tokens (List[Dict[str, Any]]): список токенов.
        start_token (Optional[int]): начальная позиция span.
        end_token (Optional[int]): конечная позиция span.
        skip_html (bool): признак пропуска HTML-токенов.

    Возвращает:
        str: текст span.
    """
    if start_token is None or end_token is None:
        return ""

    if start_token == -1 and end_token == -1:
        return ""

    parts: List[str] = []

    for i in range(start_token, end_token):
        if 0 <= i < len(tokens):
            token_obj = tokens[i]
            if skip_html and token_obj.get("html_token", False):
                continue
            parts.append(token_obj.get("token", ""))

    return " ".join(parts)


def validate_answer_span(
    start_token: Optional[int],
    end_token: Optional[int],
    tokens_len: int,
) -> List[str]:
    """
    Выполняет проверку корректности answer-span.

    Аргументы:
        start_token (Optional[int]): начальная позиция span.
        end_token (Optional[int]): конечная позиция span.
        tokens_len (int): длина последовательности токенов.

    Возвращает:
        List[str]: список кодов ошибок валидации.
    """
    errors: List[str] = []

    if start_token is None or end_token is None:
        errors.append("span_has_none")
        return errors

    if start_token == -1 and end_token == -1:
        return errors

    if start_token < 0 or end_token < 0:
        errors.append("negative_span")
    if end_token < start_token:
        errors.append("end_before_start")
    if start_token >= tokens_len:
        errors.append("start_out_of_bounds")
    if end_token > tokens_len:
        errors.append("end_out_of_bounds")
    if start_token == end_token:
        errors.append("empty_span")

    return errors