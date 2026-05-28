import gzip
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional


def iter_jsonl_gz(path: Path):
    """Итерируется по сжатому JSONL файлу."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_nq_examples(input_path: str):
    """Загружает примеры NQ из указанного пути."""
    path = Path(input_path)
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl.gz"))
    for file_path in files:
        logging.info(f"Reading file: {file_path}")
        for example in iter_jsonl_gz(file_path):
            yield example


def get_question_id(example: Dict[str, Any]) -> str:
    """Извлекает ID вопроса из примера."""
    return str(example.get("example_id") or example.get("question_id") or example.get("id"))


def get_question_text(example: Dict[str, Any]) -> str:
    """Извлекает текст вопроса."""
    return example.get("question_text", example.get("question", " ".join(example.get("question_tokens", []))))


def get_doc_tokens(example: Dict[str, Any]) -> List[str]:
    """Извлекает токены документа."""
    return [t["token"] if isinstance(t, dict) else str(t) for t in example["document_tokens"]]


def is_valid_span(span: Dict[str, Any]) -> bool:
    """Проверяет валидность индексов начала и конца."""
    return (isinstance(span, dict) and
            span.get("start_token", -1) >= 0 and
            span.get("end_token", -1) > span.get("start_token", -1))


def extract_answer_text(doc_tokens: List[str], start: int, end: int) -> str:
    """Преобразует индексы токенов в строку."""
    return " ".join(doc_tokens[start:end]).strip()


def encoded_length(tokenizer, question: str, context_tokens: List[str]) -> int:
    """Считает длину итоговой последовательности (Q + Context) в токенах модели."""
    encoded = tokenizer(question.split(), context_tokens, is_split_into_words=True, add_special_tokens=True,
                        truncation=False)
    return len(encoded["input_ids"])


def build_intra_doc_context(
        tokenizer,
        question: str,
        doc_tokens: List[str],
        short_start: int,
        short_end: int,
        max_seq_len: int,
        pos_pct: float,
        margin: float = 0.05
) -> Optional[List[str]]:
    """
    Вырезает окно из оригинального документа, размещая ответ в интервале [pos_pct - margin, pos_pct + margin].
    Гарантирует отсутствие жесткой привязки к одному индексу (jitter) и предотвращает
    размещение ответа в самых первых или последних токенах окна.
    """
    q_enc = tokenizer.encode(question, add_special_tokens=False)
    # Накладные расходы ModernBERT: [CLS] + Q + [SEP] + Context + [SEP]
    overhead = len(q_enc) + 4
    target_window_len = max_seq_len - overhead

    if len(doc_tokens) < target_window_len:
        return None  # Документ короче требуемого окна

    ans_len = short_end - short_start
    if ans_len > target_window_len:
        return None  # Ответ физически не помещается в окно

    # Определяем диапазон допустимых смещений начала ответа ВНУТРИ окна.
    # Чтобы избежать "прилипания", берем минимальный отступ 10 токенов от краев.
    min_offset = max(10, int(target_window_len * max(0.0, pos_pct - margin)))
    max_offset = int(target_window_len * min(1.0, pos_pct + margin)) - ans_len

    # Если интервал слишком узкий из-за длинного ответа, расширяем до доступного максимума
    if min_offset > max_offset:
        min_offset, max_offset = 10, target_window_len - ans_len - 10
        if min_offset > max_offset: return None  # Даже так не влезает

    # Случайно выбираем смещение внутри разрешенного интервала
    chosen_offset_in_window = random.randint(min_offset, max_offset)

    # Вычисляем начало окна в документе относительно позиции ответа
    w_start = short_start - chosen_offset_in_window

    # Корректируем, чтобы не выйти за границы документа
    w_start = max(0, min(w_start, len(doc_tokens) - target_window_len))
    w_end = w_start + target_window_len

    # Финальная проверка: ответ остался внутри выбранного окна
    if w_start <= short_start and w_end >= short_end:
        return doc_tokens[w_start:w_end]

    return None