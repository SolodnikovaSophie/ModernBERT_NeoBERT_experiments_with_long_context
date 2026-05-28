"""
Sliding window chunker с учётом границ предложений.

Идея:
  - документ задан списком word-токенов (NQ document_tokens),
  - режем на чанки длиной примерно window_size *под-токенов модели*,
  - стараемся не разрывать предложения: границы окна "доводим" до конца
    ближайшего предложения (после '.', '!' или '?'),
  - перекрытие между соседними окнами задаётся overlap_ratio
    (доля от window_size).

Метка чанка:
  - 1, если хотя бы один из gold short-answer span'ов целиком лежит
    внутри диапазона [chunk_start_word, chunk_end_word),
  - иначе 0.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


SENT_END_TOKENS = {".", "!", "?"}


@dataclass
class Chunk:
    chunk_id: int
    word_start: int            # включительно (в индексах document_tokens)
    word_end: int              # исключительно
    text: str                  # «чистый» текст (без html-токенов)
    n_subtokens: int           # длина в под-токенах токенайзера (без [CLS]/[SEP])
    label: int = 0             # 1, если содержит целиком хотя бы один gold short span
    contained_gold: List[Dict[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "n_subtokens": self.n_subtokens,
            "label": self.label,
            "contained_gold": self.contained_gold,
            "text_preview": (self.text[:200] + "…") if len(self.text) > 200 else self.text,
        }


def _count_subtokens(tokenizer, words: List[str]) -> int:
    """
    Сколько под-токенов даёт токенайзер на этом списке слов
    (без спец-токенов).
    """

    if not words:
        return 0
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return len(enc["input_ids"])


def _is_sentence_end(token_str: str) -> bool:
    if not token_str:
        return False
    if token_str in SENT_END_TOKENS:
        return True
    # «word.» / «word!» — точка/знак приклеена к слову
    return token_str[-1] in SENT_END_TOKENS


def _find_sentence_breaks(
    doc_tokens: List[Dict[str, Any]],
    skip_html: bool = True,
) -> List[int]:
    """
    Возвращает отсортированный список индексов *после которых* можно
    закрыть предложение, т.е. позиции, где следующий чанк имеет право
    начаться. Индексы в шкале document_tokens; конец предложения —
    индекс i означает, что предложение оканчивается на токене i
    (включительно), а следующий чанк начинается с i+1.
    """

    breaks: List[int] = []
    for i, t in enumerate(doc_tokens):
        if skip_html and t["html_token"]:
            continue
        if _is_sentence_end(t["token"]):
            breaks.append(i)
    return breaks


def _build_text(doc_tokens: List[Dict[str, Any]], start: int, end: int, skip_html: bool = True) -> str:
    parts = []
    for tok in doc_tokens[start:end]:
        if skip_html and tok["html_token"]:
            continue
        parts.append(tok["token"])
    return " ".join(parts).strip()


def _next_sentence_boundary(breaks: List[int], pos: int) -> Optional[int]:
    """
    Минимальный break >= pos. Если нет — None.
    """
    lo, hi = 0, len(breaks)
    while lo < hi:
        mid = (lo + hi) // 2
        if breaks[mid] < pos:
            lo = mid + 1
        else:
            hi = mid
    return breaks[lo] if lo < len(breaks) else None


def _prev_sentence_boundary(breaks: List[int], pos: int) -> Optional[int]:
    """
    Максимальный break <= pos. Если нет — None.
    """
    lo, hi = 0, len(breaks)
    while lo < hi:
        mid = (lo + hi) // 2
        if breaks[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return breaks[lo - 1] if lo > 0 else None


def create_chunks(
    tokenizer,
    doc_tokens: List[Dict[str, Any]],
    window_size: int,
    overlap_ratio: float = 0.10,
    gold_spans: Optional[List[Dict[str, int]]] = None,
    question_text: Optional[str] = None,
    skip_html: bool = True,
) -> List[Chunk]:
    """
    Разбивает документ на чанки методом sliding window с учётом
    границ предложений и помечает их меткой 0/1.

    Параметры:
      window_size       — максимум под-токенов в одном чанке
                          (включая бюджет под пару вопрос+чанк, см. ниже).
      overlap_ratio     — доля перекрытия (overlap = window_size * ratio).
      gold_spans        — список {"start_token", "end_token"} для разметки.
      question_text     — если задан, бюджет окна уменьшается на длину
                          закодированного вопроса + спец-токены, чтобы
                          итоговая пара (Q, chunk) укладывалась в window_size.

    Возвращает список объектов Chunk.
    """

    if window_size <= 16:
        raise ValueError(f"window_size too small: {window_size}")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError(f"overlap_ratio out of range: {overlap_ratio}")

    # Бюджет под подтокены *чанка* (без вопроса и спец-токенов)
    if question_text is not None:
        q_len = _count_subtokens(tokenizer, question_text.split())
        # 4 запас на [CLS] Q [SEP] chunk [SEP]
        available = window_size - q_len - 4
    else:
        available = window_size - 2  # [CLS] chunk [SEP]

    if available <= 16:
        raise ValueError(
            f"Available chunk budget too small: {available} "
            f"(window_size={window_size}, question_len={q_len if question_text else 0})"
        )

    overlap_subtok = int(round(available * overlap_ratio))
    overlap_subtok = max(0, min(overlap_subtok, available - 1))

    breaks = _find_sentence_breaks(doc_tokens, skip_html=skip_html)
    n_words = len(doc_tokens)
    gold_spans = gold_spans or []

    chunks: List[Chunk] = []

    # Жадно нарезаем «по словам»: расширяем окно, пока не превысили бюджет,
    # затем отступаем до ближайшей границы предложения.
    word_start = 0
    chunk_id = 0

    # Кэшируем подтокен-счётчик инкрементально через бинарный поиск:
    # быстрее, чем токенизировать каждый прирост, и достаточно для дев-сета.
    while word_start < n_words:
        lo = word_start + 1
        hi = n_words
        # Ищем максимальный end, при котором подтокенов <= available.
        # Бинарный поиск + локальный экспоненциальный «прощуп».
        # Сначала попробуем «оценить» end эвристикой 1 слово ≈ 1.3 подтокена.
        approx = word_start + max(1, int(available / 1.3))
        approx = min(approx, n_words)

        # Расширяем вверх, пока влезаем
        cur = approx
        cur_sub = _count_subtokens(
            tokenizer,
            [doc_tokens[i]["token"] for i in range(word_start, cur) if not (skip_html and doc_tokens[i]["html_token"])],
        )

        if cur_sub <= available:
            # двигаемся вправо
            while cur < n_words:
                step = max(1, (n_words - cur) // 2)
                trial = min(n_words, cur + step)
                trial_sub = _count_subtokens(
                    tokenizer,
                    [doc_tokens[i]["token"] for i in range(word_start, trial) if not (skip_html and doc_tokens[i]["html_token"])],
                )
                if trial_sub <= available:
                    cur = trial
                    cur_sub = trial_sub
                    if cur == n_words:
                        break
                else:
                    # уменьшаем шаг до 1
                    if step == 1:
                        break
                    # точный бинарный поиск между cur и trial
                    a, b = cur, trial
                    while b - a > 1:
                        m = (a + b) // 2
                        m_sub = _count_subtokens(
                            tokenizer,
                            [doc_tokens[i]["token"] for i in range(word_start, m) if not (skip_html and doc_tokens[i]["html_token"])],
                        )
                        if m_sub <= available:
                            a = m
                        else:
                            b = m
                    cur = a
                    cur_sub = _count_subtokens(
                        tokenizer,
                        [doc_tokens[i]["token"] for i in range(word_start, cur) if not (skip_html and doc_tokens[i]["html_token"])],
                    )
                    break
        else:
            # сжимаем влево
            a, b = word_start + 1, cur
            while b - a > 1:
                m = (a + b) // 2
                m_sub = _count_subtokens(
                    tokenizer,
                    [doc_tokens[i]["token"] for i in range(word_start, m) if not (skip_html and doc_tokens[i]["html_token"])],
                )
                if m_sub <= available:
                    a = m
                else:
                    b = m
            cur = a
            cur_sub = _count_subtokens(
                tokenizer,
                [doc_tokens[i]["token"] for i in range(word_start, cur) if not (skip_html and doc_tokens[i]["html_token"])],
            )

        word_end = cur  # эксклюзивно

        # Корректируем границу окна до конца ближайшего предложения,
        # не выходя за бюджет.
        if word_end < n_words:
            prev_break = _prev_sentence_boundary(breaks, word_end - 1)
            if prev_break is not None and prev_break >= word_start:
                # граница в шкале «индекс токена-конца предложения»:
                # следующий чанк начнётся с prev_break + 1
                candidate_end = prev_break + 1
                if candidate_end - word_start >= 1:
                    word_end = candidate_end

        # Защита от пустого/однословного чанка
        if word_end <= word_start:
            word_end = min(n_words, word_start + 1)

        # Метка чанка
        label = 0
        contained = []
        for sp in gold_spans:
            if sp["start_token"] >= word_start and sp["end_token"] <= word_end:
                label = 1
                contained.append(sp)

        text = _build_text(doc_tokens, word_start, word_end, skip_html=skip_html)
        n_sub = _count_subtokens(
            tokenizer,
            [doc_tokens[i]["token"] for i in range(word_start, word_end) if not (skip_html and doc_tokens[i]["html_token"])],
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                word_start=word_start,
                word_end=word_end,
                text=text,
                n_subtokens=n_sub,
                label=label,
                contained_gold=contained,
            )
        )
        chunk_id += 1

        if word_end >= n_words:
            break

        # Старт следующего окна с учётом overlap.
        # Сначала прикинем «целевой» старт по подтокенам:
        target_subtok = max(1, cur_sub - overlap_subtok)
        # Найдём такой word_start_next, чтобы подтокенов между ним и word_end
        # было примерно overlap_subtok. Для скорости: оцениваем словами.
        # 1 подтокен ≈ 1/1.3 слова.
        overlap_words_est = max(1, int(overlap_subtok / 1.3))
        next_start = max(word_start + 1, word_end - overlap_words_est)

        # Сдвигаем next_start до начала следующего предложения,
        # если перекрытие "разрывает" предложение — но только если
        # это не уведёт нас вперёд за word_end.
        nxt_break = _next_sentence_boundary(breaks, next_start - 1)
        if nxt_break is not None and nxt_break + 1 < word_end:
            next_start = nxt_break + 1

        if next_start <= word_start:
            next_start = word_start + 1
        word_start = next_start

    return chunks
