"""
Cross-encoder инференс (ModernBERT и совместимые модели).

Модель — AutoModelForSequenceClassification с одной (релевантность,
регрессия / логит) или несколькими меткой (тогда берём положительный
класс). На вход — пара (question, chunk_text), на выход — скаляр-скор.
"""

import logging
import gc
from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_cross_encoder(model_name: str, device: torch.device, dtype: torch.dtype = torch.float32):
    """
    Загружает предобученный cross-encoder для ранжирования.
    Поддерживает ModernBERT и любые AutoModelForSequenceClassification.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device).eval()
    return tokenizer, model


def _logits_to_score(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, num_labels].
      num_labels == 1 -> сам логит.
      num_labels == 2 -> softmax, берём вероятность положительного класса.
      num_labels >  2 -> то же, что выше, но индекс 1.
    """
    if logits.ndim == 1:
        return logits
    if logits.size(-1) == 1:
        return logits.squeeze(-1)
    return torch.softmax(logits, dim=-1)[:, 1]


@torch.no_grad()
def score_pairs_batched(
    model,
    tokenizer,
    questions: List[str],
    chunk_texts: List[str],
    device: torch.device,
    max_seq_length: int,
    batch_size: int,
) -> List[float]:
    """
    Возвращает список релевантности (float) для пар (q_i, chunk_i).
    Падает в OOM-фолбэк делением батча пополам.
    """

    assert len(questions) == len(chunk_texts)
    scores: List[float] = []

    i = 0
    n = len(questions)
    while i < n:
        bs = min(batch_size, n - i)
        q_b = questions[i : i + bs]
        c_b = chunk_texts[i : i + bs]
        try:
            enc = tokenizer(
                q_b,
                c_b,
                max_length=max_seq_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            s = _logits_to_score(out.logits).float().cpu().tolist()
            scores.extend(s)
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if bs == 1:
                logging.error(
                    f"OOM at batch_size=1 (seq_len={max_seq_length}); pushing -inf score."
                )
                scores.append(float("-inf"))
                i += 1
            else:
                # уменьшаем batch_size на лету
                new_bs = max(1, bs // 2)
                logging.warning(
                    f"OOM at batch_size={bs}, retrying with {new_bs}"
                )
                batch_size = new_bs

    return scores
