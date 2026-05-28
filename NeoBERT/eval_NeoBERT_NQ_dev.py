"""
Evaluation script for NeoBERT extractive QA on Natural Questions.

Поддерживает:
- NeoBERT с кастомной QA-head
- HuggingFace QA models
- EM / F1 / Recall
- Bootstrap confidence intervals
- Long context
- Sliding window

Запуск:

python eval_neobert.py \
  --config configs/neobert.yaml \
  --model_path /path/to/checkpoint \
  --output_dir ./eval_results
"""

from __future__ import annotations

import os
import json
import gzip
import glob
import random
import re
import string
import argparse

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple, Optional
from collections import Counter

import yaml
import numpy as np
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoConfig,
    DataCollatorWithPadding,
    Trainer,
    PreTrainedModel,
)

from transformers.modeling_outputs import QuestionAnsweringModelOutput

# ============================================================
# Config
# ============================================================


@dataclass
class EvalConfig:
    model_name: str
    validation_file: str
    output_dir: str

    max_seq_length: int
    max_query_length: int
    doc_stride: int

    n_best_size: int
    max_answer_length: int

    per_device_eval_batch_size: int
    dataloader_num_workers: int

    seed: int = 42
    bootstrap_samples: int = 1000
    allow_no_answer: bool = True


def load_config(path: str) -> EvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return EvalConfig(
        model_name=data["model_name"],
        validation_file=data["validation_file"],
        output_dir=data["output_dir"],
        max_seq_length=data["max_seq_length"],
        max_query_length=data["max_query_length"],
        doc_stride=data["doc_stride"],
        n_best_size=data["n_best_size"],
        max_answer_length=data["max_answer_length"],
        per_device_eval_batch_size=data["per_device_eval_batch_size"],
        dataloader_num_workers=data["dataloader_num_workers"],
        seed=data.get("seed", 42),
        bootstrap_samples=data.get("bootstrap_samples", 1000),
        allow_no_answer=data.get("allow_no_answer", True),
    )


# ============================================================
# Utils
# ============================================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, items: List[Any]):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# NeoBERT QA wrapper
# ============================================================


class NeoBERTForQuestionAnswering(PreTrainedModel):

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
            raise ValueError("NeoBERT config does not contain hidden_size")

        self.qa_outputs = torch.nn.Linear(hidden_size, self.num_labels)

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


def is_neobert_model(model_path: str) -> bool:
    return "neobert" in model_path.lower()


# ============================================================
# I/O
# ============================================================


def iter_jsonl_or_gz(path: str):

    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"

    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                yield json.loads(line)


def resolve_nq_input_files(path: str):

    normalized = os.path.expanduser(path)

    if os.path.isfile(normalized):
        return [normalized]

    if os.path.isdir(normalized):

        matched = []

        for pattern in ["*.jsonl.gz", "*.jsonl"]:
            matched.extend(glob.glob(os.path.join(normalized, pattern)))

        files = sorted({os.path.normpath(p) for p in matched if os.path.isfile(p)})

        return files

    raise FileNotFoundError(path)


# ============================================================
# Preprocessing
# ============================================================


def get_doc_tokens(example):
    return [t["token"] for t in example["document_tokens"]]


def normalize_nq_eval_examples(example):

    results = []

    try:

        doc_tokens = get_doc_tokens(example)

        question_tokens = example.get("question_tokens")

        if not question_tokens:
            question_tokens = example["question_text"].split()

        original_example_id = str(example["example_id"])

        for ann_idx, ann in enumerate(example["annotations"]):

            long_answer = ann.get("long_answer", {})
            short_answers = ann.get("short_answers", [])

            long_start = int(long_answer.get("start_token", -1))
            long_end = int(long_answer.get("end_token", -1))

            if long_start < 0 or long_end <= long_start:
                continue

            context_tokens = doc_tokens[long_start:long_end]

            gold_spans_local = []

            for sa in short_answers:

                s = int(sa["start_token"]) - long_start
                e = int(sa["end_token"]) - long_start - 1

                if 0 <= s <= e < len(context_tokens):
                    gold_spans_local.append((s, e))

            if not gold_spans_local:
                continue

            results.append(
                {
                    "example_id": f"{original_example_id}__ann{ann_idx}",
                    "original_example_id": original_example_id,
                    "annotation_idx": ann_idx,
                    "question_tokens": question_tokens,
                    "context_tokens": context_tokens,
                    "train_span": gold_spans_local[0],
                    "gold_spans": gold_spans_local,
                }
            )

    except Exception:
        return []

    return results


def load_nq_eval_examples(path):

    files = resolve_nq_input_files(path)

    kept = []

    for fp in files:

        for ex in iter_jsonl_or_gz(fp):

            kept.extend(normalize_nq_eval_examples(ex))

    return kept


# ============================================================
# Tokenization
# ============================================================


def _locate_answer_tokens(
    word_ids,
    sequence_ids,
    gold_start,
    gold_end,
):

    token_start = None
    token_end = None

    for idx, (sid, wid) in enumerate(zip(sequence_ids, word_ids)):

        if sid != 1 or wid is None:
            continue

        if wid == gold_start and token_start is None:
            token_start = idx

    for idx in range(len(word_ids) - 1, -1, -1):

        if sequence_ids[idx] == 1 and word_ids[idx] == gold_end:
            token_end = idx
            break

    if token_start is None or token_end is None:
        return None

    return token_start, token_end


def build_eval_features(
    examples,
    tokenizer,
    cfg,
):

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

    start_positions = []
    end_positions = []

    features_meta = []

    for i in range(len(tokenized["input_ids"])):

        sample_idx = sample_mapping[i]

        ex = examples[sample_idx]

        input_ids = tokenized["input_ids"][i]

        cls_token_id = tokenizer.cls_token_id

        if cls_token_id is not None and cls_token_id in input_ids:
            cls_index = input_ids.index(cls_token_id)
        else:
            cls_index = 0

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

        answer_pos = _locate_answer_tokens(
            word_ids,
            sequence_ids,
            gold_start,
            gold_end,
        )

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


def span_to_text(span, context_tokens):

    if span is None:
        return ""

    s, e = span

    if s < 0 or e >= len(context_tokens):
        return ""

    return " ".join(context_tokens[s : e + 1]).strip()


def normalize_answer(text):

    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def squad_exact_match_score(prediction, ground_truth):

    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def squad_f1_score(prediction, ground_truth):

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


def squad_recall_score(prediction, ground_truth):

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0

    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)

    overlap = sum(common.values())

    return overlap / len(gold_tokens)


# ============================================================
# Postprocessing
# ============================================================


def top_k_indices(logits, k):

    if len(logits) <= k:
        return np.argsort(logits)[::-1].tolist()

    idx = np.argpartition(logits, -k)[-k:]

    return idx[np.argsort(logits[idx])[::-1]].tolist()


def postprocess_predictions(
    examples,
    features_meta,
    raw_predictions,
    cfg,
):

    start_logits, end_logits = raw_predictions

    best_predictions = {}

    for feature_idx, meta in enumerate(features_meta):

        ex_id = meta["example_id"]

        word_ids = meta["word_ids"]
        context_mask = meta["context_mask"]

        s_logits = start_logits[feature_idx]
        e_logits = end_logits[feature_idx]

        best_score = -1e30
        best_span = None

        for s_idx in top_k_indices(
            s_logits,
            cfg.n_best_size,
        ):

            if context_mask[s_idx] == 0:
                continue

            for e_idx in range(
                s_idx,
                min(
                    s_idx + cfg.max_answer_length,
                    len(e_logits),
                ),
            ):

                if context_mask[e_idx] == 0:
                    continue

                if word_ids[s_idx] == -1:
                    continue

                if word_ids[e_idx] == -1:
                    continue

                if word_ids[e_idx] < word_ids[s_idx]:
                    continue

                score = s_logits[s_idx] + e_logits[e_idx]

                if score > best_score:

                    best_score = score

                    best_span = (
                        word_ids[s_idx],
                        word_ids[e_idx],
                    )

        prev = best_predictions.get(ex_id)

        if prev is None or best_score > prev["score"]:
            best_predictions[ex_id] = {
                "span": best_span,
                "score": best_score,
            }

    return {k: v["span"] for k, v in best_predictions.items()}


# ============================================================
# Evaluation
# ============================================================


def evaluate_predictions(examples, predictions):

    grouped_scores = {}

    for ex in examples:

        ex_id = ex["example_id"]

        original_example_id = ex["original_example_id"]

        pred_span = predictions.get(ex_id)

        context_tokens = ex["context_tokens"]

        gold_texts = [span_to_text(g, context_tokens) for g in ex["gold_spans"]]

        pred_text = span_to_text(
            pred_span,
            context_tokens,
        )

        em = max(squad_exact_match_score(pred_text, gt) for gt in gold_texts)

        f1 = max(squad_f1_score(pred_text, gt) for gt in gold_texts)

        recall = max(squad_recall_score(pred_text, gt) for gt in gold_texts)

        prev = grouped_scores.get(original_example_id)

        if prev is None:
            grouped_scores[original_example_id] = {
                "exact_match": em,
                "f1": f1,
                "token_recall": recall,
            }
        else:
            prev["exact_match"] = max(prev["exact_match"], em)
            prev["f1"] = max(prev["f1"], f1)
            prev["token_recall"] = max(prev["token_recall"], recall)

    em_scores = [v["exact_match"] for v in grouped_scores.values()]
    f1_scores = [v["f1"] for v in grouped_scores.values()]
    recall_scores = [v["token_recall"] for v in grouped_scores.values()]

    return {
        "exact_match": float(np.mean(em_scores)) * 100.0,
        "f1": float(np.mean(f1_scores)) * 100.0,
        "token_recall": float(np.mean(recall_scores)) * 100.0,
        "n_examples": len(grouped_scores),
    }


# ============================================================
# Bootstrap CI
# ============================================================


def bootstrap_confidence_intervals(
    values,
    n_samples=1000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    values = np.array(values)

    boot = []

    for _ in range(n_samples):

        sample_idx = rng.integers(
            0,
            len(values),
            size=len(values),
        )

        boot.append(float(np.mean(values[sample_idx])) * 100.0)

    return [
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = args.output_dir or cfg.output_dir

    ensure_dir(output_dir)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )

    print("Loading model...")

    if is_neobert_model(args.model_path):

        print("NeoBERT detected")

        config = AutoConfig.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )

        config.name_or_path = args.model_path

        model = NeoBERTForQuestionAnswering(config)

        safetensor_path = os.path.join(
            args.model_path,
            "model.safetensors",
        )

        if os.path.exists(safetensor_path):

            from safetensors.torch import load_file

            state_dict = load_file(safetensor_path)

        else:

            state_dict = torch.load(
                os.path.join(
                    args.model_path,
                    "pytorch_model.bin",
                ),
                map_location="cpu",
            )

        model.load_state_dict(state_dict)

    else:

        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )

    print("Loading eval examples...")

    eval_examples = load_nq_eval_examples(cfg.validation_file)

    print("Tokenizing...")

    eval_dataset, eval_features_meta = build_eval_features(
        eval_examples,
        tokenizer,
        cfg,
    )

    print(f"Eval examples: {len(eval_examples)}")
    print(f"Eval features: {len(eval_dataset)}")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
    )

    print("Running prediction...")

    pred_output = trainer.predict(
        eval_dataset,
    )

    predictions = postprocess_predictions(
        eval_examples,
        eval_features_meta,
        pred_output.predictions,
        cfg,
    )

    metrics = evaluate_predictions(
        eval_examples,
        predictions,
    )

    save_json(
        os.path.join(output_dir, "metrics.json"),
        metrics,
    )

    print("Done.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
