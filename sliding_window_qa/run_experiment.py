"""
Двухступенчатый пайплайн на Natural Questions (dev):

  Stage 1 — Cross-Encoder Reranker (--reranker-model-name):
            пара (question, chunk) -> скор релевантности.
            Метрики: MRR, Recall@K.

  Stage 2 — Span Extractor (--span-model-name, AutoModelForQuestionAnswering):
            прогон топ-K чанков из Stage 1, извлечение лучшего
            (start, end) span'а по сумме start+end логитов.
            Метрики: Exact Match, F1, Token Recall (vs gold short answer).

Оба пути принимаются как HF id ИЛИ как локальная папка с весами.

Пример запуска (PowerShell):

  python -m new_sliding_window.run_experiment ^
      --input-path "D:/data/nq/v1.0-simplified-nq-dev-all.jsonl.gz" ^
      --reranker-model-name "Alibaba-NLP/gte-reranker-modernbert-base" ^
      --span-model-name    "C:/models/modernbert-qa-nq" ^
      --window-sizes 512 2048 ^
      --overlap-ratios 0.10 ^
      --batch-size 16 --span-batch-size 8 --span-top-k 3
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .config import parse_args
from .data_utils import (
    iter_nq_examples,
    get_question_id,
    get_question_text,
    get_doc_tokens,
    filter_valid_annotations,
    collect_gold_spans,
    extract_text_from_tokens,
)
from .chunker import create_chunks
from .inference import load_cross_encoder, score_pairs_batched
from .span_extractor import load_span_extractor, extract_spans_for_top_k, pick_best_span
from .metrics import (
    rank_chunks,
    reciprocal_rank,
    recall_at_k,
    best_span_metrics_for_query,
    extracted_span_metrics_for_query,
    aggregate,
)
from .memory_tracker import MemoryTracker


def setup_logger(log_file: Path) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def run_one_setting(
    config,
    rer_tokenizer,
    rer_model,
    span_tokenizer,
    span_model,
    device,
    window_size: int,
    overlap_ratio: float,
) -> None:
    setting_dir = (
        config.output_dir
        / f"win_{window_size}_ovr_{int(round(overlap_ratio * 100)):02d}"
    )
    setting_dir.mkdir(parents=True, exist_ok=True)

    mem = MemoryTracker(setting_dir)
    mem.clear_memory()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem.log_memory("setup", "start")

    preds_file = open(setting_dir / "predictions.jsonl", "w", encoding="utf-8")

    per_query: dict = {}
    n_queries_seen = 0
    n_examples_skipped = 0
    total_chunks = 0
    total_pos_chunks = 0

    chunking_seconds = 0.0
    rerank_seconds = 0.0
    span_seconds = 0.0

    progress = tqdm(
        iter_nq_examples(config.input_path),
        desc=f"win={window_size} ovr={overlap_ratio:.2f}",
    )

    for example in progress:
        if config.max_examples and n_queries_seen >= config.max_examples:
            break

        annotations = filter_valid_annotations(example)
        if not annotations:
            n_examples_skipped += 1
            continue

        qid = get_question_id(example)
        q_text = get_question_text(example)
        doc_tokens = get_doc_tokens(example)
        gold_spans = collect_gold_spans(annotations)
        gold_texts = list({
            extract_text_from_tokens(doc_tokens, sp["start_token"], sp["end_token"])
            for sp in gold_spans
        })

        # ---------- chunking ----------
        t0 = time.time()
        try:
            chunks = create_chunks(
                tokenizer=rer_tokenizer,
                doc_tokens=doc_tokens,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                gold_spans=gold_spans,
                question_text=q_text,
                skip_html=True,
            )
        except ValueError as e:
            logging.warning(f"Skip qid={qid}: {e}")
            n_examples_skipped += 1
            continue
        chunking_seconds += time.time() - t0

        if not chunks:
            n_examples_skipped += 1
            continue

        # ---------- Stage 1: rerank ----------
        t0 = time.time()
        scores = score_pairs_batched(
            model=rer_model,
            tokenizer=rer_tokenizer,
            questions=[q_text] * len(chunks),
            chunk_texts=[c.text for c in chunks],
            device=device,
            max_seq_length=window_size,
            batch_size=config.batch_size,
        )
        rerank_seconds += time.time() - t0

        chunk_records = [
            {
                "chunk_id": c.chunk_id,
                "word_start": c.word_start,
                "word_end": c.word_end,
                "n_subtokens": c.n_subtokens,
                "label": c.label,
                "score": float(s),
                "text": c.text,
            }
            for c, s in zip(chunks, scores)
        ]

        ranked = rank_chunks(chunk_records)
        has_pos = int(any(c["label"] == 1 for c in chunk_records))
        mrr = reciprocal_rank(ranked)
        rec_at = {k: recall_at_k(ranked, k) for k in config.ks}

        # «Coarse» span-метрики: топ-1 чанк целиком как «ответ».
        coarse = best_span_metrics_for_query(ranked, gold_texts)

        # ---------- Stage 2: span extraction over top-K ----------
        top_k_for_span = ranked[: config.span_top_k]
        t0 = time.time()
        span_preds = extract_spans_for_top_k(
            model=span_model,
            tokenizer=span_tokenizer,
            question=q_text,
            top_chunks=top_k_for_span,
            device=device,
            max_seq_length=config.span_max_seq_length,
            max_span_words=config.max_span_words,
            batch_size=config.span_batch_size,
        )
        span_seconds += time.time() - t0
        best = pick_best_span(span_preds)
        predicted_text = best.text if best is not None else ""
        extracted = extracted_span_metrics_for_query(predicted_text, gold_texts)

        # ---------- per-query log ----------
        q_metrics = {
            "mrr": mrr,
            **{f"r@{k}": rec_at[k] for k in config.ks},
            "coarse_em": coarse["exact_match"],
            "coarse_f1": coarse["f1"],
            "coarse_tr": coarse["token_recall"],
            "span_em": extracted["exact_match"],
            "span_f1": extracted["f1"],
            "span_tr": extracted["token_recall"],
            "has_positive": has_pos,
        }
        per_query[qid] = q_metrics

        n_queries_seen += 1
        total_chunks += len(chunks)
        total_pos_chunks += sum(c.label for c in chunks)

        row = {
            "question_id": qid,
            "question": q_text,
            "n_chunks": len(chunks),
            "n_positive_chunks": sum(c.label for c in chunks),
            "gold_short_answers": gold_texts,
            "ranked_top5": [
                {
                    "chunk_id": ch["chunk_id"],
                    "score": ch["score"],
                    "label": ch["label"],
                    "word_span": [ch["word_start"], ch["word_end"]],
                    "text_preview": (ch["text"][:300] + "…") if len(ch["text"]) > 300 else ch["text"],
                }
                for ch in ranked[:5]
            ],
            "ranking_metrics": {
                "mrr": mrr,
                **{f"recall@{k}": rec_at[k] for k in config.ks},
                "has_positive_chunk": has_pos,
            },
            "coarse_span_metrics_top1_chunk_text": {
                "exact_match": coarse["exact_match"],
                "f1": coarse["f1"],
                "token_recall": coarse["token_recall"],
            },
            "extracted_span": {
                "text": predicted_text,
                "score": (best.score if best is not None and best.score != float("-inf") else None),
                "from_chunk_id": (best.chunk_id if best is not None else None),
                "word_span_in_chunk": (
                    [best.word_start, best.word_end] if best is not None else None
                ),
                "metrics": {
                    "exact_match": extracted["exact_match"],
                    "f1": extracted["f1"],
                    "token_recall": extracted["token_recall"],
                },
            },
        }
        preds_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        if n_queries_seen % 50 == 0:
            mem.log_memory("inference", f"q{n_queries_seen}")

    preds_file.close()
    mem.log_memory("inference", "done")

    # ---------- aggregation + CI ----------
    rank_keys = ["mrr"] + [f"r@{k}" for k in config.ks]
    coarse_keys = ["coarse_em", "coarse_f1", "coarse_tr"]
    span_keys = ["span_em", "span_f1", "span_tr"]

    rank_agg = aggregate(
        per_query, rank_keys,
        config.bootstrap_samples, config.confidence_level, config.seed,
    )
    coarse_agg = aggregate(
        per_query, coarse_keys,
        config.bootstrap_samples, config.confidence_level, config.seed,
    )
    span_agg = aggregate(
        per_query, span_keys,
        config.bootstrap_samples, config.confidence_level, config.seed,
    )

    has_pos_share = (
        sum(v["has_positive"] for v in per_query.values()) / len(per_query)
        if per_query else 0.0
    )

    peak = mem.peak_stats.to_dict() if mem.peak_stats else {}

    metrics_out = {
        "setting": {
            "window_size": window_size,
            "overlap_ratio": overlap_ratio,
            "batch_size": config.batch_size,
            "span_batch_size": config.span_batch_size,
            "span_top_k": config.span_top_k,
            "span_max_seq_length": config.span_max_seq_length,
            "max_span_words": config.max_span_words,
            "reranker_model_name": config.reranker_model_name,
            "span_model_name": config.span_model_name,
            "ks": config.ks,
            "bootstrap_samples": config.bootstrap_samples,
            "confidence_level": config.confidence_level,
        },
        "counts": {
            "queries_evaluated": len(per_query),
            "examples_skipped_no_valid_ann": n_examples_skipped,
            "total_chunks": total_chunks,
            "total_positive_chunks": total_pos_chunks,
            "share_queries_with_positive_chunk": has_pos_share,
        },
        "ranking_metrics": rank_agg,
        "coarse_span_metrics_top1_chunk_text": coarse_agg,
        "extracted_span_metrics": span_agg,
        "performance": {
            "chunking_seconds": chunking_seconds,
            "rerank_seconds": rerank_seconds,
            "span_seconds": span_seconds,
            "total_seconds": chunking_seconds + rerank_seconds + span_seconds,
            "throughput_rerank_chunks_per_s": (total_chunks / rerank_seconds) if rerank_seconds > 0 else 0.0,
            "throughput_queries_per_s": (
                len(per_query) / (chunking_seconds + rerank_seconds + span_seconds)
            ) if (chunking_seconds + rerank_seconds + span_seconds) > 0 else 0.0,
        },
        "peak_memory": peak,
    }

    with open(setting_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    mem.save_log()

    def fmt(d):
        if not d:
            return "n/a"
        return f"{d['mean']*100:.2f} [{d['ci_low']*100:.2f}, {d['ci_high']*100:.2f}]"

    logging.info(
        "Done win=%d ovr=%.2f | MRR=%s R@1=%s R@3=%s R@5=%s | "
        "spanEM=%s spanF1=%s spanTR=%s",
        window_size, overlap_ratio,
        fmt(rank_agg.get("mrr")),
        fmt(rank_agg.get("r@1")),
        fmt(rank_agg.get("r@3")),
        fmt(rank_agg.get("r@5")),
        fmt(span_agg.get("span_em")),
        fmt(span_agg.get("span_f1")),
        fmt(span_agg.get("span_tr")),
    )


def main():
    config = parse_args()
    config.save()
    setup_logger(config.output_dir / "run.log")

    logging.info("Config: %s", json.dumps(config.__dict__, ensure_ascii=False))

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.cpu else "cpu"
    )
    dtype = torch.float16 if (config.fp16 and device.type == "cuda") else torch.float32

    logging.info(f"Loading reranker: {config.reranker_model_name} (device={device}, dtype={dtype})")
    rer_tokenizer, rer_model = load_cross_encoder(
        config.reranker_model_name, device, dtype=dtype
    )
    rer_tokenizer.model_max_length = max(config.window_sizes)

    logging.info(f"Loading span extractor: {config.span_model_name}")
    span_tokenizer, span_model = load_span_extractor(
        config.span_model_name, device, dtype=dtype
    )
    span_tokenizer.model_max_length = max(
        span_tokenizer.model_max_length or 0, config.span_max_seq_length
    )

    for win in config.window_sizes:
        for ovr in config.overlap_ratios:
            logging.info(f"=== Setting: window={win} overlap_ratio={ovr:.2f} ===")
            run_one_setting(
                config,
                rer_tokenizer, rer_model,
                span_tokenizer, span_model,
                device, win, ovr,
            )


if __name__ == "__main__":
    main()
