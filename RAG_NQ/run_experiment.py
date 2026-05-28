"""Main loop: models × window_sizes × positions on NQ in oracle mode.

Two phases:
  Phase 1 — generation + basic metrics (EM/F1/Recall) per (model, window, position)
  Phase 2 — semantic metrics (BERTScore, BLEURT) once all decoders are freed

Output layout:
  logs_output/<run_name>/
    config.json
    run.log
    <model>/win_<size|full>/pos_<start|middle|end>/
      predictions.json
      metrics.json
      memory_usage.json
"""
import collections
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


class _TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so they don't break the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)

from config import parse_args, MODEL_REGISTRY, POSITION_RANGES, ExperimentConfig
from data_utils import (iterate_oracle_examples, build_context_at_position,
                        sample_position_ratio)
from decoder_inference import DecoderInferenceModel
from memory import MemoryTracker
from metrics import (aggregate_basic_metrics,
                     add_semantic_metrics_to_dir,
                     BERTScorer, BLEURTScorer)
from prompts import SYSTEM_PROMPT, USER_TEMPLATE


# ============================================================
# Logging
# ============================================================

def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = _TqdmLoggingHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    # Tame noisy 3rd-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def reset_peak_gpu():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# Phase 1 — generation + basic metrics
# ============================================================

def _window_dirname(window_size: int) -> str:
    return "full" if window_size == -1 else str(window_size)


def run_one_combo(
    model: DecoderInferenceModel,
    config: ExperimentConfig,
    model_key: str,
    window_size: int,
    position: str,
    output_dir: Path,
) -> Path:
    zone_lo, zone_hi = POSITION_RANGES[position]
    win_dir = output_dir / model_key / f"win_{_window_dirname(window_size)}" / f"pos_{position}"
    win_dir.mkdir(parents=True, exist_ok=True)

    mem = MemoryTracker(win_dir)
    reset_peak_gpu()
    mem.log_memory("system", "experiment_start")

    # Determine the input cap for this model (cap by user's max_input_tokens too).
    cap = (config.max_input_tokens
           if config.max_input_tokens is not None
           else model.max_position_embeddings - config.max_new_tokens - 8)
    cap = max(64, cap)

    predictions_log: List[Dict] = []
    predictions_by_q: Dict[str, List[str]] = collections.defaultdict(list)
    golds_by_q: Dict[str, List[str]] = collections.defaultdict(list)

    started_at = time.time()
    n_processed = 0
    n_oom = 0

    iterator = iterate_oracle_examples(
        config.input_path, limit_questions=config.limit_questions)
    pbar = tqdm(iterator,
                desc=f"{model_key} | win={_window_dirname(window_size)} | pos={position}",
                unit="ex", smoothing=0.05)

    # Bin label for analytics: e.g. "0-30", "30-60", "60-100".
    zone_label = f"{int(round(zone_lo*100))}-{int(round(zone_hi*100))}"

    for (qid, q_txt, doc_toks, _la_s, _la_e, sa_s, sa_e, gold) in pbar:
        # Pick the in-zone target ratio deterministically per record.
        pos_ratio = sample_position_ratio(
            zone_lo, zone_hi, qid, sa_s, sa_e, position, config.seed,
        )

        ctx_info = build_context_at_position(
            tokenizer=model.tokenizer,
            system_prompt=SYSTEM_PROMPT,
            user_template=USER_TEMPLATE,
            question=q_txt,
            doc_tokens=doc_toks,
            sa_start=sa_s,
            sa_end=sa_e,
            target_window_size=window_size,
            target_pos_ratio=pos_ratio,
            max_input_tokens=cap,
            use_chat_template=model.use_chat_template,
        )
        ctx_text = " ".join(ctx_info["context_tokens"])
        prompt_text = model.build_prompt(SYSTEM_PROMPT, USER_TEMPLATE, ctx_text, q_txt)

        # Where the gold span actually ended up after edge-clipping.
        ctx_word_len = ctx_info["context_end_token"] - ctx_info["context_start_token"]
        gold_local_center = (sa_s - ctx_info["context_start_token"]
                             + (sa_e - sa_s) / 2.0)
        actual_ratio = (gold_local_center / ctx_word_len) if ctx_word_len > 0 else 0.0

        mem.log_memory("model", "before_generate")
        record = {
            "question_id": qid,
            "question": q_txt,
            "gold_answer": gold,
            "prediction": "",
            "context_length_tokens": -1,
            "position_zone": position,                    # "start" | "middle" | "end"
            "position_zone_label": zone_label,            # "0-30" | "30-60" | "60-100"
            "position_target_ratio": round(pos_ratio, 4),
            "position_actual_ratio": round(actual_ratio, 4),
            "window_size": window_size,
        }
        try:
            pred, input_len = model.generate(prompt_text, max_input_tokens=cap)
            record["prediction"] = pred
            record["context_length_tokens"] = input_len
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            torch.cuda.empty_cache()
            gc.collect()
            record["oom"] = True
            if config.cpu_fallback and not model.is_quantized:
                logging.warning(f"OOM on qid={qid} (len={len(doc_toks)}); CPU fallback")
                try:
                    # Move to CPU temporarily, generate, move back.
                    model.model.to("cpu")
                    pred, input_len = model.generate(prompt_text, max_input_tokens=cap)
                    record["prediction"] = pred
                    record["context_length_tokens"] = input_len
                    record["cpu_fallback"] = True
                except Exception as e:
                    logging.exception(f"CPU fallback failed for {qid}: {e}")
                finally:
                    try:
                        model.model.to(model.device)
                    except Exception as e:
                        logging.exception(f"failed to move model back to {model.device}: {e}")
        except Exception as e:
            logging.exception(f"generation error qid={qid}: {e}")
            record["error"] = str(e)
        mem.log_memory("model", "after_generate")

        predictions_log.append(record)
        predictions_by_q[qid].append(record["prediction"])
        golds_by_q[qid].append(gold)

        n_processed += 1
        if n_processed % 50 == 0:
            mem.clear_memory()
        pbar.set_postfix({"n": n_processed, "oom": n_oom})

    pbar.close()
    duration = time.time() - started_at

    pred_path = win_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions_log, f, ensure_ascii=False, indent=2)

    per_q = {qid: {"predictions": predictions_by_q[qid],
                   "gold_answers": golds_by_q[qid]}
             for qid in golds_by_q}

    basic_metrics, _, _, _ = aggregate_basic_metrics(
        per_q,
        selected=[m for m in config.metrics if m in ("em", "f1", "recall")],
        bootstrap_samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
        seed=config.seed,
    )

    metrics: Dict = {
        "window_size": window_size,
        "position": position,
        "position_zone_label": zone_label,
        "model": model_key,
        **basic_metrics,
        "duration_seconds": duration,
        "n_processed": n_processed,
        "n_oom": n_oom,
        "model_size_bytes": model.get_model_size_bytes(),
    }
    metrics["model_size_mb"] = metrics["model_size_bytes"] / (1024 * 1024)

    mem.log_memory("system", "experiment_end")
    if mem.peak_stats is not None:
        metrics.update(mem.peak_stats.to_dict())
    mem.save_log()

    metrics_path = win_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    logging.info(
        f"[DONE] {model_key} | win={_window_dirname(window_size)} | pos={position} "
        f"| EM={metrics.get('EM', 0):.2f} | F1={metrics.get('F1', 0):.2f} "
        f"| Recall={metrics.get('Recall', 0):.2f} | n={n_processed} | OOM={n_oom} "
        f"| dur={duration:.1f}s"
    )
    return metrics_path


# ============================================================
# Phase 2 — semantic metrics
# ============================================================

def run_semantic_phase(config: ExperimentConfig):
    want_bs = "bertscore" in config.metrics
    want_bl = "bleurt" in config.metrics
    if not (want_bs or want_bl):
        return

    logging.info(f"\n{'='*60}\nPhase 2: semantic metrics\n{'='*60}")
    device = "cuda" if (not config.cpu and torch.cuda.is_available()) else "cpu"

    bs_scorer: Optional[BERTScorer] = None
    bl_scorer: Optional[BLEURTScorer] = None
    if want_bs:
        try:
            logging.info(f"loading BERTScorer ({config.bertscore_model}, device={device})")
            bs_scorer = BERTScorer(model_type=config.bertscore_model, device=device)
        except Exception as e:
            logging.exception(f"failed to load BERTScorer: {e}")
            bs_scorer = None
    if want_bl:
        try:
            logging.info(f"loading BLEURTScorer ({config.bleurt_model}, device={device})")
            bl_scorer = BLEURTScorer(model_path=config.bleurt_model, device=device)
        except Exception as e:
            logging.exception(f"failed to load BLEURTScorer: {e}")
            bl_scorer = None

    try:
        for model_key in config.models:
            for ws in config.window_sizes:
                for pos in config.positions:
                    d = (config.output_dir / model_key
                         / f"win_{_window_dirname(ws)}" / f"pos_{pos}")
                    metrics_path = d / "metrics.json"
                    pred_path = d / "predictions.json"
                    if not metrics_path.exists() or not pred_path.exists():
                        continue
                    logging.info(f"semantic: {d}")
                    add_semantic_metrics_to_dir(
                        metrics_path=metrics_path,
                        pred_path=pred_path,
                        bertscore=bs_scorer,
                        bleurt=bl_scorer,
                        bootstrap_samples=config.bootstrap_samples,
                        confidence_level=config.confidence_level,
                        seed=config.seed,
                    )
    finally:
        if bs_scorer is not None:
            bs_scorer.free()
        if bl_scorer is not None:
            bl_scorer.free()


# ============================================================
# Main
# ============================================================

def main():
    config = parse_args()
    config.save()
    setup_logger(config.output_dir / "run.log")

    logging.info(f"Run name      : {config.run_name}")
    logging.info(f"Output dir    : {config.output_dir}")
    logging.info(f"Models        : {config.models}")
    logging.info(f"Quantize 4-bit: {config.quantize_models}")
    logging.info(f"Window sizes  : {config.window_sizes}")
    logging.info(f"Positions     : {config.positions}")
    logging.info(f"Metrics       : {config.metrics}")
    logging.info(f"max_new_tokens: {config.max_new_tokens}, "
                 f"temperature={config.temperature}")

    # ---- Phase 1 ----
    for model_key in config.models:
        reg = MODEL_REGISTRY[model_key]
        quantize = (model_key in config.quantize_models)
        device = "cpu" if config.cpu else "cuda"

        logging.info(f"\n{'='*60}\nLoading model: {model_key} "
                     f"({reg['model_path']}) | quantize={quantize}\n{'='*60}")
        try:
            model = DecoderInferenceModel(
                model_path=reg["model_path"],
                dtype=reg.get("dtype", "bfloat16"),
                quantize=quantize,
                device=device,
                use_chat_template=reg.get("use_chat_template", True),
                qwen3_no_think=reg.get("qwen3_no_think", False),
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
        except Exception as e:
            logging.exception(f"failed to load {model_key}: {e}")
            continue

        try:
            for window_size in config.window_sizes:
                for position in config.positions:
                    try:
                        run_one_combo(model, config, model_key,
                                      window_size, position, config.output_dir)
                    except Exception as e:
                        logging.exception(
                            f"run failed for {model_key}/{window_size}/{position}: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
        finally:
            model.free()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Phase 2 ----
    run_semantic_phase(config)
    logging.info("All done.")


if __name__ == "__main__":
    main()
