"""CLI parsing and registry of decoder models for the oracle QA experiment."""
import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import List, Optional


# HuggingFace ids and per-model inference defaults.
# Add new models here; the --models flag is restricted to these keys.
MODEL_REGISTRY = {
    "gemma3_1b": {
        "model_path": "google/gemma-3-1b-it",
        "dtype": "bfloat16",
        "use_chat_template": True,
        "quantize_default": False,
    },
    "deepseek_r1_qwen_1.5b": {
        "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "dtype": "bfloat16",
        "use_chat_template": True,
        "quantize_default": False,
    },
    "qwen3_0.6b": {
        "model_path": "Qwen/Qwen3-0.6B",
        "dtype": "bfloat16",
        "use_chat_template": True,
        "quantize_default": False,
        "qwen3_no_think": True,
    },
    "smollm2_360m": {
        "model_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "dtype": "bfloat16",
        "use_chat_template": True,
        "quantize_default": False,
    },
    "phi4_mini": {
        "model_path": "microsoft/Phi-4-mini-instruct",
        "dtype": "bfloat16",
        "use_chat_template": True,
        # 3.8B params won't fit in 9 GB VRAM with full context; default to 4-bit.
        "quantize_default": True,
    },
}

# Position-bias zones (Lost-in-the-Middle): the gold span is placed
# UNIFORMLY at random within each zone (0–30 % / 30–60 % / 60–100 %).
# The draw is deterministic per (qid, sa_start, sa_end, zone, seed), so a
# rerun reproduces the exact same context.
POSITION_RANGES = {
    "start":  (0.00, 0.30),
    "middle": (0.30, 0.60),
    "end":    (0.60, 1.00),
}

ALL_METRICS = ["em", "f1", "recall", "bertscore", "bleurt"]


@dataclass
class ExperimentConfig:
    input_path: str
    run_name: str
    models: List[str]
    window_sizes: List[int]
    positions: List[str]
    metrics: List[str]
    max_new_tokens: int
    temperature: float
    batch_size: int
    bootstrap_samples: int
    confidence_level: float
    seed: int
    cpu: bool
    cpu_fallback: bool
    quantize_models: List[str]
    max_input_tokens: Optional[int]
    limit_questions: Optional[int]
    bertscore_model: str
    bleurt_model: str

    @property
    def output_dir(self) -> Path:
        return Path("logs_output") / self.run_name

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Decoder oracle QA experiment (NQ, Lost-in-the-Middle)")
    p.add_argument("--input-path", required=True,
                   help="Path to NQ *.jsonl.gz directory or single file")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--models", nargs="+", required=True,
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--window-sizes", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096, 8192, -1],
                   help="Input length budgets in decoder tokens; -1 = full doc")
    p.add_argument("--positions", nargs="+",
                   default=["start", "middle", "end"],
                   choices=list(POSITION_RANGES.keys()))
    p.add_argument("--metrics", nargs="+",
                   default=ALL_METRICS,
                   choices=ALL_METRICS)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 → greedy decoding (recommended for oracle extraction)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU for everything (debug)")
    p.add_argument("--cpu-fallback", action="store_true",
                   help="On GPU OOM, retry the example on CPU (slow; not for 4-bit models)")
    p.add_argument("--quantize", nargs="*", default=None,
                   help="Models to load in 4-bit; default: registry quantize_default")
    p.add_argument("--max-input-tokens", type=int, default=None,
                   help="Hard cap on input tokens (default: model.max_position_embeddings)")
    p.add_argument("--limit-questions", type=int, default=None,
                   help="Process only N first distinct questions (for debug)")
    p.add_argument("--bertscore-model", type=str, default="roberta-large")
    p.add_argument("--bleurt-model", type=str, default="lucadiliello/BLEURT-20-D12")

    args = p.parse_args()

    if not args.run_name:
        args.run_name = f"decoder_oracle_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if args.quantize is None:
        quantize_models = [k for k in args.models
                           if MODEL_REGISTRY[k].get("quantize_default", False)]
    else:
        quantize_models = list(args.quantize)

    return ExperimentConfig(
        input_path=args.input_path,
        run_name=args.run_name,
        models=args.models,
        window_sizes=args.window_sizes,
        positions=args.positions,
        metrics=args.metrics,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        cpu=args.cpu,
        cpu_fallback=args.cpu_fallback,
        quantize_models=quantize_models,
        max_input_tokens=args.max_input_tokens,
        limit_questions=args.limit_questions,
        bertscore_model=args.bertscore_model,
        bleurt_model=args.bleurt_model,
    )
