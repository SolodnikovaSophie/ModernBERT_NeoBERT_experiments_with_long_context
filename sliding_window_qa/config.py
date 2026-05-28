"""
Конфигурация эксперимента Cross-Encoder Reranking на NQ dev.
"""

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from typing import List


@dataclass
class RerankerConfig:
    # Stage 1 — cross-encoder реранкер (HF id или путь к локальной папке с весами)
    reranker_model_name: str
    # Stage 2 — span-extractor QA-модель (HF id или путь к локальной папке с весами)
    span_model_name: str
    # NQ dev (файл *.jsonl.gz или директория с такими файлами)
    input_path: str
    run_name: str

    window_sizes: List[int] = field(default_factory=lambda: [512, 2048])
    overlap_ratios: List[float] = field(default_factory=lambda: [0.10])
    batch_size: int = 16              # для реранкера (пары Q-chunk)
    span_batch_size: int = 8          # для span-extractor
    span_top_k: int = 3               # сколько топ-чанков прогонять через QA-модель
    span_max_seq_length: int = 512    # длина окна для QA-модели
    max_span_words: int = 30          # верхняя граница длины извлечённого span (в словах)

    max_examples: int = 0             # 0 — обработать всё
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42
    cpu: bool = False
    fp16: bool = False
    ks: List[int] = field(default_factory=lambda: [1, 3, 5])
    output_root: str = "logs_output_reranker"

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


def parse_args() -> RerankerConfig:
    p = argparse.ArgumentParser(description="Cross-Encoder Reranking on NQ dev")
    p.add_argument("--input-path", required=True,
                   help="NQ dev jsonl.gz file or directory of such files")
    p.add_argument("--reranker-model-name", required=True,
                   help="Stage-1 cross-encoder: HF id or local folder with pretrained weights "
                        "(e.g. ModernBERT reranker)")
    p.add_argument("--span-model-name", required=True,
                   help="Stage-2 span extractor: HF id or local folder with pretrained "
                        "AutoModelForQuestionAnswering weights")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--window-sizes", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--overlap-ratios", nargs="+", type=float, default=[0.10])
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for reranker pairs (Q, chunk)")
    p.add_argument("--span-batch-size", type=int, default=8,
                   help="Batch size for span extractor")
    p.add_argument("--span-top-k", type=int, default=3,
                   help="How many top-ranked chunks to feed to the span extractor")
    p.add_argument("--span-max-seq-length", type=int, default=512,
                   help="Max sequence length for span extractor")
    p.add_argument("--max-span-words", type=int, default=30,
                   help="Cap on extracted span length in words")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--output-root", type=str, default="logs_output_reranker")
    a = p.parse_args()

    if not a.run_name:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rer = Path(a.reranker_model_name).name
        spn = Path(a.span_model_name).name
        a.run_name = f"twostage_{rer}__{spn}_{ts}"

    return RerankerConfig(
        reranker_model_name=a.reranker_model_name,
        span_model_name=a.span_model_name,
        input_path=a.input_path,
        run_name=a.run_name,
        window_sizes=a.window_sizes,
        overlap_ratios=a.overlap_ratios,
        batch_size=a.batch_size,
        span_batch_size=a.span_batch_size,
        span_top_k=a.span_top_k,
        span_max_seq_length=a.span_max_seq_length,
        max_span_words=a.max_span_words,
        max_examples=a.max_examples,
        bootstrap_samples=a.bootstrap_samples,
        confidence_level=a.confidence_level,
        seed=a.seed,
        cpu=a.cpu,
        fp16=a.fp16,
        ks=a.ks,
        output_root=a.output_root,
    )
