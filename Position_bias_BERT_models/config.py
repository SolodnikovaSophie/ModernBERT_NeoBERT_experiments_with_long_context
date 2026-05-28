import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class ExperimentConfig:
    model_name: str
    input_path: str
    run_name: str
    window_sizes: list[int]
    batch_size: int
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42
    cpu: bool = False

    @property
    def output_dir(self) -> Path:
        return Path("logs_output") / self.run_name

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4, ensure_ascii=False)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Train Length x Test Length Context QA Experiment")
    parser.add_argument("--input-path", required=True, help="Path to NQ .jsonl.gz files or directory")
    parser.add_argument("--model-name", required=True, help="HF model or local checkpoint path")
    parser.add_argument("--run-name", type=str, default=None, help="Name of the run folder in logs_output")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--batch-size", type=int, default=16, help="Base batch size for window_size <= 1024")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()

    if not args.run_name:
        model_basename = Path(args.model_name).name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.run_name = f"run_{model_basename}_{timestamp}"

    return ExperimentConfig(
        model_name=args.model_name,
        input_path=args.input_path,
        run_name=args.run_name,
        window_sizes=args.window_sizes,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        cpu=args.cpu
    )