"""
Memory tracker (CPU/GPU RAM + GPU power) с логированием в JSON.
Заимствовано из slm_experiments/src/utils/memory_tracker.py (без ClearML).
"""

import psutil
import torch
import gc
import subprocess
import time
import json
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryStats:
    cpu_ram_used: float
    gpu_ram_used: Optional[float]
    gpu_ram_peak: Optional[float]
    reserved_gpu_ram: Optional[float]
    gpu_power_draw: Optional[float]
    gpu_power_limit: Optional[float]

    def to_dict(self) -> Dict[str, float]:
        return {
            "cpu_ram_used_mb": self.cpu_ram_used,
            "gpu_ram_used_mb": self.gpu_ram_used or 0.0,
            "gpu_ram_peak_mb": self.gpu_ram_peak or 0.0,
            "reserved_gpu_ram_mb": self.reserved_gpu_ram or 0.0,
            "gpu_power_draw_w": self.gpu_power_draw or 0.0,
            "gpu_power_limit_w": self.gpu_power_limit or 0.0,
        }


class MemoryTracker:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.memory_log = []
        self.peak_stats: Optional[MemoryStats] = None
        self._t0 = time.time()

    def _query_nvidia_smi(self) -> (Optional[float], Optional[float]):
        try:
            gpu_id = torch.cuda.current_device()
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3 and int(parts[0]) == gpu_id:
                    draw = None if parts[1] == "[N/A]" else float(parts[1])
                    lim = None if parts[2] == "[N/A]" else float(parts[2])
                    return draw, lim
        except Exception as e:
            self.logger.debug(f"nvidia-smi unavailable: {e}")
        return None, None

    def get_current_memory_usage(self) -> MemoryStats:
        process = psutil.Process()
        cpu_ram = process.memory_info().rss / (1024 * 1024)

        gpu_used = gpu_peak = gpu_reserved = None
        gpu_pw_draw = gpu_pw_lim = None

        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_pw_draw, gpu_pw_lim = self._query_nvidia_smi()

        stats = MemoryStats(
            cpu_ram_used=cpu_ram,
            gpu_ram_used=gpu_used,
            gpu_ram_peak=gpu_peak,
            reserved_gpu_ram=gpu_reserved,
            gpu_power_draw=gpu_pw_draw,
            gpu_power_limit=gpu_pw_lim,
        )

        if self.peak_stats is None:
            self.peak_stats = stats
        else:
            self.peak_stats = MemoryStats(
                cpu_ram_used=max(self.peak_stats.cpu_ram_used, stats.cpu_ram_used),
                gpu_ram_used=max(self.peak_stats.gpu_ram_used or 0, stats.gpu_ram_used or 0),
                gpu_ram_peak=max(self.peak_stats.gpu_ram_peak or 0, stats.gpu_ram_peak or 0),
                reserved_gpu_ram=max(self.peak_stats.reserved_gpu_ram or 0, stats.reserved_gpu_ram or 0),
                gpu_power_draw=max(self.peak_stats.gpu_power_draw or 0, stats.gpu_power_draw or 0)
                if stats.gpu_power_draw is not None
                else self.peak_stats.gpu_power_draw,
                gpu_power_limit=stats.gpu_power_limit
                if stats.gpu_power_limit is not None
                else self.peak_stats.gpu_power_limit,
            )

        return stats

    def log_memory(self, component: str, operation: str):
        stats = self.get_current_memory_usage()
        entry = {
            "component": component,
            "operation": operation,
            "timestamp": time.time(),
            "elapsed_s": time.time() - self._t0,
            **stats.to_dict(),
        }
        self.memory_log.append(entry)
        return stats

    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_log(self):
        log_file = self.log_dir / "memory_usage.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "detailed_log": self.memory_log,
                    "peak_usage": self.peak_stats.to_dict() if self.peak_stats else None,
                    "total_elapsed_s": time.time() - self._t0,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        self.logger.info(f"Memory log saved to {log_file}")
        if self.peak_stats:
            self.logger.info("Peak memory usage:")
            for k, v in self.peak_stats.to_dict().items():
                self.logger.info(f"  {k}: {v:.2f}")
