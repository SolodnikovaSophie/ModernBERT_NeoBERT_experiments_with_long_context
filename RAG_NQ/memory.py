"""CPU/GPU memory tracking with peak aggregation and nvidia-smi power readings."""
import psutil
import torch
import gc
import subprocess
import json
import logging
import time
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
            "gpu_ram_used_mb": self.gpu_ram_used or 0,
            "gpu_ram_peak_mb": self.gpu_ram_peak or 0,
            "reserved_gpu_ram_mb": self.reserved_gpu_ram or 0,
            "gpu_power_draw_w": self.gpu_power_draw or 0,
            "gpu_power_limit_w": self.gpu_power_limit or 0,
        }


class MemoryTracker:
    """Per-(model, window, position) memory tracker.

    Reset between runs by instantiating a new tracker. Power readings come from
    nvidia-smi and reflect the entire GPU (multi-process noise possible).
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        self.memory_log = []
        self.peak_stats: Optional[MemoryStats] = None

    def get_current_memory_usage(self) -> MemoryStats:
        proc = psutil.Process()
        cpu_ram = proc.memory_info().rss / (1024 * 1024)

        gpu_ram_used = gpu_ram_peak = reserved_gpu_ram = None
        gpu_power_draw = gpu_power_limit = None

        if torch.cuda.is_available():
            gpu_ram_used = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_ram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            reserved_gpu_ram = torch.cuda.memory_reserved() / (1024 * 1024)

            try:
                gid = torch.cuda.current_device()
                res = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,power.draw,power.limit",
                     "--format=csv,noheader,nounits",
                     f"--id={gid}"],
                    capture_output=True, text=True, timeout=2, check=True,
                )
                for line in res.stdout.strip().split("\n"):
                    parts = line.split(",")
                    if len(parts) < 3:
                        continue
                    if int(parts[0].strip()) != gid:
                        continue
                    try:
                        gpu_power_draw = (float(parts[1].strip())
                                          if parts[1].strip() != "[N/A]" else None)
                        gpu_power_limit = (float(parts[2].strip())
                                           if parts[2].strip() != "[N/A]" else None)
                    except (ValueError, IndexError):
                        pass
                    break
            except (subprocess.CalledProcessError, FileNotFoundError,
                    subprocess.TimeoutExpired, ValueError) as e:
                self.logger.debug(f"nvidia-smi unavailable: {e}")

        stats = MemoryStats(cpu_ram, gpu_ram_used, gpu_ram_peak,
                            reserved_gpu_ram, gpu_power_draw, gpu_power_limit)

        if self.peak_stats is None:
            self.peak_stats = stats
        else:
            self.peak_stats = MemoryStats(
                cpu_ram_used=max(self.peak_stats.cpu_ram_used, stats.cpu_ram_used),
                gpu_ram_used=max(self.peak_stats.gpu_ram_used or 0,
                                 stats.gpu_ram_used or 0),
                gpu_ram_peak=max(self.peak_stats.gpu_ram_peak or 0,
                                 stats.gpu_ram_peak or 0),
                reserved_gpu_ram=max(self.peak_stats.reserved_gpu_ram or 0,
                                     stats.reserved_gpu_ram or 0),
                gpu_power_draw=(max(self.peak_stats.gpu_power_draw or 0,
                                    stats.gpu_power_draw or 0)
                                if stats.gpu_power_draw is not None
                                else self.peak_stats.gpu_power_draw),
                gpu_power_limit=(stats.gpu_power_limit
                                 if stats.gpu_power_limit is not None
                                 else self.peak_stats.gpu_power_limit),
            )
        return stats

    def log_memory(self, component: str, operation: str) -> None:
        s = self.get_current_memory_usage()
        self.memory_log.append({
            "component": component,
            "operation": operation,
            "timestamp": time.time(),
            **s.to_dict(),
        })

    def clear_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_log(self) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "memory_usage.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "detailed_log": self.memory_log,
                "peak_usage": self.peak_stats.to_dict() if self.peak_stats else None,
            }, f, indent=2)
        return path
