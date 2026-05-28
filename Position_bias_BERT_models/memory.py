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
            "gpu_power_limit_w": self.gpu_power_limit or 0
        }

class MemoryTracker:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        self.memory_log = []
        self.peak_stats = None

    def get_current_memory_usage(self) -> MemoryStats:
        process = psutil.Process()
        cpu_ram = process.memory_info().rss / (1024 * 1024)

        gpu_ram_used = gpu_ram_peak = reserved_gpu_ram = gpu_power_draw = gpu_power_limit = None

        if torch.cuda.is_available():
            gpu_ram_used = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_ram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            reserved_gpu_ram = torch.cuda.memory_reserved() / (1024 * 1024)

            try:
                gpu_device_id = torch.cuda.current_device()
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,power.draw,power.limit', '--format=csv,noheader,nounits',
                     f'--id={gpu_device_id}'],
                    capture_output=True, text=True, timeout=2, check=True
                )
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3 and int(parts[0].strip()) == gpu_device_id:
                    gpu_power_draw = float(parts[1].strip()) if parts[1].strip() != "[N/A]" else None
                    gpu_power_limit = float(parts[2].strip()) if parts[2].strip() != "[N/A]" else None
            except Exception:
                pass

        stats = MemoryStats(cpu_ram, gpu_ram_used, gpu_ram_peak, reserved_gpu_ram, gpu_power_draw, gpu_power_limit)

        if self.peak_stats is None:
            self.peak_stats = stats
        else:
            self.peak_stats.cpu_ram_used = max(self.peak_stats.cpu_ram_used, stats.cpu_ram_used)
            self.peak_stats.gpu_ram_peak = max(self.peak_stats.gpu_ram_peak or 0, stats.gpu_ram_peak or 0)
            self.peak_stats.gpu_power_draw = max(self.peak_stats.gpu_power_draw or 0, stats.gpu_power_draw or 0)

        return stats

    def log_memory(self, component: str, operation: str):
        stats = self.get_current_memory_usage()
        log_entry = {"component": component, "operation": operation, "timestamp": time.time(), **stats.to_dict()}
        self.memory_log.append(log_entry)

    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_log(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / "memory_usage.json"
        with open(log_file, "w") as f:
            json.dump({
                "detailed_log": self.memory_log,
                "peak_usage": self.peak_stats.to_dict() if self.peak_stats else None
            }, f, indent=2)