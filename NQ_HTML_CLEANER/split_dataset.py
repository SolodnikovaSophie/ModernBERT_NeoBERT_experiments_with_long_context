from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List


def split_train_val_by_files(
    cleaned_files: List[Path],
    output_root: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict:
    """
    Разбивает очищенные файлы на train и val на уровне файлов.

    Файлы копируются в отдельные директории:
        - split_by_files/train
        - split_by_files/val

    При этом полный набор очищенных файлов сохраняется в output/.

    Аргументы:
        cleaned_files (List[Path]): список очищенных файлов.
        output_root (Path): корневая директория текущего запуска.
        val_ratio (float): доля файлов, выделяемых в validation.
        seed (int): seed для воспроизводимого перемешивания.

    Возвращает:
        Dict: сводная информация о выполненном разбиении.

    Исключения:
        ValueError: если список cleaned_files пуст.
    """
    if not cleaned_files:
        raise ValueError("No cleaned files provided for split")

    files = sorted(cleaned_files)
    rnd = random.Random(seed)
    rnd.shuffle(files)

    n_total = len(files)
    n_val = max(1, int(round(n_total * val_ratio))) if n_total > 1 else 0
    n_val = min(n_val, n_total)

    val_files = files[:n_val]
    train_files = files[n_val:]

    split_root = output_root / "split_by_files"
    train_dir = split_root / "train"
    val_dir = split_root / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for src in train_files:
        shutil.copy2(src, train_dir / src.name)

    for src in val_files:
        shutil.copy2(src, val_dir / src.name)

    summary = {
        "split_type": "by_files",
        "total_files": n_total,
        "train_files_count": len(train_files),
        "val_files_count": len(val_files),
        "val_ratio_requested": val_ratio,
        "val_ratio_actual": (len(val_files) / n_total) if n_total else 0.0,
        "seed": seed,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
    }

    return summary