from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional


class JsonlWriter:
    """
    Выполняет построчную запись JSON-объектов в файл формата JSONL.
    """

    def __init__(self, path: Path):
        """
        Инициализирует объект записи JSONL.

        Аргументы:
            path (Path): путь к выходному файлу.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, payload: dict) -> None:
        """
        Записывает один JSON-объект в файл.

        Аргументы:
            payload (dict): словарь для сохранения.
        """
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """
        Закрывает файловый дескриптор.
        """
        self._fh.close()


def setup_main_logger(log_path: Path, level: str = "INFO") -> logging.Logger:
    """
    Создаёт основной logger с выводом в консоль и в файл.

    Аргументы:
        log_path (Path): путь к log-файлу.
        level (str): уровень логирования.

    Возвращает:
        logging.Logger: настроенный объект logger.
    """
    logger = logging.getLogger("nq_html_cleaner")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def build_optional_jsonl_logger(enabled: bool, path: Path) -> Optional[JsonlWriter]:
    """
    Создаёт JSONL-логгер только в том случае, если логирование включено.

    Аргументы:
        enabled (bool): признак необходимости создания логгера.
        path (Path): путь к JSONL-файлу.

    Возвращает:
        Optional[JsonlWriter]: объект записи JSONL или None.
    """
    if not enabled:
        return None
    return JsonlWriter(path)