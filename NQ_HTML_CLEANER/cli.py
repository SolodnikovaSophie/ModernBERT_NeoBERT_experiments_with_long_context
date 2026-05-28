from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Iterable, List

from logging_utils import setup_main_logger, build_optional_jsonl_logger
from processor import process_records
from split_dataset import split_train_val_by_files


def parse_args():
    """
    Выполняет разбор аргументов командной строки.

    Возвращает:
        argparse.Namespace: объект с параметрами запуска.
    """
    parser = argparse.ArgumentParser(
        description="Clean NQ HTML tokens and remap answer spans."
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--input-path", default=None, help="Override input path")
    parser.add_argument("--run-name", default=None, help="Override run name")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """
    Загружает конфигурацию из JSON-файла и дополняет её значениями по умолчанию.

    Аргументы:
        config_path (str): путь к JSON-файлу конфигурации.

    Возвращает:
        Dict: словарь с параметрами конфигурации.

    Исключения:
        FileNotFoundError: если файл конфигурации не найден.
        ValueError: если отсутствуют обязательные ключи конфигурации.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["input_path", "runs_root", "run_name"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    config.setdefault("input_glob", "*.jsonl.gz")
    config.setdefault("encoding", "utf-8")

    validation_cfg = config.setdefault("validation", {})
    validation_cfg.setdefault("enabled", True)

    logging_cfg = config.setdefault("logging", {})
    logging_cfg.setdefault("level", "INFO")
    logging_cfg.setdefault("save_token_stats", True)

    split_cfg = config.setdefault("split_dataset", {})
    split_cfg.setdefault("enabled", False)
    split_cfg.setdefault("mode", "train")
    split_cfg.setdefault("val_ratio", 0.1)
    split_cfg.setdefault("seed", 42)

    return config


def find_input_files(input_path: str, pattern: str = "*.jsonl.gz") -> List[Path]:
    """
    Находит входные файлы для обработки.

    Если указан путь к одному файлу, возвращает список из одного элемента.
    Если указан путь к директории, выполняет поиск файлов по заданному шаблону.

    Аргументы:
        input_path (str): путь к файлу или директории.
        pattern (str): шаблон поиска файлов в директории.

    Возвращает:
        List[Path]: список найденных файлов.

    Исключения:
        FileNotFoundError: если путь не существует или подходящие файлы не найдены.
    """
    path = Path(input_path)

    if path.is_file():
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    files = sorted(p for p in path.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in: {path}"
        )

    return files


def read_jsonl_gz(file_path: Path, encoding: str = "utf-8") -> Generator[Dict, None, None]:
    """
    Читает gzip-сжатый JSONL-файл построчно.

    Аргументы:
        file_path (Path): путь к входному файлу.
        encoding (str): кодировка файла.

    Возвращает:
        Generator[Dict, None, None]: генератор JSON-объектов.

    Исключения:
        ValueError: если обнаружена некорректная JSON-строка.
    """
    with gzip.open(file_path, mode="rt", encoding=encoding) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {file_path}, line {line_num}: {e}") from e


def write_jsonl_gz(records: Iterable[Dict], file_path: Path, encoding: str = "utf-8") -> None:
    """
    Записывает последовательность записей в gzip-сжатый JSONL-файл.

    Аргументы:
        records (Iterable[Dict]): записи для сохранения.
        file_path (Path): путь к выходному файлу.
        encoding (str): кодировка файла.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(file_path, mode="wt", encoding=encoding) as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def init_stats() -> Dict[str, int]:
    """
    Инициализирует словарь статистики обработки.

    Возвращает:
        Dict[str, int]: словарь со счётчиками статистики.
    """
    return {
        "files_total": 0,
        "files_processed": 0,
        "examples_total": 0,
        "examples_processed": 0,
        "examples_failed": 0,
        "tokens_before_total": 0,
        "tokens_after_total": 0,
        "html_tokens_removed_total": 0,
        "long_answers_remapped": 0,
        "short_answers_remapped": 0,
        "annotation_long_answer_empty_after_html": 0,
        "annotation_short_answer_empty_after_html": 0,
        "validation_errors": 0,
    }


def finalize_stats(stats: Dict[str, int]) -> Dict:
    """
    Формирует итоговую статистику и вычисляет производные показатели.

    Аргументы:
        stats (Dict[str, int]): накопленная статистика обработки.

    Возвращает:
        Dict: итоговый словарь статистики.
    """
    result = dict(stats)

    if stats["tokens_before_total"] > 0:
        result["html_removed_ratio"] = (
            stats["html_tokens_removed_total"] / stats["tokens_before_total"]
        )
    else:
        result["html_removed_ratio"] = 0.0

    return result


def build_run_dirs(runs_root: str, run_name: str) -> Dict[str, Path]:
    """
    Создаёт директории текущего запуска.

    Аргументы:
        runs_root (str): корневая директория для запусков.
        run_name (str): имя текущего запуска.

    Возвращает:
        Dict[str, Path]: словарь с путями к директориям запуска.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(runs_root) / f"{run_name}_{timestamp}"
    output_dir = run_dir / "output"
    logs_dir = run_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "logs_dir": logs_dir,
    }


def main():
    """
    Запускает основной пайплайн очистки датасета.

    Функция выполняет:
        - загрузку конфигурации;
        - поиск входных файлов;
        - обработку и сохранение очищенных данных;
        - при необходимости разбиение train на train/val;
        - сохранение итоговой статистики.
    """
    args = parse_args()
    config = load_config(args.config)

    if args.input_path:
        config["input_path"] = args.input_path
    if args.run_name:
        config["run_name"] = args.run_name

    dirs = build_run_dirs(
        runs_root=config["runs_root"],
        run_name=config["run_name"],
    )

    logger = setup_main_logger(
        log_path=dirs["logs_dir"] / "run.log",
        level=config["logging"]["level"],
    )

    token_stats_logger = build_optional_jsonl_logger(
        config["logging"]["save_token_stats"],
        dirs["logs_dir"] / "token_stats.jsonl",
    )

    stats = init_stats()

    input_files = find_input_files(
        input_path=config["input_path"],
        pattern=config["input_glob"],
    )
    stats["files_total"] = len(input_files)

    logger.info("Start processing")
    logger.info(f"Input path: {config['input_path']}")
    logger.info(f"Run dir: {dirs['run_dir']}")
    logger.info(f"Output dir: {dirs['output_dir']}")
    logger.info(f"Logs dir: {dirs['logs_dir']}")
    logger.info(f"Files found: {len(input_files)}")

    cleaned_output_files = []

    try:
        for input_file in input_files:
            logger.info(f"Processing file: {input_file}")

            output_file = dirs["output_dir"] / input_file.name
            records = read_jsonl_gz(
                file_path=input_file,
                encoding=config["encoding"],
            )

            processed_records = process_records(
                records=records,
                stats=stats,
                token_stats_logger=token_stats_logger,
                validation_enabled=config["validation"]["enabled"],
                tqdm_desc=input_file.name,
            )

            write_jsonl_gz(
                records=processed_records,
                file_path=output_file,
                encoding=config["encoding"],
            )

            stats["files_processed"] += 1
            logger.info(f"Saved cleaned file: {output_file}")
            cleaned_output_files.append(output_file)

        split_summary = None

        if (
            config["split_dataset"]["enabled"]
            and config["split_dataset"]["mode"] == "train"
        ):
            logger.info("Splitting cleaned train into train/val...")

            split_summary = split_train_val_by_files(
                cleaned_files=cleaned_output_files,
                output_root=dirs["run_dir"],
                val_ratio=config["split_dataset"]["val_ratio"],
                seed=config["split_dataset"]["seed"],
            )

            logger.info("Split completed")
            logger.info(json.dumps(split_summary, ensure_ascii=False, indent=2))

        summary = finalize_stats(stats)

        if split_summary is not None:
            summary["split_dataset"] = split_summary

        summary_path = dirs["logs_dir"] / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Processing finished")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(json.dumps(summary, ensure_ascii=False, indent=2))

    finally:
        if token_stats_logger is not None:
            token_stats_logger.close()


if __name__ == "__main__":
    main()