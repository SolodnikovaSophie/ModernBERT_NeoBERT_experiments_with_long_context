# NQ HTML Cleaner

## Описание

Данный проект предназначен для предобработки датасета **Natural Questions (NQ)**.

Основная задача — удалить из поля `document_tokens` все HTML-токены и корректно пересчитать индексы ответов после такой очистки. Это необходимо, потому что в оригинальном формате NQ документ содержит не только текстовые токены, но и токены HTML-разметки, из-за чего индексы `start_token` и `end_token` относятся к смешанной последовательности из текста и HTML.

После удаления HTML-токенов исходные индексы ответов перестают быть корректными. Поэтому проект выполняет:

- удаление HTML-токенов из `document_tokens`;
- пересчёт индексов `long_answer`;
- пересчёт индексов `short_answers`;
- удаление полей, которые больше не соответствуют новой токенизации;
- сохранение очищённых файлов в новый каталог;
- сбор статистики обработки;
- при необходимости разбиение train-файлов на `train/val` на уровне файлов.

Проект ориентирован на подготовку данных для последующего обучения моделей extractive question answering на датасете Natural Questions.

---

## Пайплайн

Для каждого примера датасета выполняются следующие шаги:

1. Проверяется, что пример:
   - не содержит ответов типа `YES/NO` для упрощения обучения;
   - содержит корректный `long_answer`;
   - содержит хотя бы один корректный `short_answer`.

2. Удаляются неиспользуемые поля:
   - `document_html`
   - `long_answer_candidates`

3. Из `document_tokens` удаляются все HTML-токены.

4. Строится отображение индексов:
   - старый индекс токена → новый индекс токена.

5. Пересчитываются границы:
   - `annotations[].long_answer.start_token / end_token`
   - `annotations[].short_answers[].start_token / end_token`

6. Удаляются байтовые поля:
   - `start_byte`
   - `end_byte`
   - `text`

7. Сохраняется очищенная версия примера.

8. Собирается статистика:
   - количество обработанных файлов;
   - количество обработанных примеров;
   - количество отброшенных примеров;
   - количество токенов до и после очистки;
   - количество удалённых HTML-токенов;
   - количество пересчитанных `long_answer` и `short_answer`;
   - количество ошибок валидации span.

---

## Оригинальная структура Natural Questions

Ниже приведён пример структуры одного объекта в исходном формате NQ.

```json
{
  "annotations": [
    {
      "annotation_id": 6782080525527814293,
      "long_answer": {
        "candidate_index": 92,
        "end_byte": 96948,
        "end_token": 3538,
        "start_byte": 82798,
        "start_token": 2114
      },
      "short_answers": [
        {
          "end_byte": 96731,
          "end_token": 3525,
          "start_byte": 96715,
          "start_token": 3521
        }
      ],
      "yes_no_answer": "NONE"
    }
  ],
  "document_html": "<!DOCTYPE html> ...",
  "document_tokens": [
    {"token": "<!DOCTYPE>", "start_byte": 0, "end_byte": 10, "html_token": true},
    {"token": "The", "start_byte": 100, "end_byte": 103, "html_token": false},
    {"token": "Walking", "start_byte": 104, "end_byte": 111, "html_token": false},
    {"token": "Dead", "start_byte": 112, "end_byte": 116, "html_token": false}
  ],
  "long_answer_candidates": [
    {
      "start_byte": 108036,
      "end_byte": 109347,
      "start_token": 4193,
      "end_token": 4251,
      "top_level": true
    }
  ],
  "question_text": "when is the last episode of season 8 of the walking dead",
  "question_tokens": [
    "when",
    "is",
    "the",
    "last",
    "episode",
    "of",
    "season",
    "8",
    "of",
    "the",
    "walking",
    "dead"
  ]
}
````

### Особенности исходной структуры

В оригинальном формате:

* `document_html` хранит полный HTML страницы;
* `document_tokens` содержит и текстовые токены, и HTML-токены;
* `long_answer_candidates` содержит список кандидатов длинного ответа;
* `annotations` содержит разметку ответов;
* границы `start_token` и `end_token` заданы по исходной последовательности токенов;
* поля `start_byte` и `end_byte` относятся к исходному HTML-документу.

---

## Новая структура после очистки

Пример:

```json
{
  "annotations": [
    {
      "annotation_id": 6782080525527814293,
      "long_answer": {
        "candidate_index": 92,
        "start_token": 1800,
        "end_token": 3001
      },
      "short_answers": [
        {
          "start_token": 2988,
          "end_token": 2992
        }
      ],
      "yes_no_answer": "NONE"
    }
  ],
  "document_tokens": [
    {"token": "The", "html_token": false},
    {"token": "Walking", "html_token": false},
    {"token": "Dead", "html_token": false},
    {"token": "season", "html_token": false},
    {"token": "8", "html_token": false}
  ],
  "question_text": "when is the last episode of season 8 of the walking dead",
  "question_tokens": [
    "when",
    "is",
    "the",
    "last",
    "episode",
    "of",
    "season",
    "8",
    "of",
    "the",
    "walking",
    "dead"
  ]
}
```

### Что изменяется после обработки

Из примера удаляются:

* `document_html`
* `long_answer_candidates`

Из span-структур удаляются:

* `start_byte`
* `end_byte`
* `text`

В `document_tokens` остаются только токены вида:

```json
{"token": "...", "html_token": false}
```

Индексы `start_token` и `end_token` в ответах пересчитываются уже относительно очищенного списка токенов.

---

## config file

Пример `config.json`:

```json
{
  "input_path": " ",
  "runs_root": " ",
  "run_name": " ",
  "input_glob": "*.jsonl.gz",
  "encoding": "utf-8",
  "validation": {
    "enabled": true
  },
  "logging": {
    "level": "INFO",
    "save_token_stats": true
  },
  "split_dataset": {
    "enabled": true,
    "mode": "train",
    "val_ratio": 0.15,
    "seed": 42
  }
}
```

### Основные параметры

* `input_path` — путь к входному файлу или директории с `.jsonl.gz`
* `runs_root` — корневая директория для сохранения запусков
* `run_name` — имя запуска
* `input_glob` — шаблон поиска файлов
* `encoding` — кодировка
* `validation.enabled` — включение проверки корректности span
* `logging.level` — уровень логирования
* `logging.save_token_stats` — сохранять ли `token_stats.jsonl`
* `split_dataset.enabled` — выполнять ли разбиение train/val
* `split_dataset.mode` — режим работы разбиения
* `split_dataset.val_ratio` — доля файлов для validation
* `split_dataset.seed` — seed для воспроизводимого разбиения

---

## Команды для запуска

### 1. Запуск обработки с конфигурацией

```bash
python cli.py --config config.json
```

### 2. Запуск с переопределением входного пути

```bash
python cli.py --config config.json --input-path "add_your/path/to/nq/train"
```

### 3. Запуск с переопределением имени запуска

```bash
python cli.py --config config.json --run-name nq_clean_dev_run
```

### 4. Запуск с переопределением и пути, и имени запуска

```bash
python cli.py --config config.json --input-path "add_your/path/to/nq/train" --run-name nq_clean_train_experiment
```

---

## Пример структуры результата

После выполнения можно получить, например, такую структуру:

```text
runs/
└── nq_clean_answers_V2_train_2026-04-17_14-35-10/
    ├── output/
    │   ├── nq-train-00.jsonl.gz
    │   ├── nq-train-01.jsonl.gz
    │   └── ...
    ├── logs/
    │   ├── run.log
    │   ├── token_stats.jsonl
    │   └── summary.json
    └── split_by_files/
        ├── train/
        │   ├── nq-train-00.jsonl.gz
        │   └── ...
        └── val/
            ├── nq-train-14.jsonl.gz
            └── ...
```

Вот аккуратный блок для README, который можно вставить в раздел **“Как скачать датасет Natural Questions”**:

---

## Как скачать Natural Questions (оригинальный формат)

Официальный датасет Natural Questions можно скачать с Google Cloud Storage:

🔗 [https://ai.google.com/research/NaturalQuestions/download](https://ai.google.com/research/NaturalQuestions/download)

### Шаг 1. Установить `gsutil`

Можно установить как часть Google Cloud SDK.
---

### Шаг 2. Скачать датасет

Выполни команду:

```bash
gsutil -m cp -R gs://natural_questions/v1.0 <path_to_your_data_directory>
```

---

### Структура датасета NQ

* **train** — ~41 GB (основной обучающий набор)
* **dev** — ~1 GB (валидационный набор)
  
Структура:

```text
v1.0/
├── train/
│   ├── nq-train-00.jsonl.gz
│   ├── nq-train-01.jsonl.gz
│   └── ...
├── dev/
│   ├── nq-dev-00.jsonl.gz
│   └── ...
```
---


