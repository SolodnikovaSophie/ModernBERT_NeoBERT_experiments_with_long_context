# Decoder Oracle QA Experiment 

Данный репозиторий представляет собой эксперимент со сравнением результатов **encoder-only** моделей с **decoder-only** моделями (LLM). Проект тестирует способность современных открытых моделей находить и извлекать ответ из длинного контекста в режиме **Oracle QA** на датасете Natural Questions.

Модели предоставляется вопрос и контекст, который содержит правильный ответ. Прогон осуществляется по сетке параметров: `Модель × Длина окна × Позиция ответа`.

## 🚀 Ключевые особенности
* **Продвинутые метрики:** Помимо классических EM и F1, вычисляются семантические метрики BERTScore и BLEURT с расчетом доверительных интервалов (Bootstrap CI 95%).
* **Адаптация под LLM:** Поддержка 4-bit квантования (для тяжелых моделей вроде Phi-4), парсинг `<think>` тегов для reasoning-моделей и защита от CoT-утечек.
* **Глубокий мониторинг:** Трекинг потребления CPU/GPU памяти и энергопотребления (через `nvidia-smi`) на каждом шаге.

---

## 🔬 Архитектура эксперимента (Lost-in-the-Middle)

Для каждого размера окна контекста (от 512 до 8192 токенов) запускаются 3 независимых прогона. Контекст «обстраивается» вокруг gold-фрагмента так, чтобы суммарная длина промпта (system + chat-template + user) строго укладывалась в лимит, а ответ находился в нужной зоне:

| Зона | Флаг `--positions` | Точка ответа в окне |
| :--- | :--- | :--- |
| **0 — 30%** | `start` | ~15% |
| **30 — 60%** | `middle` | ~45% |
| **60 — 100%** | `end` | ~80% |

## 🧠 Поддерживаемые модели

Код автоматически обрабатывает специфику загрузки и генерации для следующих архитектур:

| Ключ | Идентификатор на HuggingFace | Режим по умолчанию |
| :--- | :--- | :--- |
| `gemma3_1b` | `google/gemma-3-1b-it` | bf16 |
| `deepseek_r1_qwen_1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | bf16 |
| `qwen3_0.6b` | `Qwen/Qwen3-0.6B` | bf16, `/no_think` |
| `smollm2_360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` | bf16 |
| `phi4_mini` | `microsoft/Phi-4-mini-instruct` | **4-bit (bnb)** |

*Примечание: Phi-4-mini (3.8B параметров) в FP16 не помещается в 9 ГБ VRAM при длинном контексте, поэтому автоматически загружается в 4-bit (NF4 + double quantization).*

---

## 🛠 Требования и Установка

```bash
pip install -r requirements.txt

```

*Примечание для Windows: для корректной работы 4-bit квантования (Phi-4) может потребоваться специальная сборка `bitsandbytes`:*

```bash
pip install bitsandbytes --extra-index-url [https://jllllll.github.io/bitsandbytes-windows-webui](https://jllllll.github.io/bitsandbytes-windows-webui)

```

---

## 💻 Запуск и использование

### Полная сетка эксперимента

Запуск пайплайна на всех моделях, окнах и позициях:

```powershell
python run_experiment.py `
  --input-path "ADD YOUR PATH" `
  --models gemma3_1b qwen3_0.6b smollm2_360m deepseek_r1_qwen_1.5b phi4_mini `
  --window-sizes 512 1024 2048 4096 8192 -1 `
  --positions start middle end `
  --max-new-tokens 64 --temperature 0.0 `
  --metrics em f1 recall bertscore bleurt `
  --run-name decoder_litm_full

```

### Быстрый Smoke-test

Проверка работоспособности на одной легкой модели и сокращенном пуле вопросов:

```powershell
python run_experiment.py `
  --input-path "...\output" `
  --models smollm2_360m `
  --window-sizes 512 1024 `
  --positions middle `
  --metrics em f1 recall `
  --limit-questions 50 `
  --run-name smoke_test

```

### Полезные флаги конфигурации

* `--quantize` — принудительно указать модели для загрузки в 4-bit (по умолчанию берутся из реестра).
* `--max-input-tokens` — жесткий лимит на входные токены (по умолчанию равен `model.max_position_embeddings`).
* `--cpu-fallback` — при OutOfMemory (OOM) временно переносить генерацию примера на CPU (медленно, не поддерживается для 4-bit моделей).
* `--limit-questions N` — обрезать датасет до N уникальных вопросов (для дебага).
* `--bertscore-model` / `--bleurt-model` — переопределить чекпоинты семантических метрик.

---

## 📁 Структура проекта и логика работы

Эксперимент разбит на независимые фазы во избежание переполнения памяти: (1) Генерация ответов + базовые метрики (EM/F1), (2) Очистка VRAM от LLM и загрузка тяжелых моделей для BERTScore/BLEURT.

* `prompts.py` — фиксированные system prompt и user templates для воспроизводимости.
* `config.py` — настройка CLI (`argparse`), реестр моделей `MODEL_REGISTRY` и расчет соотношений позиций `POSITION_RATIOS`.
* `data_utils.py` — итератор датасета `*.jsonl.gz`, бинарный поиск оптимального окна контекста вокруг gold-фрагмента.
* `decoder_inference.py` — инициализация `AutoModelForCausalLM` с chat-template, обработка хаков вроде `enable_thinking=False` (для Qwen3), парсинг `is_autocast_enabled` и `BitsAndBytesConfig`.
* `metrics.py` — ленивая загрузка NLP-метрик (BERTScore, BLEURT) и расчет bootstrap CI.
* `memory.py` — класс `MemoryTracker` для логирования пикового потребления и мощности.
* `run_experiment.py` — центральный контроллер, объединяющий генерацию и оценку.

### Защита от OOM (Out Of Memory)

1. Строгий cap по `model.max_position_embeddings`.
2. При возникновении `OutOfMemoryError`: в лог `predictions.json` записывается флаг `oom: true`, счетчик `n_oom` увеличивается, и пайплайн продолжает работу без падения.
3. Опция `--cpu-fallback` для спасения тяжелых примеров.

### Важные технические нюансы LLM

* **Температура:** Жестко `temperature=0.0` (greedy decoding). Это снижает вероятность галлюцинаций и необходимо для задач extractive oracle.
* **Qwen3:** Добавлен флаг `enable_thinking=False` и инструкция `/no_think` в user-prompt для предотвращения утечек Chain-of-Thought в итоговый ответ.
* **DeepSeek-R1-Distill:** Встроен пост-процессинг для автоматического удаления тегов `<think>...</think>`.

---

## 📊 Структура выходных данных

Все результаты сохраняются в директорию `logs_output/<run_name>/`:

```text
logs_output/<run_name>/
├── config.json
├── run.log
└── <model>/
    └── win_<size>/
        └── pos_<start|middle|end>/
            ├── predictions.json    # Ответы модели и контекст
            ├── metrics.json        # Итоговые метрики (EM, F1, BERTScore) + метаданные GPU
            └── memory_usage.json   # Детальный лог потребления VRAM/RAM

```

*Пример агрегированных метрик (`metrics.json`):*

```json
{
    "window_size": 512,
    "position": "middle",
    "model": "qwen3_0.6b",
    "count": 2235,
    "EM": 73.15,
    "EM_CI": [71.41, 74.99],
    "F1": 80.89,
    "F1_CI": [79.45, 82.36],
    "BERTScore": 91.23,
    "BLEURT": 0.6420,
    "gpu_ram_peak_mb": 1454.77,
    "n_oom": 0
}

```
