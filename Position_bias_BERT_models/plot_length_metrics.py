import os
import json
import re
import string
import collections
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# --- НАСТРОЙКИ ПУТЕЙ И ПАРАМЕТРОВ ---
BASE_DIR = "/data/scratch/solodnicova/Position_Bias_bert_models/logs_output/"
MODEL_TYPE = "base_result"
# Список длин контекста, которые нужно обработать
LENGTHS = [512, 1024, 2048, 4096, 8192] 
# Фиксированная позиция для анализа (середина)
TARGET_POSITION = "0.5" 

# --- 1. МЕТРИКИ И НОРМАЛИЗАЦИЯ (Полное совпадение методик) ---
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    # Корректная обработка пустых строк из первого эксперимента
    if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
        return 100.0 if prediction_tokens == ground_truth_tokens else 0.0
        
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: 
        return 0.0
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall) * 100.0

def exact_match_score(prediction, ground_truth):
    return 100.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

# --- 2. БУТСТРЕП ДЛЯ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (как в твоем примере) ---
def bootstrap_ci(values: List[float], n_samples: int = 1000, confidence_level: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    if not values: 
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum([values[rng.randrange(n)] for _ in range(n)]) / n for _ in range(n_samples))
    alpha = 1.0 - confidence_level
    return means[int((alpha / 2) * n_samples)], means[int((1 - alpha / 2) * n_samples) - 1]

# --- 3. СБОР И СТАТИСТИЧЕСКАЯ ОБРАБОТКА ДАННЫХ ---
def main():
    window_sizes = []
    em_values = []
    em_ci_low = []
    em_ci_high = []
    f1_values = []
    f1_ci_low = []
    f1_ci_high = []

    print(f"=== Сбор данных из папки: {os.path.join(BASE_DIR, MODEL_TYPE)} ===")
    
    for length in LENGTHS:
        file_path = os.path.join(BASE_DIR, MODEL_TYPE, str(length), TARGET_POSITION, 'predictions_log.json')
        
        if not os.path.exists(file_path):
            print(f"Пропуск: файл для длины {length} не найден -> {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
            
        em_list = []
        f1_list = []
        
        for item in log_data:
            em_list.append(exact_match_score(item['pred'], item['gold']))
            f1_list.append(f1_score(item['pred'], item['gold']))
            
        if not em_list:
            print(f"Предупреждение: Файл для длины {length} пустой.")
            continue
            
        # Сохраняем размер окна
        window_sizes.append(length)
        
        # Считаем средние значения
        mean_em = np.mean(em_list)
        mean_f1 = np.mean(f1_list)
        em_values.append(mean_em)
        f1_values.append(mean_f1)
        
        # Считаем доверительные интервалы через Bootstrap
        em_low, em_high = bootstrap_ci(em_list)
        f1_low, f1_high = bootstrap_ci(f1_list)
        
        em_ci_low.append(em_low)
        em_ci_high.append(em_high)
        f1_ci_low.append(f1_low)
        f1_ci_high.append(f1_high)
        
        print(f"Длина {length:4d} | Документов: {len(em_list):4d} | EM: {mean_em:.2f}% ({em_low:.1f}-{em_high:.1f}) | F1: {mean_f1:.2f}% ({f1_low:.1f}-{f1_high:.1f})")

    if not window_sizes:
        print("Ошибка: Не найдено ни одного файла с данными для построения графика.")
        return

    # --- 4. ОРИГИНАЛЬНОЕ ПОСТРОЕНИЕ ГРАФИКА ---
    styles = {
        "EM": {"color": "#1f77b4", "marker": "o", "label": "Exact Match (EM)"},
        "F1": {"color": "#ff7f0e", "marker": "s", "label": "F1-Score"},
    }

    plt.figure(figsize=(10, 6))

    # Линия и область CI для Exact Match
    plt.plot(window_sizes, em_values, label=styles["EM"]["label"], color=styles["EM"]["color"], 
             marker=styles["EM"]["marker"], linewidth=2, markersize=8)
    plt.fill_between(window_sizes, em_ci_low, em_ci_high, color=styles["EM"]["color"], alpha=0.15)

    # Линия и область CI для F1-Score
    plt.plot(window_sizes, f1_values, label=styles["F1"]["label"], color=styles["F1"]["color"], 
             marker=styles["F1"]["marker"], linewidth=2, markersize=8)
    plt.fill_between(window_sizes, f1_ci_low, f1_ci_high, color=styles["F1"]["color"], alpha=0.15)

    # Оформление
    title = f"Зависимость качества от длины контекста ({MODEL_TYPE})\nПозиция ответа: {int(float(TARGET_POSITION) * 100)}% (середина)"
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Длина последовательности (window_size)", fontsize=12)
    plt.ylabel("Значение метрики (%)", fontsize=12)
    
    plt.xticks(window_sizes)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Метрики", fontsize=10, title_fontsize=11, loc="best")

    plt.tight_layout()
    save_path = f"quality_vs_length_{MODEL_TYPE}_pos_{TARGET_POSITION}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[ГОТОВО] График успешно сохранен в: {save_path}")

if __name__ == "__main__":
    main()