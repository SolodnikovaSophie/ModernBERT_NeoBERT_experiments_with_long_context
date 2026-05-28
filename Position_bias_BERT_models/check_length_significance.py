import os
import json
import re
import string
import collections
import numpy as np
from scipy import stats
from collections import defaultdict

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
    if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
        return 100.0 if prediction_tokens == ground_truth_tokens else 0.0
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall) * 100.0

def exact_match_score(prediction, ground_truth):
    return 100.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def analyze_length_significance(base_path, model_type, target_pos='0.5', metric='f1'):
    lengths = [512, 1024, 2048, 4096]
    data_by_qid = defaultdict(dict)
    
    # 1. Сбор данных по всем длинам для конкретной позиции
    for length in lengths:
        file_path = os.path.join(base_path, model_type, str(length), target_pos, 'predictions_log.json')
        if not os.path.exists(file_path):
            print(f"Пропуск длины {length}: файл не найден.")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
            for item in log_data:
                qid = item['qid']
                score = f1_score(item['pred'], item['gold']) if metric == 'f1' else exact_match_score(item['pred'], item['gold'])
                data_by_qid[qid][length] = score

    # 2. Выравнивание выборки (пересечение qid по всем длинам)
    aligned_scores = []
    for qid, length_scores in data_by_qid.items():
        if len(length_scores) == len(lengths):  # Документ есть во всех 4 конфигурациях длины
            aligned_scores.append([length_scores[l] for l in lengths])
            
    matrix = np.array(aligned_scores)
    
    print(f"=== СТАТИСТИЧЕСКИЙ АНАЛИЗ ВЛИЯНИЯ ДЛИНЫ КОНТЕКСТА ===")
    print(f"Модель: {model_type} | Позиция: {target_pos} | Метрика: {metric.upper()}")
    print(f"Размер выровненной выборки документов: {len(matrix)}")
    
    for i, l in enumerate(lengths):
        print(f"Среднее для длины {l}: {np.mean(matrix[:, i]):.2f}%")
        
    print("-" * 50)
    
    # 3. Тест Фридмана (зависимые выборки, так как документы одни и те же)
    stat_f, p_f = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    print(f"Критерий Фридмана: Statistic = {stat_f:.4f}, p-value = {p_f:.5e}")
    if p_f < 0.05:
        print("Результат: Влияние длины контекста СТАТИСТИЧЕСКИ ЗНАЧИМО.")
    else:
        print("Результат: Влияние длины контекста не подтверждено.")

if __name__ == "__main__":
    BASE_DIR = "/data/scratch/solodnicova/Position_Bias_bert_models/logs_output/"
    # Запустим для модифицированной модели (замени имя папки, если оно отличается)
    analyze_length_significance(BASE_DIR, 'base_result', target_pos='0.5', metric='f1')