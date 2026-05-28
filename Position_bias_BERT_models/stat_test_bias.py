"""
Статистический анализ позиционного байаса ModernBERT.

Что делает этот скрипт по сравнению с исходной версией:
  1. Friedman test          — есть ли вообще различия между 5 позициями (для F1).
  2. Cochran's Q            — корректный аналог Фридмана ДЛЯ EM (бинарная метрика 0/100).
  3. Kendall's W            — РАЗМЕР ЭФФЕКТА для Фридмана (величина, а не только значимость).
  4. Page's trend test      — проверка МОНОТОННОГО убывания качества к концу (направление проверено).
  5. Post-hoc Wilcoxon      — попарные сравнения 5 позиций с поправкой Холма (какие позиции отличаются).

Запускается для двух моделей: base_result (базовая) и переобученный чекпоинт.
Cochran's Q берётся из statsmodels, при его отсутствии считается вручную на numpy.

Автор-помощник: научный руководитель :)
"""

import json
import os
import re
import string
import collections
import itertools
import numpy as np
from scipy import stats
from collections import defaultdict

# Cochran's Q: пробуем statsmodels, иначе ручная реализация (см. ниже)
try:
    from statsmodels.stats.contingency_tables import cochrans_q as _sm_cochrans_q
    _HAS_SM = True
except Exception:
    _HAS_SM = False


# --- 1. ФУНКЦИИ ДЛЯ РАСЧЕТА МЕТРИК (без изменений — совместимы с SQuAD/NQ) ---

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1 * 100.0


def exact_match_score(prediction, ground_truth):
    return 100.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


# --- 2. ДОПОЛНИТЕЛЬНЫЕ СТАТИСТИЧЕСКИЕ ФУНКЦИИ ---

def kendalls_w(aligned_scores, friedman_chi2):
    """
    Коэффициент конкордации Кендалла W — размер эффекта для критерия Фридмана.
    W = chi2 / (N * (k - 1)), где N — число объектов (документов), k — число условий (позиций).
    Диапазон [0, 1]: 0 — нет согласованности, 1 — полная.
    Ориентиры: ~0.1 слабый, ~0.3 умеренный, ~0.5 сильный эффект.
    """
    n = aligned_scores.shape[0]
    k = aligned_scores.shape[1]
    return friedman_chi2 / (n * (k - 1))


def cochrans_q_manual(binary_matrix):
    """
    Ручная реализация Cochran's Q для связанных бинарных данных (на случай отсутствия statsmodels).
    binary_matrix: массив N x k из 0/1.
    Q = (k-1) * [k * sum(C_j^2) - T^2] / [k*T - sum(R_i^2)]
        C_j — сумма по столбцу j (успехи в позиции j),
        R_i — сумма по строке i (успехи у документа i),
        T   — общая сумма успехов.
    Q ~ chi2 с (k-1) степенями свободы.
    """
    X = np.asarray(binary_matrix, dtype=float)
    N, k = X.shape
    C = X.sum(axis=0)          # по столбцам (позициям)
    R = X.sum(axis=1)          # по строкам (документам)
    T = X.sum()
    denom = (k * T - np.sum(R**2))
    if denom == 0:
        return np.nan, np.nan   # вырожденный случай (все ответы одинаковы)
    Q = (k - 1) * (k * np.sum(C**2) - T**2) / denom
    p = stats.chi2.sf(Q, k - 1)
    return Q, p


def cochrans_q(binary_matrix):
    """Cochran's Q через statsmodels, если доступен, иначе ручной расчёт."""
    if _HAS_SM:
        res = _sm_cochrans_q(np.asarray(binary_matrix, dtype=float))
        return float(res.statistic), float(res.pvalue)
    return cochrans_q_manual(binary_matrix)


def bootstrap_ci_mean(x, n_boot=5000, ci=95, seed=0):
    """
    Бутстрэп-перцентильный доверительный интервал для СРЕДНЕГО.
    Не предполагает нормальности — корректен для бимодальных F1 и бинарной EM.
    Возвращает (mean, lo, hi).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return x.mean(), lo, hi


def wilson_ci_proportion(successes, n, z=1.96):
    """
    Доверительный интервал Уилсона для биномиальной пропорции (для EM как доли успехов).
    Академический стандарт для долей; корректен даже при p у границ [0,1].
    Возвращает (mean%, lo%, hi%).
    """
    if n == 0:
        return np.nan, np.nan, np.nan
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p * 100.0, (centre - half) * 100.0, (centre + half) * 100.0


def posthoc_wilcoxon_holm(aligned_scores, positions):
    """
    Post-hoc: попарный знаковый критерий Уилкоксона между всеми парами позиций
    с поправкой Холма на множественные сравнения (10 пар при 5 позициях).
    Возвращает список строк-результатов.
    """
    k = len(positions)
    pairs = list(itertools.combinations(range(k), 2))
    raw = []
    for i, j in pairs:
        a, b = aligned_scores[:, i], aligned_scores[:, j]
        # zero_method='wilcox' отбрасывает нулевые разности; при полном совпадении вернём p=1
        if np.all(a - b == 0):
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = stats.wilcoxon(a, b, zero_method='wilcox')
            except ValueError:
                stat, p = 0.0, 1.0
        raw.append((i, j, stat, p))

    # Поправка Холма
    order = sorted(range(len(raw)), key=lambda idx: raw[idx][3])
    m = len(raw)
    adj = [None] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        p = raw[idx][3]
        p_holm = min(1.0, (m - rank) * p)
        p_holm = max(p_holm, prev)   # монотонность поправки Холма
        prev = p_holm
        adj[idx] = p_holm

    lines = []
    for (i, j, stat, p), p_holm in zip(raw, adj):
        sig = "знач." if p_holm < 0.05 else "н/з"
        lines.append(f"     {positions[i]} vs {positions[j]}: "
                     f"W={stat:.1f}, p_raw={p:.3e}, p_holm={p_holm:.3e} [{sig}]")
    return lines


# --- 3. ФУНКЦИЯ АНАЛИЗА ОДНОЙ КОНФИГУРАЦИИ ---

def evaluate_bias_with_trends(base_path, model_type, length, metric, log_file):
    def log_and_print(message):
        print(message)
        log_file.write(message + '\n')

    log_and_print(f"\n[{model_type.upper()}] Анализ для длины: {length} | Метрика: {metric.upper()}")

    positions = ['0.0', '0.25', '0.5', '0.75', '1.0']
    data_by_qid = defaultdict(dict)

    # Чтение файлов
    for pos in positions:
        file_path = os.path.join(base_path, model_type, str(length), pos, 'predictions_log.json')

        if not os.path.exists(file_path):
            log_and_print(f"ВНИМАНИЕ: Файл не найден -> {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
            for item in log_data:
                qid = item['qid']
                if metric == 'f1':
                    score = f1_score(item['pred'], item['gold'])
                else:
                    score = exact_match_score(item['pred'], item['gold'])
                data_by_qid[qid][pos] = score

    # Выравнивание по qid: берём только документы, прошедшие ВСЕ 5 позиций (связанные выборки)
    aligned_scores = []
    for qid, pos_scores in data_by_qid.items():
        if len(pos_scores) == len(positions):
            aligned_scores.append([pos_scores[p] for p in positions])

    aligned_scores = np.array(aligned_scores, dtype=float)

    if len(aligned_scores) == 0:
        log_and_print("Ошибка: Нет пересекающихся документов для всех позиций.")
        return

    N = len(aligned_scores)
    log_and_print(f"Успешно выровнено документов: {N}")
    means = aligned_scores.mean(axis=0)
    log_and_print(f"Средние по позициям (0.0 -> 1.0): {np.round(means, 2)}")

    # --- (0) 95% доверительные интервалы для среднего по КАЖДОЙ позиции ---
    # Бутстрэп-перцентильный CI (не предполагает нормальности; корректен для
    # бимодальной F1 и бинарной EM). Для EM дополнительно приводим интервал Уилсона.
    log_and_print("0. Доверительные интервалы (95%) по позициям:")
    for i, pos in enumerate(positions):
        col = aligned_scores[:, i]
        m, lo, hi = bootstrap_ci_mean(col, n_boot=10000, ci=95, seed=42)
        if metric == 'em':
            succ = int((col > 0).sum())
            wm, wlo, whi = wilson_ci_proportion(succ, N)
            log_and_print(f"   pos {pos}: mean={m:.2f}  "
                          f"bootstrap[{lo:.2f}, {hi:.2f}]  wilson[{wlo:.2f}, {whi:.2f}]")
        else:
            log_and_print(f"   pos {pos}: mean={m:.2f}  bootstrap[{lo:.2f}, {hi:.2f}]")

    # --- (1) Тест на наличие различий: Friedman (F1) или Cochran's Q (EM) ---
    if metric == 'em':
        # EM бинарна (0/100). Корректный тест для связанных бинарных данных — Cochran's Q.
        binary = (aligned_scores > 0).astype(int)   # 100 -> 1, 0 -> 0
        q_stat, q_p = cochrans_q(binary)
        src = "statsmodels" if _HAS_SM else "ручной расчёт"
        log_and_print(f"1. Критерий Cochran's Q [{src}] (корректен для бинарной EM): "
                      f"Q = {q_stat:.4f}, p-value = {q_p:.5e}")
        if q_p < 0.05:
            log_and_print("   Результат: Различия между позициями СТАТИСТИЧЕСКИ ЗНАЧИМЫ.")
        else:
            log_and_print("   Результат: Значимых различий не обнаружено.")
        # Фридман для EM считаем дополнительно (для сопоставимости с прежним отчётом), но помечаем
        chi2_f, p_f = stats.friedmanchisquare(*[aligned_scores[:, i] for i in range(len(positions))])
        log_and_print(f"   [справочно] Фридман на EM: chi2 = {chi2_f:.4f}, p = {p_f:.5e} "
                      f"(на бинарных данных много связок — основным считать Cochran's Q)")
    else:
        chi2_f, p_f = stats.friedmanchisquare(*[aligned_scores[:, i] for i in range(len(positions))])
        log_and_print(f"1. Критерий Фридмана: chi2 = {chi2_f:.4f}, p-value = {p_f:.5e}")
        if p_f < 0.05:
            log_and_print("   Результат: Различия между позициями СТАТИСТИЧЕСКИ ЗНАЧИМЫ.")
        else:
            log_and_print("   Результат: Значимых различий не обнаружено.")

    # --- (2) Размер эффекта: Kendall's W (на базе chi2 Фридмана) ---
    W = kendalls_w(aligned_scores, chi2_f)
    if W < 0.1:
        strength = "пренебрежимо малый"
    elif W < 0.3:
        strength = "слабый"
    elif W < 0.5:
        strength = "умеренный"
    else:
        strength = "сильный"
    log_and_print(f"2. Размер эффекта (Kendall's W): W = {W:.4f} -> {strength} эффект")

    # --- (3) Направленный тренд: Page's L (направление [5,4,3,2,1] = убывание, проверено) ---
    res_page = stats.page_trend_test(aligned_scores, predicted_ranks=[5, 4, 3, 2, 1])
    log_and_print(f"3. Критерий Пейджа (монотонное убывание к концу): "
                  f"L = {res_page.statistic:.4f}, p-value = {res_page.pvalue:.5e}")
    if res_page.pvalue < 0.05:
        log_and_print("   Результат: Монотонное ухудшение качества к концу документа ПОДТВЕРЖДЕНО.")
    else:
        log_and_print("   Результат: Монотонный спад НЕ подтверждён "
                      "(возможен немонотонный паттерн, напр. 'плато + обрыв').")

    # --- (4) Post-hoc: попарный Уилкоксон с поправкой Холма ---
    log_and_print("4. Post-hoc попарные сравнения (Wilcoxon + поправка Холма):")
    for line in posthoc_wilcoxon_holm(aligned_scores, positions):
        log_and_print(line)

    log_and_print("-" * 50)


# --- 4. АВТОМАТИЧЕСКИЙ ЦИКЛ ПО ВСЕМ ПАРАМЕТРАМ ---

if __name__ == "__main__":
    BASE_DIR = "/data/scratch/solodnicova/Position_Bias_bert_models/logs_output/"

    # Две модели: базовая и переобученный чекпоинт.
    # Ключ — имя подпапки внутри logs_output, значение — человекочитаемая метка.
    MODEL_DIRS = {
        "base_result": "БАЗОВАЯ модель",
        "run_checkpoint-6287_2026-05-12_11-59-46": "ПЕРЕОБУЧЕННАЯ модель",
    }

    lengths = [512, 1024, 2048, 4096, 8192]
    metrics = ['f1', 'em']

    for model_dir, label in MODEL_DIRS.items():
        out_txt = f"statistical_report_({model_dir}).txt"
        with open(out_txt, 'w', encoding='utf-8') as log_file:
            header = f"=== СТАТИСТИЧЕСКИЙ ОТЧЕТ ПО ПОЗИЦИОННОМУ БАЙАСУ (ModernBERT) — {label} ==="
            print("\n" + "=" * len(header))
            print(header)
            print("=" * len(header))
            log_file.write(header + "\n")
            for length in lengths:
                for metric in metrics:
                    evaluate_bias_with_trends(BASE_DIR, model_dir, length, metric, log_file)
        print(f"\n[ГОТОВО] Отчёт для '{label}' сохранён в файл: {out_txt}")