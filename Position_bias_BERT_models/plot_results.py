import json
import matplotlib.pyplot as plt
import numpy as np

# Путь к твоему файлу с результатами
# Используем сырую строку (r""), чтобы Windows-пути с бекслешами читались корректно
FILE_PATH = r"/data/scratch/solodnicova/Position_Bias_bert_models/logs_output/longformer_position_bias_squadv2/position_bias_final.json"


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric(data, metric_name, ci_name, title, ylabel, save_path):
    # Группируем данные по размеру окна
    windows = sorted(list(set(item["window_size"] for item in data)))

    plt.figure(figsize=(10, 6))

    # Настройки стилей линий для разных размеров окон
    styles = {
        512: {"color": "#1f77b4", "marker": "o"},
        1024: {"color": "#ff7f0e", "marker": "s"},
        2048: {"color": "#2ca02c", "marker": "^"},
        4096: {"color": "#d62728", "marker": "D"},
        8192: {"color": "#9467bd", "marker": "v"},
    }

    for w in windows:
        # Фильтруем данные для текущего окна и сортируем по позиции
        w_data = sorted(
            [d for d in data if d["window_size"] == w], key=lambda x: x["position"]
        )

        positions = [
            d["position"] * 100 for d in w_data
        ]  # Переводим в проценты для оси X
        metric_values = [d[metric_name] for d in w_data]
        ci_lower = [d[ci_name][0] for d in w_data]
        ci_upper = [d[ci_name][1] for d in w_data]

        style = styles.get(w, {"color": "black", "marker": "x"})

        # Строим основную линию
        plt.plot(
            positions,
            metric_values,
            label=f"Context: {w} tokens",
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            markersize=7,
        )

        # Закрашиваем доверительный интервал
        plt.fill_between(
            positions, ci_lower, ci_upper, color=style["color"], alpha=0.15
        )

    # Оформление графика
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Позиция ответа в контексте (%)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(
        [0, 25, 50, 75, 100], ["0% (Начало)", "25%", "50%", "75%", "100% (Конец)"]
    )

    # Сетка и легенда
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Размер окна", fontsize=10, title_fontsize=11)

    # Настройка отступов и сохранение
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"График сохранен: {save_path}")


def main():
    try:
        data = load_data(FILE_PATH)
    except FileNotFoundError:
        print(f"Файл не найден по пути: {FILE_PATH}")
        return

    # График для Exact Match
    plot_metric(
        data=data,
        metric_name="EM",
        ci_name="EM_CI",
        title="Влияние позиции ответа на метрику Exact Match (Longformer)",
        ylabel="Exact Match Score (%)",
        save_path="position_bias_em_longformer.png",
    )

    # График для F1-Score
    plot_metric(
        data=data,
        metric_name="F1",
        ci_name="F1_CI",
        title="Влияние позиции ответа на метрику F1-Score (Longformer)",
        ylabel="F1 Score (%)",
        save_path="position_bias_f1_longformer.png",
    )


if __name__ == "__main__":
    main()
