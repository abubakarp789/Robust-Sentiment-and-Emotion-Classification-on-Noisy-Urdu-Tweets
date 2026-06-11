"""Create matplotlib-only neural training and comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .train_baseline import resolve_project_path
    from .utils import load_config
except ImportError:
    from train_baseline import resolve_project_path
    from utils import load_config


DISPLAY_NAMES = {
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression",
    "multinomial_nb": "Multinomial NB",
    "text_cnn": "Text-CNN",
    "bilstm_attention": "BiLSTM-Attention",
}


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_neural_results(config_path: str | Path = "config.yaml") -> list[Path]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    results_dir = resolve_project_path(project_root, config["outputs"]["results_dir"])
    figures_dir = resolve_project_path(project_root, config["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((results_dir / "neural_metrics.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(results_dir / "model_comparison_leaderboard.csv")
    outputs: list[Path] = []

    models = list(metrics["models"])
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    for offset, split, color in (
        (-width / 2, "validation", "#4472C4"),
        (width / 2, "test", "#ED7D31"),
    ):
        values = [metrics["models"][model][split]["macro_f1"] for model in models]
        bars = ax.bar(x + offset, values, width, label=split.title(), color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, [DISPLAY_NAMES.get(model, model) for model in models])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Neural Model Macro-F1")
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    ax.legend()
    path = figures_dir / "neural_model_macro_f1_comparison.png"
    _finish(fig, path); outputs.append(path)

    plot_rows = comparison.loc[
        comparison["model_name"].isin(["linear_svm", *models])
    ].sort_values("test_macro_f1")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        [DISPLAY_NAMES.get(name, name) for name in plot_rows["model_name"]],
        plot_rows["test_macro_f1"],
        color=["#70AD47" if family == "neural" else "#4472C4" for family in plot_rows["model_family"]],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Test Macro-F1")
    ax.set_title("Neural Models vs Best Statistical Baseline")
    path = figures_dir / "neural_vs_baseline_macro_f1.png"
    _finish(fig, path); outputs.append(path)

    for model_name in ("text_cnn", "bilstm_attention"):
        history = pd.read_csv(results_dir / f"neural_{model_name}_training_history.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Train")
        axes[0].plot(history["epoch"], history["validation_loss"], marker="o", label="Validation")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
        axes[1].plot(history["epoch"], history["validation_macro_f1"], marker="o", color="#70AD47", label="Macro-F1")
        axes[1].plot(history["epoch"], history["validation_accuracy"], marker="o", color="#4472C4", label="Accuracy")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_ylim(0, 1); axes[1].legend()
        fig.suptitle(f"{DISPLAY_NAMES[model_name]} Training Curves")
        path = figures_dir / f"neural_training_curves_{model_name}.png"
        _finish(fig, path); outputs.append(path)

    best_model = max(models, key=lambda model: metrics["models"][model]["test"]["macro_f1"])
    matrix = pd.read_csv(
        results_dir / f"confusion_matrix_neural_{best_model}_test.csv", index_col=0
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.values, cmap="Purples")
    fig.colorbar(image, ax=ax, label="Count")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"Best Neural Model: {DISPLAY_NAMES[best_model]}")
    threshold = matrix.values.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:,}", ha="center", va="center", color="white" if matrix.iloc[i, j] > threshold else "black")
    path = figures_dir / "neural_best_model_confusion_heatmap.png"
    _finish(fig, path); outputs.append(path)

    family_best = comparison.groupby("model_family", as_index=False)["test_macro_f1"].max()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(family_best["model_family"].str.title(), family_best["test_macro_f1"], color=["#4472C4", "#70AD47"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, max(0.6, family_best["test_macro_f1"].max() + 0.1))
    ax.set_ylabel("Best Test Macro-F1")
    ax.set_title("Best Model by Family")
    path = figures_dir / "model_family_comparison.png"
    _finish(fig, path); outputs.append(path)

    print(f"Saved {len(outputs)} neural result figures to {figures_dir}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    plot_neural_results(parse_args().config)


if __name__ == "__main__":
    main()
