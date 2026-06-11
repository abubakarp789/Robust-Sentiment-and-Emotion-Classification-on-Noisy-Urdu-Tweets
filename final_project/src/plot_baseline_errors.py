"""Create matplotlib-only figures for baseline model error analysis."""

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
    from .error_analysis import add_error_categories, load_predictions
    from .train_baseline import resolve_project_path
    from .utils import load_config
except ImportError:
    from error_analysis import add_error_categories, load_predictions
    from train_baseline import resolve_project_path
    from utils import load_config


DISPLAY_NAMES = {
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression",
    "multinomial_nb": "Multinomial NB",
}
COLORS = {"validation": "#4472C4", "test": "#ED7D31"}


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_errors(config_path: str | Path = "config.yaml") -> list[Path]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    results_dir = resolve_project_path(project_root, config["outputs"]["results_dir"])
    predictions_dir = resolve_project_path(project_root, config["outputs"]["predictions_dir"])
    figures_dir = resolve_project_path(project_root, config["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((results_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(results_dir / "baseline_leaderboard.csv")
    outputs: list[Path] = []

    model_order = ["linear_svm", "logistic_regression", "multinomial_nb"]
    x = np.arange(len(model_order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, split in ((-width / 2, "validation"), (width / 2, "test")):
        values = [metrics["models"][model][split]["macro_f1"] for model in model_order]
        ax.bar(x + offset, values, width, label=split.title(), color=COLORS[split])
    ax.set_xticks(x, [DISPLAY_NAMES[m] for m in model_order], rotation=12)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Baseline Macro-F1 Comparison")
    ax.set_ylim(0, 0.6)
    ax.legend()
    path = figures_dir / "baseline_model_macro_f1_comparison.png"
    _finish(fig, path); outputs.append(path)

    test_rows = leaderboard.loc[leaderboard["split"].eq("test")]
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in test_rows.itertuples(index=False):
        ax.scatter(row.accuracy, row.macro_f1, s=90)
        ax.annotate(DISPLAY_NAMES.get(row.model_name, row.model_name), (row.accuracy, row.macro_f1), xytext=(6, 5), textcoords="offset points")
    ax.plot([0.35, 0.95], [0.35, 0.95], linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("Test Accuracy")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Accuracy Can Hide Minority-Class Errors")
    path = figures_dir / "baseline_model_accuracy_vs_macro_f1.png"
    _finish(fig, path); outputs.append(path)

    class_metrics = metrics["models"]["linear_svm"]["test"]["per_class"]
    labels = list(class_metrics)
    values = [class_metrics[label]["f1"] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#C0504D", "#9BBB59", "#4F81BD"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test F1")
    ax.set_title("Linear SVM Class-wise F1")
    path = figures_dir / "baseline_linear_svm_class_f1.png"
    _finish(fig, path); outputs.append(path)

    svm_predictions = load_predictions(predictions_dir / "baseline_linear_svm_test_predictions.csv")
    errors = add_error_categories(svm_predictions)
    category_counts = errors["error_category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(category_counts.index, category_counts.values, color="#8064A2")
    ax.set_xlabel("Number of test errors")
    ax.set_title("Linear SVM Error Categories")
    path = figures_dir / "baseline_linear_svm_error_distribution.png"
    _finish(fig, path); outputs.append(path)

    matrix = pd.read_csv(
        results_dir / "confusion_matrix_baseline_linear_svm_test.csv", index_col=0
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.values, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Count")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Linear SVM Test Confusion Matrix")
    threshold = matrix.values.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:,}", ha="center", va="center", color="white" if matrix.iloc[i, j] > threshold else "black")
    path = figures_dir / "baseline_linear_svm_confusion_heatmap.png"
    _finish(fig, path); outputs.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(svm_predictions.loc[svm_predictions["is_correct"], "confidence"].dropna(), bins=30, alpha=0.65, label="Correct", color="#4F81BD", density=True)
    ax.hist(svm_predictions.loc[~svm_predictions["is_correct"], "confidence"].dropna(), bins=30, alpha=0.65, label="Wrong", color="#C0504D", density=True)
    ax.set_xlabel("Normalized decision-margin confidence")
    ax.set_ylabel("Density")
    ax.set_title("Linear SVM Confidence Distribution")
    ax.legend()
    path = figures_dir / "baseline_linear_svm_confidence_distribution.png"
    _finish(fig, path); outputs.append(path)

    print(f"Saved {len(outputs)} baseline error figures to {figures_dir}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    plot_baseline_errors(parse_args().config)


if __name__ == "__main__":
    main()
