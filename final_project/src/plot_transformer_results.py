"""Create matplotlib-only transformer and combined family comparison figures."""

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
    "mbert": "mBERT",
    "xlm_roberta": "XLM-RoBERTa",
    "urdu_roberta": "Urdu-RoBERTa",
}


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_transformer_results(config_path: str | Path = "config.yaml") -> list[Path]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    results_dir = resolve_project_path(project_root, config["outputs"]["results_dir"])
    figures_dir = resolve_project_path(project_root, config["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = json.loads((results_dir / "transformer_metrics.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(results_dir / "model_comparison_leaderboard.csv")
    outputs: list[Path] = []
    
    models = list(metrics["models"])
    
    # 1. transformer_model_macro_f1_comparison.png
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
    ax.set_title("Transformer Model Macro-F1 Comparison")
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    ax.legend()
    path = figures_dir / "transformer_model_macro_f1_comparison.png"
    _finish(fig, path)
    outputs.append(path)
    
    # 2. transformer_vs_baseline_neural_macro_f1.png
    # Plot all transformer models compared to all baseline and neural models on test split
    test_rows = comparison.sort_values("test_macro_f1")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = []
    for family in test_rows["model_family"]:
        if family == "transformer":
            colors.append("#70AD47")  # Green for transformer
        elif family == "neural":
            colors.append("#FFC000")  # Yellow/Gold for neural
        else:
            colors.append("#4472C4")  # Blue for baseline
            
    bars = ax.barh(
        [DISPLAY_NAMES.get(name, name) for name in test_rows["model_name"]],
        test_rows["test_macro_f1"],
        color=colors,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Test Macro-F1")
    ax.set_title("All Models Test Macro-F1 Comparison")
    path = figures_dir / "transformer_vs_baseline_neural_macro_f1.png"
    _finish(fig, path)
    outputs.append(path)
    
    # 3. transformer_best_model_class_f1.png
    # Find the best transformer model
    best_transformer = max(models, key=lambda m: metrics["models"][m]["test"]["macro_f1"])
    class_metrics = metrics["models"][best_transformer]["test"]["per_class"]
    classes = list(class_metrics.keys())
    class_f1s = [class_metrics[c]["f1"] for c in classes]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(classes, class_f1s, color="#70AD47", width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test F1-Score")
    ax.set_title(f"Class F1-Scores: Best Transformer ({DISPLAY_NAMES.get(best_transformer, best_transformer)})")
    path = figures_dir / "transformer_best_model_class_f1.png"
    _finish(fig, path)
    outputs.append(path)
    
    # 4. transformer_best_model_confusion_heatmap.png
    matrix = pd.read_csv(
        results_dir / f"confusion_matrix_transformer_{best_transformer}_test.csv", index_col=0
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.values, cmap="Greens")
    fig.colorbar(image, ax=ax, label="Count")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Best Transformer ({DISPLAY_NAMES.get(best_transformer, best_transformer)}) Confusion Matrix")
    threshold = matrix.values.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix.iloc[i, j]:,}",
                ha="center", va="center",
                color="white" if matrix.iloc[i, j] > threshold else "black"
            )
    path = figures_dir / "transformer_best_model_confusion_heatmap.png"
    _finish(fig, path)
    outputs.append(path)
    
    # 5. final_model_family_comparison.png
    family_best = comparison.groupby("model_family", as_index=False)["test_macro_f1"].max()
    # Sort family best by test_macro_f1 ascending for plotting
    family_best = family_best.sort_values("test_macro_f1")
    fig, ax = plt.subplots(figsize=(6, 5))
    colors_family = []
    for fam in family_best["model_family"]:
        if fam == "transformer":
            colors_family.append("#70AD47")
        elif fam == "neural":
            colors_family.append("#FFC000")
        else:
            colors_family.append("#4472C4")
            
    bars = ax.bar(family_best["model_family"].str.title(), family_best["test_macro_f1"], color=colors_family, width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, max(0.6, family_best["test_macro_f1"].max() + 0.1))
    ax.set_ylabel("Best Test Macro-F1")
    ax.set_title("Best Model Test Macro-F1 by Family")
    path = figures_dir / "final_model_family_comparison.png"
    _finish(fig, path)
    outputs.append(path)
    
    print(f"Saved {len(outputs)} transformer result figures to {figures_dir}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    plot_transformer_results(parse_args().config)


if __name__ == "__main__":
    main()
