"""Generate submission figures from saved experimental artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_figures(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Create dataset, leaderboard, and confusion-matrix figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    split_summary = json.loads((results_dir / "split_summary.json").read_text(encoding="utf-8"))
    distribution = pd.Series(split_summary["class_distribution_before_split"]).sort_values()
    ax = distribution.plot(kind="barh", color="#3b82f6", figsize=(7, 4))
    ax.set_title(f"{split_summary['task'].title()} Class Distribution")
    ax.set_xlabel("Examples")
    plt.tight_layout()
    path = figures_dir / "class_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()
    created.append(path)

    leaderboard = pd.read_csv(results_dir / "model_comparison_leaderboard.csv")
    plot_data = leaderboard[["model_name", "test_macro_f1", "test_weighted_f1"]].melt(
        id_vars="model_name", var_name="metric", value_name="score"
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_data, x="model_name", y="score", hue="metric")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Test Macro-F1 vs Weighted-F1")
    plt.tight_layout()
    path = figures_dir / "macro_f1_vs_weighted_f1.png"
    plt.savefig(path, dpi=200)
    plt.close()
    created.append(path)

    for matrix_path in sorted(results_dir.glob("confusion_matrix_*_test.csv")):
        matrix = pd.read_csv(matrix_path, index_col=0)
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues")
        title = matrix_path.stem.replace("confusion_matrix_", "").replace("_", " ").title()
        plt.title(title)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        output = figures_dir / f"{matrix_path.stem}.png"
        plt.savefig(output, dpi=180)
        plt.close()
        created.append(output)

    return created
