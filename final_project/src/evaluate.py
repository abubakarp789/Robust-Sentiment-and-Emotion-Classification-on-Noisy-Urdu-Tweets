"""Reusable evaluation utilities for classification experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""
    resolved_labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=resolved_labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=resolved_labels, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=resolved_labels, average=None, zero_division=0
    )

    class_metrics = {
        label: {
            "precision": float(per_class[0][index]),
            "recall": float(per_class[1][index]),
            "f1": float(per_class[2][index]),
            "support": int(per_class[3][index]),
        }
        for index, label in enumerate(resolved_labels)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": class_metrics,
    }


def save_classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], output_path: str | Path
) -> None:
    """Save the sklearn classification report as UTF-8 JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def save_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    output_path: str | Path,
) -> None:
    """Save a labelled confusion matrix as UTF-8 CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    frame = pd.DataFrame(matrix, index=list(labels), columns=list(labels))
    frame.index.name = "true_label"
    frame.to_csv(path, encoding="utf-8")


def build_leaderboard(metrics: Mapping[str, Any], output_path: str | Path) -> None:
    """Create a model/split leaderboard sorted by macro-F1."""
    model_metrics = metrics.get("models", metrics)
    rows: list[dict[str, Any]] = []
    for model_name, split_metrics in model_metrics.items():
        for split, values in split_metrics.items():
            if not isinstance(values, Mapping) or "macro_f1" not in values:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "accuracy": values["accuracy"] if "accuracy" in values else None,
                    "macro_precision": values.get("macro_precision"),
                    "macro_recall": values.get("macro_recall"),
                    "macro_f1": values["macro_f1"],
                    "weighted_precision": values.get("weighted_precision"),
                    "weighted_recall": values.get("weighted_recall"),
                    "weighted_f1": values.get("weighted_f1"),
                }
            )

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["macro_f1", "weighted_f1"], ascending=False, na_position="last"
        ).reset_index(drop=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(path, index=False, encoding="utf-8")
