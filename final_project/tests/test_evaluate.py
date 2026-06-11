from __future__ import annotations

import json

import pandas as pd

from src.evaluate import (
    build_leaderboard,
    compute_classification_metrics,
    save_classification_report,
    save_confusion_matrix,
)


def test_compute_classification_metrics_includes_aggregate_and_per_class_values() -> None:
    metrics = compute_classification_metrics(
        ["Positive", "Negative", "Neutral", "Positive"],
        ["Positive", "Negative", "Positive", "Neutral"],
        labels=["Negative", "Neutral", "Positive"],
    )

    assert set(metrics) >= {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "per_class",
    }
    assert set(metrics["per_class"]) == {"Negative", "Neutral", "Positive"}
    assert metrics["per_class"]["Negative"]["support"] == 1


def test_report_confusion_matrix_and_leaderboard_are_exported(tmp_path) -> None:
    y_true = ["Positive", "Negative", "Neutral"]
    y_pred = ["Positive", "Positive", "Neutral"]
    labels = ["Negative", "Neutral", "Positive"]
    report_path = tmp_path / "report.json"
    matrix_path = tmp_path / "matrix.csv"
    leaderboard_path = tmp_path / "leaderboard.csv"

    save_classification_report(y_true, y_pred, report_path)
    save_confusion_matrix(y_true, y_pred, labels, matrix_path)
    build_leaderboard(
        {
            "model_a": {"validation": {"macro_f1": 0.4}},
            "model_b": {"validation": {"macro_f1": 0.6}},
        },
        leaderboard_path,
    )

    assert json.loads(report_path.read_text(encoding="utf-8"))["accuracy"] == 2 / 3
    matrix = pd.read_csv(matrix_path, index_col=0)
    assert list(matrix.index) == labels
    assert list(matrix.columns) == labels
    leaderboard = pd.read_csv(leaderboard_path)
    assert leaderboard.iloc[0]["model_name"] == "model_b"
    assert leaderboard.iloc[0]["split"] == "validation"
