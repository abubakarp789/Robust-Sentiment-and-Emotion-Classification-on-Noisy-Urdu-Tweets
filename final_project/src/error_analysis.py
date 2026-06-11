"""Reusable analysis utilities for baseline classification errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = (
    "id",
    "raw_text",
    "clean_text",
    "true_label",
    "predicted_label",
    "confidence",
    "split",
    "model_name",
    "is_correct",
    "text_length",
)


def load_predictions(prediction_path: str | Path) -> pd.DataFrame:
    """Load and validate a baseline prediction CSV with UTF-8 text."""
    path = Path(prediction_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8", dtype={"id": "string"})
    missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"Prediction file is empty: {path}")
    frame["is_correct"] = frame["true_label"].eq(frame["predicted_label"])
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["text_length"] = pd.to_numeric(frame["text_length"], errors="coerce")
    return frame


def get_misclassified_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where the predicted label differs from the true label."""
    return df.loc[df["true_label"].ne(df["predicted_label"])].copy().reset_index(drop=True)


def _sample_grouped(
    df: pd.DataFrame, group_columns: Sequence[str], n: int
) -> pd.DataFrame:
    errors = get_misclassified_examples(df)
    if errors.empty:
        return errors
    samples = []
    for _, group in errors.groupby(list(group_columns), sort=True, dropna=False):
        samples.append(group.sample(n=min(n, len(group)), random_state=42))
    return pd.concat(samples, ignore_index=True)


def sample_errors_by_class(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Sample misclassified examples grouped by true label."""
    return _sample_grouped(df, ["true_label"], n)


def sample_errors_by_confusion_pair(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Sample errors grouped by true-label to predicted-label pair."""
    return _sample_grouped(df, ["true_label", "predicted_label"], n)


def compute_error_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute total errors, rate, class-wise errors, and confusion-pair counts."""
    errors = get_misclassified_examples(df)
    total_rows = int(len(df))
    total_errors = int(len(errors))

    class_errors: dict[str, dict[str, Any]] = {}
    for label, group in df.groupby("true_label", sort=True):
        error_count = int(group["true_label"].ne(group["predicted_label"]).sum())
        support = int(len(group))
        class_errors[str(label)] = {
            "support": support,
            "errors": error_count,
            "error_rate": float(error_count / support) if support else 0.0,
        }

    pair_counts = (
        errors.groupby(["true_label", "predicted_label"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "true_label", "predicted_label"], ascending=[False, True, True])
    )
    confusion_pairs = [
        {
            "true_label": str(row.true_label),
            "predicted_label": str(row.predicted_label),
            "count": int(row.count),
            "share_of_errors": float(row.count / total_errors) if total_errors else 0.0,
        }
        for row in pair_counts.itertuples(index=False)
    ]

    summary: dict[str, Any] = {
        "total_rows": total_rows,
        "total_correct": total_rows - total_errors,
        "total_errors": total_errors,
        "error_rate": float(total_errors / total_rows) if total_rows else 0.0,
        "class_errors": class_errors,
        "confusion_pairs": confusion_pairs,
    }
    if "error_category" in errors.columns:
        summary["error_categories"] = {
            str(key): int(value)
            for key, value in errors["error_category"].value_counts().items()
        }
    return summary


def categorize_error_type(row: pd.Series) -> str:
    """Assign a deterministic heuristic category to a misclassified example."""
    true_label = str(row.get("true_label", ""))
    predicted_label = str(row.get("predicted_label", ""))
    length = pd.to_numeric(row.get("text_length"), errors="coerce")
    confidence = pd.to_numeric(row.get("confidence"), errors="coerce")

    if true_label == predicted_label:
        return "correct"
    if true_label == "Neutral" and predicted_label == "Positive":
        return "neutral_to_positive"
    if true_label == "Neutral":
        return "minority_class_confusion"
    if pd.notna(length) and length <= 3:
        return "short_text_ambiguity"
    if pd.notna(length) and length >= 30:
        return "long_text_noise"
    if pd.notna(confidence) and confidence >= 0.80:
        return "possible_label_noise"
    if true_label == "Positive" and predicted_label == "Negative":
        return "positive_to_negative"
    if true_label == "Negative" and predicted_label == "Positive":
        return "negative_to_positive"
    return "unknown"


def add_error_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Return misclassified examples with an error-category column."""
    errors = get_misclassified_examples(df)
    errors["error_category"] = errors.apply(categorize_error_type, axis=1)
    return errors


def export_error_report(
    df: pd.DataFrame, output_dir: str | Path, model_name: str, split: str
) -> dict[str, Path]:
    """Save misclassifications plus class, confusion-pair, and JSON summaries."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"baseline_{model_name}_{split}"
    errors = add_error_categories(df)
    summary = compute_error_summary(errors if len(errors) == len(df) else df.assign(
        error_category=df.apply(
            lambda row: categorize_error_type(row)
            if row["true_label"] != row["predicted_label"]
            else "correct",
            axis=1,
        )
    ))
    summary["model_name"] = model_name
    summary["split"] = split

    class_rows = [
        {"true_label": label, **values}
        for label, values in summary["class_errors"].items()
    ]
    pair_rows = summary["confusion_pairs"]
    paths = {
        "misclassified": directory / f"{prefix}_misclassified.csv",
        "class_errors": directory / f"{prefix}_class_errors.csv",
        "confusion_pairs": directory / f"{prefix}_confusion_pairs.csv",
        "summary": directory / f"{prefix}_summary.json",
    }
    errors.to_csv(paths["misclassified"], index=False, encoding="utf-8")
    pd.DataFrame(class_rows).to_csv(paths["class_errors"], index=False, encoding="utf-8")
    pd.DataFrame(
        pair_rows,
        columns=["true_label", "predicted_label", "count", "share_of_errors"],
    ).to_csv(paths["confusion_pairs"], index=False, encoding="utf-8")
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


# Compatibility wrappers for the original placeholder API.
def save_misclassified_examples(
    predictions: pd.DataFrame, output_path: str | Path
) -> pd.DataFrame:
    errors = get_misclassified_examples(predictions)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_path, index=False, encoding="utf-8")
    return errors


def generate_error_summary(errors: pd.DataFrame) -> dict[str, Any]:
    return compute_error_summary(errors)
