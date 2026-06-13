"""Evaluate saved model predictions without retraining models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluate import (
    compute_classification_metrics,
    save_classification_report,
    save_confusion_matrix,
)


def evaluate_predictions(input_dir: Path, output_dir: Path, pattern: str) -> pd.DataFrame:
    """Recompute metrics for every saved prediction file matching ``pattern``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files matched {input_dir / pattern}")

    for path in files:
        frame = pd.read_csv(path, encoding="utf-8")
        required = {"true_label", "predicted_label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        labels = sorted(set(frame["true_label"]) | set(frame["predicted_label"]))
        metrics = compute_classification_metrics(
            frame["true_label"], frame["predicted_label"], labels=labels
        )
        stem = path.stem
        (output_dir / f"{stem}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_classification_report(
            frame["true_label"],
            frame["predicted_label"],
            output_dir / f"{stem}_classification_report.json",
        )
        save_confusion_matrix(
            frame["true_label"],
            frame["predicted_label"],
            labels,
            output_dir / f"{stem}_confusion_matrix.csv",
        )
        rows.append(
            {
                "prediction_file": path.name,
                "rows": len(frame),
                **{key: value for key, value in metrics.items() if key != "per_class"},
            }
        )

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False, encoding="utf-8")
    return summary
