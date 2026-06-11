"""Generate baseline error-analysis artifacts from existing prediction CSVs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .error_analysis import (
        add_error_categories,
        compute_error_summary,
        export_error_report,
        load_predictions,
        sample_errors_by_class,
        sample_errors_by_confusion_pair,
    )
    from .train_baseline import resolve_project_path
    from .utils import load_config
except ImportError:
    from error_analysis import (
        add_error_categories,
        compute_error_summary,
        export_error_report,
        load_predictions,
        sample_errors_by_class,
        sample_errors_by_confusion_pair,
    )
    from train_baseline import resolve_project_path
    from utils import load_config


PREDICTION_PATTERN = re.compile(
    r"^baseline_(?P<model>.+)_(?P<split>validation|test)_predictions\.csv$"
)


def analyze_baseline_errors(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Analyze every saved baseline prediction file without retraining models."""
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    predictions_dir = resolve_project_path(project_root, config["outputs"]["predictions_dir"])
    output_dir = resolve_project_path(project_root, config["outputs"]["error_analysis_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_config = config.get("error_analysis", {})
    high_threshold = float(analysis_config.get("high_confidence_threshold", 0.80))
    low_threshold = float(analysis_config.get("low_confidence_threshold", 0.55))
    short_max = int(analysis_config.get("short_text_max_tokens", 3))
    minority_classes = set(analysis_config.get("minority_classes", ["Neutral"]))

    aggregate: dict[str, Any] = {
        "best_baseline_model": "linear_svm",
        "thresholds": {
            "high_confidence": high_threshold,
            "low_confidence": low_threshold,
            "short_text_max_tokens": short_max,
            "minority_classes": sorted(minority_classes),
        },
        "models": {},
    }

    prediction_paths = sorted(predictions_dir.glob("baseline_*_predictions.csv"))
    if not prediction_paths:
        raise FileNotFoundError(f"No baseline prediction files found in {predictions_dir}")

    for prediction_path in prediction_paths:
        match = PREDICTION_PATTERN.match(prediction_path.name)
        if not match:
            continue
        model_name = match.group("model")
        split = match.group("split")
        frame = load_predictions(prediction_path)
        if frame["model_name"].nunique() != 1 or frame["model_name"].iloc[0] != model_name:
            raise ValueError(f"Model name mismatch in {prediction_path.name}")
        if frame["split"].nunique() != 1 or frame["split"].iloc[0] != split:
            raise ValueError(f"Split mismatch in {prediction_path.name}")

        paths = export_error_report(frame, output_dir, model_name, split)
        errors = add_error_categories(frame)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        summary["artifacts"] = {key: str(value.relative_to(project_root)) for key, value in paths.items()}
        aggregate["models"].setdefault(model_name, {})[split] = summary

        # Reproducible samples for qualitative inspection for every model/split.
        sample_errors_by_class(errors, n=20).to_csv(
            output_dir / f"baseline_{model_name}_{split}_sample_by_class.csv",
            index=False,
            encoding="utf-8",
        )
        sample_errors_by_confusion_pair(errors, n=20).to_csv(
            output_dir / f"baseline_{model_name}_{split}_sample_by_confusion_pair.csv",
            index=False,
            encoding="utf-8",
        )

        if model_name == "linear_svm":
            errors.loc[errors["confidence"].ge(high_threshold)].sort_values(
                "confidence", ascending=False
            ).to_csv(
                output_dir / f"baseline_linear_svm_{split}_high_confidence_wrong.csv",
                index=False,
                encoding="utf-8",
            )
            frame.loc[
                frame["is_correct"] & frame["confidence"].le(low_threshold)
            ].sort_values("confidence").to_csv(
                output_dir / f"baseline_linear_svm_{split}_low_confidence_correct.csv",
                index=False,
                encoding="utf-8",
            )
            errors.loc[errors["text_length"].le(short_max)].to_csv(
                output_dir / f"baseline_linear_svm_{split}_short_text_errors.csv",
                index=False,
                encoding="utf-8",
            )
            errors.loc[errors["true_label"].isin(minority_classes)].to_csv(
                output_dir / f"baseline_linear_svm_{split}_minority_class_errors.csv",
                index=False,
                encoding="utf-8",
            )

    summary_path = output_dir / "baseline_error_summary.json"
    summary_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved baseline error analysis to {output_dir}")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    analyze_baseline_errors(parse_args().config)


if __name__ == "__main__":
    main()
