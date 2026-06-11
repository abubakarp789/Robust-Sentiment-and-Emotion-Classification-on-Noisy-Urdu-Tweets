"""Validate baseline error-analysis tables, summaries, and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd

try:
    from .error_analysis import REQUIRED_PREDICTION_COLUMNS
    from .train_baseline import resolve_project_path
    from .utils import load_config
except ImportError:
    from error_analysis import REQUIRED_PREDICTION_COLUMNS
    from train_baseline import resolve_project_path
    from utils import load_config


FIGURES = (
    "baseline_model_macro_f1_comparison.png",
    "baseline_model_accuracy_vs_macro_f1.png",
    "baseline_linear_svm_class_f1.png",
    "baseline_linear_svm_error_distribution.png",
    "baseline_linear_svm_confusion_heatmap.png",
    "baseline_linear_svm_confidence_distribution.png",
)


def validate_error_analysis(config_path: str | Path = "config.yaml") -> dict[str, object]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    error_dir = resolve_project_path(project_root, config["outputs"]["error_analysis_dir"])
    figures_dir = resolve_project_path(project_root, config["outputs"]["figures_dir"])

    summary_path = error_dir / "baseline_error_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "linear_svm" not in summary.get("models", {}):
        raise ValueError("Aggregate summary is missing Linear SVM")

    required_csvs = {
        "baseline_linear_svm_test_misclassified.csv": set(REQUIRED_PREDICTION_COLUMNS) | {"error_category"},
        "baseline_linear_svm_validation_misclassified.csv": set(REQUIRED_PREDICTION_COLUMNS) | {"error_category"},
        "baseline_linear_svm_test_confusion_pairs.csv": {"true_label", "predicted_label", "count", "share_of_errors"},
        "baseline_linear_svm_validation_confusion_pairs.csv": {"true_label", "predicted_label", "count", "share_of_errors"},
        "baseline_linear_svm_test_class_errors.csv": {"true_label", "support", "errors", "error_rate"},
        "baseline_linear_svm_validation_class_errors.csv": {"true_label", "support", "errors", "error_rate"},
        "baseline_linear_svm_test_high_confidence_wrong.csv": set(REQUIRED_PREDICTION_COLUMNS) | {"error_category"},
        "baseline_linear_svm_test_low_confidence_correct.csv": set(REQUIRED_PREDICTION_COLUMNS),
        "baseline_linear_svm_test_short_text_errors.csv": set(REQUIRED_PREDICTION_COLUMNS) | {"error_category"},
        "baseline_linear_svm_test_minority_class_errors.csv": set(REQUIRED_PREDICTION_COLUMNS) | {"error_category"},
    }
    for name, required_columns in required_csvs.items():
        path = error_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, encoding="utf-8")
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if name.endswith("misclassified.csv") and frame.empty:
            raise ValueError(f"{name} is empty")

    readable_tables = 0
    readable_summaries = 0
    for path in sorted(error_dir.glob("*.csv")):
        pd.read_csv(path, encoding="utf-8")
        readable_tables += 1
    for path in sorted(error_dir.glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
        readable_summaries += 1

    for name in FIGURES:
        path = figures_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        image = mpimg.imread(path)
        if image.size == 0:
            raise ValueError(f"Unreadable figure: {path}")

    result = {
        "status": "passed",
        "summary": str(summary_path.relative_to(project_root)),
        "validated_csv_files": len(required_csvs),
        "readable_csv_files": readable_tables,
        "readable_json_files": readable_summaries,
        "validated_figures": len(FIGURES),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    validate_error_analysis(parse_args().config)


if __name__ == "__main__":
    main()
