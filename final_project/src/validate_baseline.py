"""Validate saved statistical-baseline artifacts and leakage audit metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from .train_baseline import (
        EVALUATION_SPLITS,
        MODEL_FILE_NAMES,
        REQUIRED_DATA_COLUMNS,
        REQUIRED_PREDICTION_COLUMNS,
        resolve_project_path,
    )
    from .utils import load_config
except ImportError:  # Support direct execution: python src/validate_baseline.py
    from train_baseline import (
        EVALUATION_SPLITS,
        MODEL_FILE_NAMES,
        REQUIRED_DATA_COLUMNS,
        REQUIRED_PREDICTION_COLUMNS,
        resolve_project_path,
    )
    from utils import load_config


def validate_baseline(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Check all baseline inputs, outputs, schemas, and fit-audit metadata."""
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    output_config = config["outputs"]
    split_dir = resolve_project_path(project_root, config["data"]["output_dir"])
    models_dir = resolve_project_path(project_root, output_config["models_dir"])
    results_dir = resolve_project_path(project_root, output_config["results_dir"])
    predictions_dir = resolve_project_path(project_root, output_config["predictions_dir"])
    checks: list[str] = []

    split_sizes: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_path = split_dir / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        frame = pd.read_csv(split_path, encoding="utf-8", usecols=list(REQUIRED_DATA_COLUMNS))
        missing = sorted(set(REQUIRED_DATA_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{split_path.name} missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{split_path.name} is empty")
        split_sizes[split] = len(frame)
    checks.append("split files and required columns")

    enabled_models = [
        name
        for name, model_config in config["baseline_models"]["models"].items()
        if model_config.get("enabled", False)
    ]
    for model_name in enabled_models:
        model_path = models_dir / MODEL_FILE_NAMES[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        pipeline = joblib.load(model_path)
        audit = getattr(pipeline, "fit_audit_", {})
        if audit.get("tfidf_fit_split") != "train":
            raise ValueError(f"{model_name} lacks a training-only TF-IDF fit audit")
        if audit.get("train_rows") != split_sizes["train"]:
            raise ValueError(f"{model_name} train row audit does not match train.csv")
        if audit.get("validation_rows_used_for_fit") != 0 or audit.get("test_rows_used_for_fit") != 0:
            raise ValueError(f"{model_name} audit indicates evaluation leakage")
    checks.append("saved pipelines and training-only TF-IDF audit")

    metrics_path = results_dir / "baseline_metrics.json"
    leaderboard_path = results_dir / "baseline_leaderboard.csv"
    if not metrics_path.exists() or not leaderboard_path.exists():
        raise FileNotFoundError("Missing baseline metrics or leaderboard")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if set(metrics.get("models", {})) != set(enabled_models):
        raise ValueError("Metrics file does not contain all enabled models")
    leaderboard = pd.read_csv(leaderboard_path)
    if leaderboard.empty or "macro_f1" not in leaderboard.columns:
        raise ValueError("Baseline leaderboard is empty or unreadable")
    checks.append("readable metrics and leaderboard")

    for model_name in enabled_models:
        for split in EVALUATION_SPLITS:
            prediction_path = (
                predictions_dir / f"baseline_{model_name}_{split}_predictions.csv"
            )
            if not prediction_path.exists():
                raise FileNotFoundError(f"Missing predictions: {prediction_path}")
            predictions = pd.read_csv(prediction_path, encoding="utf-8")
            if predictions.empty:
                raise ValueError(f"Prediction file is empty: {prediction_path}")
            missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns))
            if missing:
                raise ValueError(f"{prediction_path.name} missing columns: {missing}")
            expected_rows = split_sizes[split]
            if len(predictions) != expected_rows:
                raise ValueError(
                    f"{prediction_path.name} has {len(predictions)} rows; expected {expected_rows}"
                )
            matrix_path = (
                results_dir / f"confusion_matrix_baseline_{model_name}_{split}.csv"
            )
            if not matrix_path.exists() or pd.read_csv(matrix_path).empty:
                raise FileNotFoundError(f"Missing or empty confusion matrix: {matrix_path}")
    checks.append("prediction schemas, row counts, and confusion matrices")

    result = {
        "status": "passed",
        "checks": checks,
        "split_sizes": split_sizes,
        "models": enabled_models,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_baseline(args.config)


if __name__ == "__main__":
    main()
