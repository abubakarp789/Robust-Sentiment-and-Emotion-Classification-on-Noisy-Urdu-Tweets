"""Validate Milestone 3 neural checkpoints, metrics, predictions, and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd
import torch

try:
    from .train_baseline import REQUIRED_PREDICTION_COLUMNS, resolve_project_path
    from .train_neural import EVALUATION_SPLITS, MODEL_FILES
    from .utils import load_config
except ImportError:
    from train_baseline import REQUIRED_PREDICTION_COLUMNS, resolve_project_path
    from train_neural import EVALUATION_SPLITS, MODEL_FILES
    from utils import load_config


FIGURES = (
    "neural_model_macro_f1_comparison.png",
    "neural_vs_baseline_macro_f1.png",
    "neural_training_curves_text_cnn.png",
    "neural_training_curves_bilstm_attention.png",
    "neural_best_model_confusion_heatmap.png",
    "model_family_comparison.png",
)


def validate_neural(config_path: str | Path = "config.yaml") -> dict[str, object]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    models_dir = resolve_project_path(project_root, config["outputs"]["models_dir"])
    results_dir = resolve_project_path(project_root, config["outputs"]["results_dir"])
    predictions_dir = resolve_project_path(project_root, config["outputs"]["predictions_dir"])
    figures_dir = resolve_project_path(project_root, config["outputs"]["figures_dir"])
    enabled_models = [
        name
        for name, values in config["neural_models"]["models"].items()
        if values.get("enabled", False)
    ]

    for name in ("neural_vocab.json", "neural_label_mapping.json"):
        path = models_dir / name
        if not path.exists() or not json.loads(path.read_text(encoding="utf-8")):
            raise FileNotFoundError(path)
    for model_name in enabled_models:
        checkpoint_path = models_dir / MODEL_FILES[model_name]
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = checkpoint.get("metadata", {})
        if metadata.get("vocabulary_fit_split") != "train":
            raise ValueError(f"{model_name} checkpoint lacks train-only vocabulary audit")

    metrics_path = results_dir / "neural_metrics.json"
    leaderboard_path = results_dir / "neural_leaderboard.csv"
    comparison_path = results_dir / "model_comparison_leaderboard.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if set(metrics["models"]) != set(enabled_models):
        raise ValueError("Neural metrics do not contain every enabled model")
    for path in (leaderboard_path, comparison_path):
        if not path.exists() or pd.read_csv(path).empty:
            raise FileNotFoundError(path)

    for model_name in enabled_models:
        history = results_dir / f"neural_{model_name}_training_history.csv"
        if not history.exists() or pd.read_csv(history).empty:
            raise FileNotFoundError(history)
        for split in EVALUATION_SPLITS:
            prediction_path = predictions_dir / f"neural_{model_name}_{split}_predictions.csv"
            predictions = pd.read_csv(prediction_path, encoding="utf-8")
            if predictions.empty:
                raise ValueError(f"Empty predictions: {prediction_path}")
            missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns))
            if missing:
                raise ValueError(f"{prediction_path.name} missing columns: {missing}")
            matrix = results_dir / f"confusion_matrix_neural_{model_name}_{split}.csv"
            report = results_dir / f"classification_report_neural_{model_name}_{split}.json"
            if not matrix.exists() or pd.read_csv(matrix).empty:
                raise FileNotFoundError(matrix)
            if not report.exists() or not json.loads(report.read_text(encoding="utf-8")):
                raise FileNotFoundError(report)

    for name in FIGURES:
        path = figures_dir / name
        if not path.exists() or path.stat().st_size == 0 or mpimg.imread(path).size == 0:
            raise FileNotFoundError(path)

    result = {
        "status": "passed",
        "models": enabled_models,
        "prediction_files": len(enabled_models) * len(EVALUATION_SPLITS),
        "figures": len(FIGURES),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    validate_neural(parse_args().config)


if __name__ == "__main__":
    main()
