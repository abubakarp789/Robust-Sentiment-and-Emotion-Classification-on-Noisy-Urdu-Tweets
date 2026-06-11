"""Validate Milestone 4 transformer checkpoints, metrics, predictions, explanations, and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd

try:
    from .train_baseline import REQUIRED_PREDICTION_COLUMNS, resolve_project_path
    from .utils import load_config
except ImportError:
    from train_baseline import REQUIRED_PREDICTION_COLUMNS, resolve_project_path
    from utils import load_config


EVALUATION_SPLITS = ("validation", "test")

FIGURES = (
    "transformer_model_macro_f1_comparison.png",
    "transformer_vs_baseline_neural_macro_f1.png",
    "transformer_best_model_class_f1.png",
    "transformer_best_model_confusion_heatmap.png",
    "final_model_family_comparison.png",
)


def validate_transformer(config_path: str | Path = "config.yaml") -> dict[str, object]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    
    models_dir = resolve_project_path(project_root, config["outputs"]["models_dir"])
    results_dir = resolve_project_path(project_root, config["outputs"]["results_dir"])
    predictions_dir = resolve_project_path(project_root, config["outputs"]["predictions_dir"])
    figures_dir = resolve_project_path(project_root, config["outputs"]["figures_dir"])
    error_analysis_dir = resolve_project_path(project_root, config["outputs"]["error_analysis_dir"])
    
    enabled_models = [
        name
        for name, m_cfg in config["transformer_models"]["models"].items()
        if m_cfg.get("enabled", False)
    ]
    
    # 1. Check label mapping file
    label_mapping = models_dir / "transformer_label_mapping.json"
    if not label_mapping.exists() or not json.loads(label_mapping.read_text(encoding="utf-8")):
        raise FileNotFoundError(label_mapping)
        
    # 2. Check saved model directories for enabled models
    for model_name in enabled_models:
        best_dir = models_dir / f"transformer_{model_name}" / "best"
        if not best_dir.exists():
            raise FileNotFoundError(f"Missing best model directory: {best_dir}")
        config_path = best_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in best model: {config_path}")
            
    # 3. Check metrics, leaderboard, training metadata and comparison leaderboard
    metrics_path = results_dir / "transformer_metrics.json"
    leaderboard_path = results_dir / "transformer_leaderboard.csv"
    metadata_path = results_dir / "transformer_training_metadata.json"
    comparison_path = results_dir / "model_comparison_leaderboard.csv"
    
    for path in (metrics_path, leaderboard_path, metadata_path, comparison_path):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty result file: {path}")
            
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if set(metrics["models"]) != set(enabled_models):
        raise ValueError(f"Transformer metrics key mismatch: {set(metrics['models'])} vs {set(enabled_models)}")
        
    # 4. Check model comparison leaderboard columns and ordering
    comp_df = pd.read_csv(comparison_path)
    required_comp_cols = {
        "model_family", "model_name", "validation_accuracy", "validation_macro_f1",
        "test_accuracy", "test_macro_f1", "test_weighted_f1", "neutral_f1",
        "negative_f1", "positive_f1", "beats_linear_svm"
    }
    missing_comp = sorted(required_comp_cols - set(comp_df.columns))
    if missing_comp:
        raise ValueError(f"Comparison leaderboard missing columns: {missing_comp}")
        
    # Check sorting order (descending test_macro_f1)
    if not comp_df["test_macro_f1"].is_monotonic_decreasing:
        raise ValueError("Model comparison leaderboard is not sorted by test_macro_f1 descending")
        
    # 5. Check prediction files, confusion matrices, reports for each enabled model
    for model_name in enabled_models:
        for split in EVALUATION_SPLITS:
            prediction_path = predictions_dir / f"transformer_{model_name}_{split}_predictions.csv"
            predictions = pd.read_csv(prediction_path, encoding="utf-8")
            if predictions.empty:
                raise ValueError(f"Empty predictions file: {prediction_path}")
                
            missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns))
            if missing:
                raise ValueError(f"{prediction_path.name} missing columns: {missing}")
                
            matrix = results_dir / f"confusion_matrix_transformer_{model_name}_{split}.csv"
            report = results_dir / f"classification_report_transformer_{model_name}_{split}.json"
            
            if not matrix.exists() or pd.read_csv(matrix).empty:
                raise FileNotFoundError(matrix)
            if not report.exists() or not json.loads(report.read_text(encoding="utf-8")):
                raise FileNotFoundError(report)
                
    # 6. Check explanation samples
    explanation_samples = error_analysis_dir / "explanation_samples.json"
    if not explanation_samples.exists() or not json.loads(explanation_samples.read_text(encoding="utf-8")):
        raise FileNotFoundError(explanation_samples)
        
    for model_name in enabled_models:
        exp_samples = error_analysis_dir / f"explanation_samples_{model_name}.json"
        if not exp_samples.exists() or not json.loads(exp_samples.read_text(encoding="utf-8")):
            raise FileNotFoundError(exp_samples)
            
    # 7. Check figures
    for name in FIGURES:
        path = figures_dir / name
        if not path.exists() or path.stat().st_size == 0 or mpimg.imread(path).size == 0:
            raise FileNotFoundError(path)
            
    result = {
        "status": "passed",
        "models": enabled_models,
        "prediction_files": len(enabled_models) * len(EVALUATION_SPLITS),
        "figures": len(FIGURES),
        "explanation_assistant": "verified"
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    validate_transformer(parse_args().config)


if __name__ == "__main__":
    main()
