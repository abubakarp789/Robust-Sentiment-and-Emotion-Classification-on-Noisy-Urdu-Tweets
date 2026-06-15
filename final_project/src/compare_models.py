"""Combine baseline and neural metrics into one model leaderboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

try:
    from .train_baseline import resolve_project_path
    from .utils import load_config
except ImportError:
    from train_baseline import resolve_project_path
    from utils import load_config


def build_comparison_rows(
    baseline_metrics: Mapping[str, Any] | None,
    neural_metrics: Mapping[str, Any] | None,
    transformer_metrics: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build comparison rows sorted by descending validation macro-F1."""
    rows: list[dict[str, Any]] = []
    
    # Validation metrics drive all model-comparison and selection decisions.
    svm_f1 = None
    if baseline_metrics and "models" in baseline_metrics and "linear_svm" in baseline_metrics["models"]:
        svm_f1 = baseline_metrics["models"]["linear_svm"]["validation"].get("macro_f1")
        
    families = [
        ("baseline", baseline_metrics),
        ("neural", neural_metrics),
        ("transformer", transformer_metrics),
    ]
    
    for family, metrics in families:
        if not metrics or "models" not in metrics:
            continue
        for model_name, splits in metrics["models"].items():
            validation = splits["validation"]
            test = splits["test"]
            per_class = test["per_class"]
            validation_macro_f1 = validation["macro_f1"]
            test_macro_f1 = test["macro_f1"]
            beats_svm = bool(validation_macro_f1 > svm_f1) if svm_f1 is not None else False
            
            rows.append(
                {
                    "model_family": family,
                    "model_name": model_name,
                    "validation_accuracy": validation["accuracy"],
                    "validation_macro_f1": validation_macro_f1,
                    "test_accuracy": test["accuracy"],
                    "test_macro_f1": test_macro_f1,
                    "test_weighted_f1": test["weighted_f1"],
                    "neutral_f1": per_class.get("Neutral", {}).get("f1", 0.0),
                    "negative_f1": per_class.get("Negative", {}).get("f1", 0.0),
                    "positive_f1": per_class.get("Positive", {}).get("f1", 0.0),
                    "beats_linear_svm": beats_svm,
                }
            )
    return sorted(rows, key=lambda row: row["validation_macro_f1"], reverse=True)


def compare_models(config_path: str | Path = "config.yaml") -> pd.DataFrame:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    results_dir = resolve_project_path(project_root, config["outputs"]["results_dir"])
    
    baseline_path = results_dir / "baseline_metrics.json"
    neural_path = results_dir / "neural_metrics.json"
    transformer_path = results_dir / "transformer_metrics.json"
    
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    neural = json.loads(neural_path.read_text(encoding="utf-8")) if neural_path.exists() else None
    transformer = json.loads(transformer_path.read_text(encoding="utf-8")) if transformer_path.exists() else None
    
    comparison = pd.DataFrame(build_comparison_rows(baseline, neural, transformer))
    output_path = results_dir / "model_comparison_leaderboard.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8")
    print(comparison.to_string(index=False))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    compare_models(parse_args().config)


if __name__ == "__main__":
    main()
