"""Validate the complete dual-task benchmark and submission-facing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASKS = ("sentiment", "emotion")
SPLITS = ("train", "validation", "test")
SPLIT_PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))
EXPECTED_MODELS = {
    "logistic_regression",
    "linear_svm",
    "multinomial_nb",
    "text_cnn",
    "bilstm_attention",
    "mbert",
    "xlm_roberta",
    "urdu_roberta",
}
EXPECTED_SEEDS = {"baseline": 3, "neural": 3, "transformer": 1}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_official_benchmark(project_root: str | Path) -> dict[str, Any]:
    """Return a machine-readable integrity audit for both official tasks."""
    root = Path(project_root).resolve()
    errors: list[str] = []
    overlap_checks = 0
    selected_models: dict[str, str] = {}

    report_dir = root / "reports"
    manifest_path = report_dir / "experiment_manifest.json"
    _error(errors, manifest_path.is_file(), "Missing reports/experiment_manifest.json")
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {"runs": []}
    runs = manifest.get("runs", [])

    protocol = manifest.get("protocol", {})
    _error(errors, protocol.get("selection_metric") == "validation_macro_f1_mean", "Selection metric is not validation macro-F1.")
    _error(errors, protocol.get("human_gold_evaluation") is False, "Manifest must disclose that no human-gold evaluation was performed.")
    _error(errors, len(runs) == 36, f"Expected 36 official runs; found {len(runs)}.")

    for task in TASKS:
        split_dir = root / "data" / "splits" / task
        split_frames: dict[str, pd.DataFrame] = {}
        for split in SPLITS:
            path = split_dir / f"{split}.csv"
            _error(errors, path.is_file(), f"Missing {task} {split} split: {path}")
            if path.is_file():
                frame = pd.read_csv(path, usecols=["id", "clean_text", "task_label"], encoding="utf-8")
                _error(errors, not frame.empty, f"{task} {split} split is empty.")
                _error(errors, not frame[["id", "clean_text", "task_label"]].isna().any().any(), f"{task} {split} contains missing required values.")
                split_frames[split] = frame

        if len(split_frames) == 3:
            for left, right in SPLIT_PAIRS:
                for column in ("id", "clean_text"):
                    overlap_checks += 1
                    left_values = set(split_frames[left][column].astype(str))
                    right_values = set(split_frames[right][column].astype(str))
                    overlap = left_values.intersection(right_values)
                    _error(errors, not overlap, f"{task}: {column} overlap between {left} and {right}: {len(overlap)}")

        results_dir = root / "outputs" / task / "results"
        aggregate_path = results_dir / "aggregate_metrics.json"
        leaderboard_path = results_dir / "model_comparison_leaderboard.csv"
        split_summary_path = results_dir / "split_summary.json"
        for path in (aggregate_path, leaderboard_path, split_summary_path):
            _error(errors, path.is_file(), f"Missing official result: {path}")
        if not aggregate_path.is_file() or not leaderboard_path.is_file():
            continue

        aggregate = _load_json(aggregate_path)
        leaderboard = pd.read_csv(leaderboard_path)
        model_names = set(leaderboard["model_name"].astype(str))
        _error(errors, model_names == EXPECTED_MODELS, f"{task}: leaderboard model set is incomplete: {sorted(model_names)}")
        _error(errors, aggregate.get("ranking_metric") == "validation_macro_f1_mean", f"{task}: aggregate ranking metric is incorrect.")
        _error(errors, leaderboard.iloc[0]["model_name"] == aggregate["selected_model"]["model_name"], f"{task}: leaderboard and aggregate selections disagree.")
        selected_models[task] = aggregate["selected_model"]["model_name"]

        task_runs = [run for run in runs if run.get("task") == task]
        _error(errors, len(task_runs) == 18, f"{task}: expected 18 runs; found {len(task_runs)}.")
        for family, expected_seed_count in EXPECTED_SEEDS.items():
            family_runs = [run for run in task_runs if run.get("family") == family]
            family_models = {run.get("model") for run in family_runs}
            expected_models = {row["model_name"] for row in aggregate["models"] if row["model_family"] == family}
            _error(errors, family_models == expected_models, f"{task}/{family}: manifest model set differs from aggregate.")
            for model in expected_models:
                seeds = {int(run["seed"]) for run in family_runs if run.get("model") == model}
                _error(errors, len(seeds) == expected_seed_count, f"{task}/{family}/{model}: expected {expected_seed_count} seeds; found {sorted(seeds)}.")

        for run in task_runs:
            metrics_path = root / run["metrics_path"]
            _error(errors, metrics_path.is_file(), f"Missing run metrics: {metrics_path}")
            artifacts = run.get("artifacts", [])
            _error(errors, bool(artifacts), f"No saved model artifacts for {task}/{run.get('family')}/{run.get('model')}/seed_{run.get('seed')}")
            for artifact in artifacts:
                artifact_path = root / artifact["path"]
                _error(errors, artifact_path.is_file(), f"Missing model artifact: {artifact_path}")
                if artifact_path.is_file():
                    _error(errors, artifact_path.stat().st_size == int(artifact["bytes"]), f"Artifact size mismatch: {artifact_path}")

            if metrics_path.is_file():
                run_root = metrics_path.parents[1]
                prediction_dir = run_root / "predictions"
                for split in ("validation", "test"):
                    predictions = list(prediction_dir.glob(f"*_{split}_predictions.csv"))
                    _error(errors, len(predictions) == 1, f"Expected one {split} prediction file in {prediction_dir}; found {len(predictions)}.")

    required_reports = (
        "final_report.md",
        "final_evaluation_summary.md",
        "dataset_card.md",
        "model_card.md",
    )
    for filename in required_reports:
        path = report_dir / filename
        _error(errors, path.is_file() and path.stat().st_size > 0, f"Missing or empty report: {path}")

    return {
        "valid": not errors,
        "tasks": list(TASKS),
        "run_count": len(runs),
        "overlap_checks": overlap_checks,
        "selected_models": selected_models,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate_official_benchmark(args.project_root)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Official runs: {result['run_count']}")
        print(f"Cross-split overlap checks: {result['overlap_checks']}")
        print(f"Selected models: {result['selected_models']}")
        for error in result["errors"]:
            print(f"[FAILURE] {error}")
        if result["valid"]:
            print("[SUCCESS] Dual-task benchmark integrity validation passed.")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
