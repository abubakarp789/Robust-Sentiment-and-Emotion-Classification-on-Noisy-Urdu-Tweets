"""Aggregate isolated experiment runs using validation-only model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    from .run_experiments import MODEL_NAMES
    from .utils import load_config
except ImportError:
    from run_experiments import MODEL_NAMES
    from utils import load_config


METRICS = ("accuracy", "macro_f1", "weighted_f1")


def summarize_records(records: Iterable[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate repeated runs and select a model by mean validation macro-F1."""
    records = list(records)
    if not records:
        raise ValueError("No experiment records were provided")
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["family"], record["model"]), []).append(record)

    for (family, model), model_records in grouped.items():
        row: dict[str, Any] = {
            "model_family": family,
            "model_name": model,
            "seed_count": len(model_records),
            "seeds": ",".join(str(item["seed"]) for item in sorted(model_records, key=lambda item: item["seed"])),
        }
        for split in ("validation", "test"):
            for metric in METRICS:
                values = np.asarray([item[split][metric] for item in model_records], dtype=float)
                row[f"{split}_{metric}_mean"] = float(values.mean())
                row[f"{split}_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1_mean", "validation_accuracy_mean"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    winner = leaderboard.iloc[0]
    winner_records = grouped[(str(winner["model_family"]), str(winner["model_name"]))]
    canonical = max(
        winner_records,
        key=lambda item: (item["validation"]["macro_f1"], item["validation"]["accuracy"]),
    )
    summary = {
        "task": records[0]["task"],
        "ranking_metric": "validation_macro_f1_mean",
        "models": rows,
        "selected_model": {
            "model_family": str(winner["model_family"]),
            "model_name": str(winner["model_name"]),
            "canonical_seed": int(canonical["seed"]),
            "validation_macro_f1_mean": float(winner["validation_macro_f1_mean"]),
            "validation_macro_f1_std": float(winner["validation_macro_f1_std"]),
            "test_macro_f1_mean": float(winner["test_macro_f1_mean"]),
            "test_macro_f1_std": float(winner["test_macro_f1_std"]),
        },
    }
    return leaderboard, summary


def bootstrap_macro_f1(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute a deterministic non-parametric 95% macro-F1 interval."""
    truth = true_labels.to_numpy()
    prediction = predicted_labels.to_numpy()
    rng = np.random.default_rng(seed)
    scores = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.integers(0, len(truth), size=len(truth))
        scores[index] = f1_score(truth[chosen], prediction[chosen], average="macro", zero_division=0)
    return {
        "samples": int(samples),
        "lower_95": float(np.quantile(scores, 0.025)),
        "median": float(np.quantile(scores, 0.5)),
        "upper_95": float(np.quantile(scores, 0.975)),
    }


def collect_records(project_root: Path, task: str) -> list[dict[str, Any]]:
    """Read all complete run metrics for a task."""
    run_root = project_root / "outputs" / task / "runs"
    records: list[dict[str, Any]] = []
    for family, models in MODEL_NAMES.items():
        metrics_name = f"{family}_metrics.json"
        for model in models:
            for metrics_path in sorted((run_root / family / model).glob(f"seed_*/results/{metrics_name}")):
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                if model not in payload.get("models", {}):
                    continue
                seed = int(metrics_path.parents[1].name.removeprefix("seed_"))
                splits = payload["models"][model]
                records.append(
                    {
                        "task": task,
                        "family": family,
                        "model": model,
                        "seed": seed,
                        "validation": splits["validation"],
                        "test": splits["test"],
                    }
                )
    return records


def aggregate_task(config_path: str | Path, bootstrap_samples: int = 1000) -> dict[str, Any]:
    """Aggregate one task and write its official leaderboard and summary."""
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    task = str(config["labels"]["task"])
    records = collect_records(project_root, task)
    leaderboard, summary = summarize_records(records)
    results_dir = project_root / config["outputs"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    selected = summary["selected_model"]
    family = selected["model_family"]
    model = selected["model_name"]
    seed = selected["canonical_seed"]
    prediction_path = (
        project_root
        / "outputs"
        / task
        / "runs"
        / family
        / model
        / f"seed_{seed}"
        / "predictions"
        / f"{family}_{model}_test_predictions.csv"
    )
    predictions = pd.read_csv(prediction_path, usecols=["true_label", "predicted_label"])
    summary["selected_model"]["bootstrap_95"] = bootstrap_macro_f1(
        predictions["true_label"],
        predictions["predicted_label"],
        samples=bootstrap_samples,
        seed=seed,
    )
    summary["selected_model"]["prediction_path"] = str(
        prediction_path.relative_to(project_root)
    ).replace("\\", "/")
    leaderboard.to_csv(results_dir / "model_comparison_leaderboard.csv", index=False)
    (results_dir / "aggregate_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    summary = aggregate_task(args.config, args.bootstrap_samples)
    print(json.dumps(summary["selected_model"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
