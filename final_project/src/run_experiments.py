"""Run isolated task/model/seed experiments without overwriting artifacts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

try:
    from .train_baseline import train_baselines
    from .train_neural import train_neural_models
    from .train_transformer import train_transformer_models
    from .utils import load_config
except ImportError:
    from train_baseline import train_baselines
    from train_neural import train_neural_models
    from train_transformer import train_transformer_models
    from utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "baseline": ("logistic_regression", "linear_svm", "multinomial_nb"),
    "neural": ("text_cnn", "bilstm_attention"),
    "transformer": ("mbert", "xlm_roberta", "urdu_roberta"),
}


def build_run_config(
    task_config_path: str | Path,
    *,
    family: str,
    model: str,
    seed: int,
) -> dict[str, Any]:
    """Return a complete config with one model and isolated output directories."""
    if family not in MODEL_NAMES or model not in MODEL_NAMES[family]:
        raise ValueError(f"Unsupported experiment: {family}/{model}")
    config = load_config(task_config_path)
    task = str(config["labels"]["task"])
    run_root = f"outputs/{task}/runs/{family}/{model}/seed_{seed}"
    config["project"]["random_seed"] = int(seed)
    config["neural_models"]["training"]["random_seed"] = int(seed)
    config["transformer_models"]["training"]["random_seed"] = int(seed)
    config["outputs"] = {
        "results_dir": f"{run_root}/results",
        "figures_dir": f"{run_root}/figures",
        "predictions_dir": f"{run_root}/predictions",
        "models_dir": f"{run_root}/models",
        "error_analysis_dir": f"{run_root}/error_analysis",
    }

    config["baseline_models"]["models"] = {
        name: {**values, "enabled": family == "baseline" and name == model}
        for name, values in config["baseline_models"]["models"].items()
    }
    config["neural_models"]["models"] = {
        name: {**values, "enabled": family == "neural" and name == model}
        for name, values in config["neural_models"]["models"].items()
    }
    config["transformer_models"]["models"] = {
        name: {**values, "enabled": family == "transformer" and name == model}
        for name, values in config["transformer_models"]["models"].items()
    }
    config["run"] = {"task": task, "family": family, "model": model, "seed": int(seed)}
    return config


def execute_run(
    task_config_path: str | Path,
    *,
    family: str,
    model: str,
    seed: int,
) -> dict[str, Any]:
    """Execute one isolated experiment and return its metrics payload."""
    config = build_run_config(task_config_path, family=family, model=model, seed=seed)
    experiment = config.get("experiments", {})
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"run_{config['labels']['task']}_{family}_{model}_{seed}_",
        dir=PROJECT_ROOT,
        encoding="utf-8",
        delete=False,
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        run_config_path = Path(handle.name)
    try:
        if family == "baseline":
            metrics = train_baselines(run_config_path)
        elif family == "neural":
            metrics = train_neural_models(run_config_path, selected_model=model)
        else:
            metrics = train_transformer_models(
                run_config_path,
                sample_size=int(experiment.get("transformer_sample_size", 50_000)),
                epochs_override=int(experiment.get("transformer_epochs", 1)),
                selected_model=model,
            )
    finally:
        run_config_path.unlink(missing_ok=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_run_config(
        args.config, family=args.family, model=args.model, seed=args.seed
    )
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    execute_run(args.config, family=args.family, model=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
