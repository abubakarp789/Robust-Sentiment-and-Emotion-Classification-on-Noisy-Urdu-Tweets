"""Shared utilities for the final project."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration, optionally inheriting from another YAML file."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        config = _deep_merge(load_config(parent_path), config)
    return config


def resolve_task_paths(config_path: str | Path) -> dict[str, Any]:
    """Resolve all task-owned data and artifact directories."""
    config_file = Path(config_path).resolve()
    root = config_file.parent
    config = load_config(config_file)
    task = str(config["labels"]["task"])

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path).resolve()

    data = config["data"]
    outputs = config["outputs"]
    processed_value = data.get("processed_dir", f"data/processed/{task}")
    return {
        "task": task,
        "project_root": root,
        "processed_dir": resolve(processed_value),
        "split_dir": resolve(data["output_dir"]),
        "results_dir": resolve(outputs["results_dir"]),
        "figures_dir": resolve(outputs["figures_dir"]),
        "predictions_dir": resolve(outputs["predictions_dir"]),
        "models_dir": resolve(outputs["models_dir"]),
        "error_analysis_dir": resolve(outputs["error_analysis_dir"]),
    }


def set_seed(seed: int) -> None:
    """Set common random seeds."""
    random.seed(seed)
    np.random.seed(seed)
