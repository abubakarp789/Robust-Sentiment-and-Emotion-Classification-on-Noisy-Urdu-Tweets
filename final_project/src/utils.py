"""Shared utilities for the final project."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set common random seeds."""
    random.seed(seed)
    np.random.seed(seed)
