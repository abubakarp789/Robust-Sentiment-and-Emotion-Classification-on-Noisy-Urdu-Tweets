"""Stable entry point for local-only Transformer fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train_transformer import main  # noqa: E402


if "--config" not in sys.argv:
    sys.argv.extend(["--config", str(ROOT / "config.yaml")])


if __name__ == "__main__":
    main()
