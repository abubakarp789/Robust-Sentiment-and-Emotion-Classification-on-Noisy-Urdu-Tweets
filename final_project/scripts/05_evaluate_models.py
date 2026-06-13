"""Recompute metrics from saved prediction CSV files without retraining models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_workflow import evaluate_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "metrics")
    parser.add_argument("--pattern", default="*_predictions.csv")
    args = parser.parse_args()
    summary = evaluate_predictions(args.input_dir, args.output_dir, args.pattern)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
