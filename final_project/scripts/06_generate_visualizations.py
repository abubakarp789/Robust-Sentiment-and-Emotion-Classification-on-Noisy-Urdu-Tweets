"""Generate evidence-backed figures from saved metrics and confusion matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization_workflow import generate_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "results")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()
    for path in generate_figures(args.results_dir, args.figures_dir):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
