"""Create an optional balanced annotation sample from the saved test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_LABELS = ("Positive", "Negative", "Neutral")
OUTPUT_COLUMNS = (
    "id",
    "raw_text",
    "clean_text",
    "current_label",
    "manual_label",
    "annotator_notes",
)
ANNOTATION_README = """# Annotation Sample

`annotation_sample.csv` is an optional support file for future manual review.

- It was sampled from the existing test split with a fixed random seed.
- It was not used for model training.
- It was not used for model evaluation.
- It does not affect any reported metric.
- It can later support a manually verified clean test subset.
"""


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (project_root / path).resolve()


def build_annotation_sample(
    test_split: pd.DataFrame,
    per_class: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic class-stratified annotation frame."""
    required = {"id", "raw_text", "clean_text", "task_label"}
    missing = sorted(required.difference(test_split.columns))
    if missing:
        raise ValueError(f"Test split is missing required columns: {missing}")

    sampled_groups = []
    for label in TARGET_LABELS:
        class_rows = test_split.loc[test_split["task_label"].eq(label)]
        sample_size = min(per_class, len(class_rows))
        if sample_size:
            sampled_groups.append(class_rows.sample(n=sample_size, random_state=random_seed))

    if not sampled_groups:
        raise ValueError("No supported sentiment labels were found in the test split.")

    sample = pd.concat(sampled_groups, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    sample = sample.rename(columns={"task_label": "current_label"})
    sample["manual_label"] = ""
    sample["annotator_notes"] = ""
    return sample[list(OUTPUT_COLUMNS)]


def create_annotation_sample(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the saved test split and write optional annotation support files."""
    resolved_config = Path(config_path or PROJECT_ROOT / "config.yaml").resolve()
    config = load_config(resolved_config)
    project_root = resolved_config.parent
    split_dir = resolve_project_path(project_root, config["data"]["output_dir"])
    output_dir = project_root / "data" / "annotation"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_path = split_dir / "test.csv"
    test_split = pd.read_csv(test_path, encoding="utf-8")
    seed = int(config["project"]["random_seed"])
    sample = build_annotation_sample(test_split, per_class=100, random_seed=seed)

    sample_path = output_dir / "annotation_sample.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8")
    (output_dir / "annotation_readme.md").write_text(ANNOTATION_README, encoding="utf-8")

    distribution = {
        label: int(sample["current_label"].eq(label).sum()) for label in TARGET_LABELS
    }
    summary = {
        "annotation_sample_path": "data/annotation/annotation_sample.csv",
        "rows": int(len(sample)),
        "class_distribution": distribution,
        "used_for_training": False,
        "used_for_evaluation": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced manual-annotation sample.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    create_annotation_sample(args.config)


if __name__ == "__main__":
    main()
