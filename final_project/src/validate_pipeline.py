"""Validate one task-specific preprocessing and group-safe split pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .preprocessing import preprocess_text
    from .utils import load_config, resolve_task_paths
except ImportError:
    from preprocessing import preprocess_text
    from utils import load_config, resolve_task_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_REQUIRED_COLUMNS = {"Id", "Text", "Category"}
PROCESSED_REQUIRED_COLUMNS = {"id", "raw_text", "clean_text", "raw_label", "normalized_label", "task_label", "text_length"}


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_data_assets(config: dict) -> None:
    """Validate raw data and the task-owned processed/split artifacts."""
    raw_path = _resolve(config["data"]["raw_dataset_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset is missing: {raw_path}")
    raw_header = pd.read_csv(raw_path, nrows=0, encoding="utf-8")
    missing_raw = RAW_REQUIRED_COLUMNS.difference(raw_header.columns)
    if missing_raw:
        raise ValueError(f"Raw dataset is missing columns: {sorted(missing_raw)}")

    task = config["labels"]["task"]
    processed_dir = _resolve(config["data"].get("processed_dir", f"data/processed/{task}"))
    processed_path = processed_dir / config["data"].get("processed_filename", "dataset.csv")
    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed {task} dataset is missing: {processed_path}")
    processed_header = pd.read_csv(processed_path, nrows=0, encoding="utf-8")
    missing_processed = PROCESSED_REQUIRED_COLUMNS.difference(processed_header.columns)
    if missing_processed:
        raise ValueError(f"Processed {task} dataset is missing columns: {sorted(missing_processed)}")

    split_dir = _resolve(config["data"]["output_dir"])
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        path = split_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {task} {split} split: {path}")
        frame = pd.read_csv(path, usecols=["id", "clean_text", "task_label"], encoding="utf-8")
        if frame.empty or frame.isna().any().any():
            raise ValueError(f"{task} {split} is empty or contains missing required values.")
        frames[split] = frame

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for column in ("id", "clean_text"):
            overlap = set(frames[left][column].astype(str)).intersection(frames[right][column].astype(str))
            if overlap:
                raise ValueError(f"{task} {column} overlap between {left} and {right}: {len(overlap)}")


def validate_pipeline(config_path: str | Path) -> None:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    validate_data_assets(config)
    paths = resolve_task_paths(config_file)
    sample = preprocess_text("Assalam Alikum @user https://example.com #Urdu 123!", config["preprocessing"])
    if not sample:
        raise ValueError("Sample preprocessing unexpectedly produced empty text.")
    print(f"[SUCCESS] {paths['task']} pipeline validation passed: {paths['split_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    validate_pipeline(args.config)


if __name__ == "__main__":
    main()
