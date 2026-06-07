"""Validate the final-project preprocessing and split pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from label_mapping import map_to_sentiment, normalize_label
from preprocessing import preprocess_text
from utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def validate_pipeline(config_path: str | Path | None = None) -> None:
    """Run lightweight validation checks for config, data, preprocessing, and splits."""
    cfg = load_config(config_path or PROJECT_ROOT / "config.yaml")
    print("Config loaded.")

    data_cfg = cfg["data"]
    raw_dataset_path = _resolve_project_path(data_cfg["raw_dataset_path"])
    if not raw_dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_dataset_path}")
    print(f"Raw dataset path exists: {raw_dataset_path}")

    sample = pd.read_csv(raw_dataset_path, nrows=5, encoding="utf-8")
    required_columns = [
        data_cfg["id_column"],
        data_cfg["text_column"],
        data_cfg["label_column"],
    ]
    missing_columns = [col for col in required_columns if col not in sample.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {missing_columns}")
    print(f"Required columns exist: {required_columns}")

    sample_text = "Assalam Alikum 😊 @user https://example.com #Urdu 123!"
    cleaned = preprocess_text(sample_text, cfg["preprocessing"])
    if not cleaned:
        raise ValueError("Preprocessing produced empty text for the sample.")
    print(f"Sample preprocessing output: {cleaned}")

    sample_label = " Joy, Sad"
    normalized = normalize_label(sample_label)
    sentiment = map_to_sentiment(sample_label)
    if normalized is None or sentiment is None:
        raise ValueError("Label normalization failed for sample label.")
    print(f"Sample label normalization: {sample_label!r} -> {normalized} -> {sentiment}")

    split_dir = _resolve_project_path(data_cfg["output_dir"])
    split_files = {
        "train": split_dir / "train.csv",
        "validation": split_dir / "validation.csv",
        "test": split_dir / "test.csv",
    }
    for name, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} split does not exist: {path}")
        split_df = pd.read_csv(path, encoding="utf-8")
        if split_df.empty:
            raise ValueError(f"{name} split is empty: {path}")
        if split_df["clean_text"].fillna("").str.strip().eq("").any():
            raise ValueError(f"{name} split contains empty clean_text values.")
        if split_df["task_label"].fillna("").str.strip().eq("").any():
            raise ValueError(f"{name} split contains empty task_label values.")
        print(f"\n{name} split: {len(split_df):,} rows")
        print(split_df["task_label"].value_counts().sort_index().to_string())

    print("\nPipeline validation passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final-project data pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    validate_pipeline(args.config)


if __name__ == "__main__":
    main()
