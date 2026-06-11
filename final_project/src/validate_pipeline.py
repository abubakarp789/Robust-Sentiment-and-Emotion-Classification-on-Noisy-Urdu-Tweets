"""Validate the final-project preprocessing and split pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from label_mapping import map_to_sentiment, normalize_label
from preprocessing import preprocess_text
from utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RAW_ROWS = 1_048_000
EXPECTED_PROCESSED_ROWS = 517_966
EXPECTED_SPLIT_ROWS = {"train": 362_576, "validation": 77_695, "test": 77_695}
RAW_REQUIRED_COLUMNS = {"Id", "Text", "Category"}
PROCESSED_REQUIRED_COLUMNS = {
    "id",
    "raw_text",
    "clean_text",
    "raw_label",
    "normalized_label",
    "task_label",
    "text_length",
}
ANNOTATION_REQUIRED_COLUMNS = {
    "id",
    "raw_text",
    "clean_text",
    "current_label",
    "manual_label",
    "annotator_notes",
}


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _csv_row_count(path: Path, usecols: list[str] | None = None) -> int:
    return sum(
        len(chunk)
        for chunk in pd.read_csv(path, usecols=usecols, encoding="utf-8", chunksize=100_000)
    )


def _warn_count(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        print(f"WARNING: {name} has {actual:,} rows; current project expectation is {expected:,}.")


def validate_data_assets(cfg: dict) -> None:
    """Validate raw, processed, annotation, and data-documentation artifacts."""
    data_cfg = cfg["data"]
    raw_path = _resolve_project_path(data_cfg["raw_dataset_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset does not exist: {raw_path}")
    raw_header = pd.read_csv(raw_path, nrows=0, encoding="utf-8")
    missing_raw = sorted(RAW_REQUIRED_COLUMNS.difference(raw_header.columns))
    if missing_raw:
        raise ValueError(f"Raw dataset is missing columns: {missing_raw}")
    raw_rows = _csv_row_count(raw_path, usecols=[data_cfg["id_column"]])
    _warn_count("Raw dataset", raw_rows, EXPECTED_RAW_ROWS)
    print(f"Raw dataset validated: {raw_rows:,} rows")

    processed_path = PROJECT_ROOT / "data" / "processed" / "processed_sentiment_dataset.csv"
    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed dataset does not exist: {processed_path}")
    processed_header = pd.read_csv(processed_path, nrows=0, encoding="utf-8")
    missing_processed = sorted(PROCESSED_REQUIRED_COLUMNS.difference(processed_header.columns))
    if missing_processed:
        raise ValueError(f"Processed dataset is missing columns: {missing_processed}")
    processed_rows = 0
    for chunk in pd.read_csv(
        processed_path,
        usecols=["clean_text", "task_label"],
        encoding="utf-8",
        chunksize=100_000,
    ):
        processed_rows += len(chunk)
        if chunk["clean_text"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError("Processed dataset contains empty clean_text values.")
        if chunk["task_label"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError("Processed dataset contains empty task_label values.")
    _warn_count("Processed dataset", processed_rows, EXPECTED_PROCESSED_ROWS)
    print(f"Processed dataset validated: {processed_rows:,} rows")

    annotation_path = PROJECT_ROOT / "data" / "annotation" / "annotation_sample.csv"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation sample does not exist: {annotation_path}")
    annotation = pd.read_csv(annotation_path, encoding="utf-8")
    missing_annotation = sorted(ANNOTATION_REQUIRED_COLUMNS.difference(annotation.columns))
    if missing_annotation:
        raise ValueError(f"Annotation sample is missing columns: {missing_annotation}")
    if not annotation["manual_label"].fillna("").astype(str).str.strip().eq("").all():
        raise ValueError("Annotation sample manual_label must remain empty before human review.")
    if not annotation["annotator_notes"].fillna("").astype(str).str.strip().eq("").all():
        raise ValueError("Annotation sample annotator_notes must remain empty before human review.")
    counts = annotation["current_label"].value_counts()
    if not {"Positive", "Negative", "Neutral"}.issubset(counts.index):
        raise ValueError("Annotation sample must include Positive, Negative, and Neutral rows.")
    if int(counts.max() - counts.min()) > 5:
        print(f"WARNING: Annotation sample is not approximately balanced: {counts.to_dict()}")
    print(f"Annotation sample validated: {len(annotation):,} rows; {counts.to_dict()}")

    readmes = [
        PROJECT_ROOT / "data" / "raw" / "README.md",
        PROJECT_ROOT / "data" / "processed" / "README.md",
        PROJECT_ROOT / "data" / "annotation" / "README.md",
        PROJECT_ROOT / "data" / "annotation" / "annotation_readme.md",
        PROJECT_ROOT / "data" / "splits" / "README.md",
    ]
    missing_readmes = [str(path) for path in readmes if not path.is_file()]
    if missing_readmes:
        raise FileNotFoundError(f"Data README files are missing: {missing_readmes}")
    print("Data folder README files validated.")


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

    validate_data_assets(cfg)

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
        _warn_count(f"{name} split", len(split_df), EXPECTED_SPLIT_ROWS[name])
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
