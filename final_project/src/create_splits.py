"""Create reproducible train/validation/test splits for the final project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .label_mapping import (
        normalize_label,
        normalize_task_label,
        save_label_mapping_summary,
    )
    from .preprocessing import is_valid_clean_text, preprocess_series, token_count
    from .utils import load_config, set_seed
except ImportError:  # Support direct execution: python src/create_splits.py
    from label_mapping import (
        normalize_label,
        normalize_task_label,
        save_label_mapping_summary,
    )
    from preprocessing import is_valid_clean_text, preprocess_series, token_count
    from utils import load_config, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _read_dataset(path: Path, required_columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=required_columns, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, usecols=required_columns, encoding="utf-8-sig")


def _distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def build_group_safe_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Group rows linked by ID or normalized text and remove unsafe duplicates."""
    frame = df.reset_index(drop=True).copy()
    row_count = len(frame)
    parent = list(range(row_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_id: dict[str, int] = {}
    first_text: dict[str, int] = {}
    for index, (row_id, text) in enumerate(zip(frame["id"], frame["clean_text"])):
        id_key = "" if pd.isna(row_id) else str(row_id).strip()
        text_key = str(text)
        if id_key:
            if id_key in first_id:
                union(index, first_id[id_key])
            else:
                first_id[id_key] = index
        if text_key in first_text:
            union(index, first_text[text_key])
        else:
            first_text[text_key] = index

    roots = [find(index) for index in range(row_count)]
    root_to_group = {root: group for group, root in enumerate(dict.fromkeys(roots))}
    frame["group_id"] = [root_to_group[root] for root in roots]
    conflicting = set(
        frame.groupby("group_id")["task_label"].nunique().loc[lambda values: values > 1].index
    )
    conflicting_mask = frame["group_id"].isin(conflicting)
    rows_removed_in_conflicting_groups = int(conflicting_mask.sum())
    frame = frame.loc[~conflicting_mask].copy()

    rows_before_deduplication = len(frame)
    frame = frame.drop_duplicates(subset=["clean_text"], keep="first").reset_index(drop=True)
    rows_removed_as_duplicate = rows_before_deduplication - len(frame)
    return frame, {
        "duplicate_groups": int(len(root_to_group)),
        "conflicting_groups_removed": int(len(conflicting)),
        "rows_removed_in_conflicting_groups": rows_removed_in_conflicting_groups,
        "rows_removed_as_duplicate": int(rows_removed_as_duplicate),
    }


def split_group_safe(
    df: pd.DataFrame,
    *,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_seed: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split complete duplicate groups while preserving task-label distributions."""

    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0; got {total}")

    groups = df.groupby("group_id", as_index=False).agg(task_label=("task_label", "first"))
    group_stratify = groups["task_label"] if stratify else None
    train_groups, temp_groups = train_test_split(
        groups,
        train_size=train_size,
        random_state=random_seed,
        stratify=group_stratify,
    )

    temp_stratify = temp_groups["task_label"] if stratify else None
    validation_fraction_of_temp = validation_size / (validation_size + test_size)
    validation_groups, test_groups = train_test_split(
        temp_groups,
        train_size=validation_fraction_of_temp,
        random_state=random_seed,
        stratify=temp_stratify,
    )

    def select(group_frame: pd.DataFrame) -> pd.DataFrame:
        selected = df[df["group_id"].isin(group_frame["group_id"])].copy()
        return selected.reset_index(drop=True)

    return select(train_groups), select(validation_groups), select(test_groups)


def _split_data(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg["split"]
    return split_group_safe(
        df,
        train_size=float(split_cfg["train_size"]),
        validation_size=float(split_cfg["validation_size"]),
        test_size=float(split_cfg["test_size"]),
        random_seed=int(cfg["project"]["random_seed"]),
        stratify=bool(split_cfg.get("stratify", True)),
    )


def create_splits(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load data, preprocess text, normalize labels, and save split CSV files."""
    cfg = load_config(config_path or PROJECT_ROOT / "config.yaml")
    set_seed(int(cfg["project"]["random_seed"]))

    data_cfg = cfg["data"]
    labels_cfg = cfg["labels"]
    preprocessing_cfg = cfg["preprocessing"]
    outputs_cfg = cfg["outputs"]

    raw_dataset_path = _resolve_project_path(data_cfg["raw_dataset_path"])
    split_dir = _resolve_project_path(data_cfg["output_dir"])
    task = labels_cfg.get("task", "sentiment")
    processed_dir = _resolve_project_path(data_cfg.get("processed_dir", "data/processed"))
    processed_name = data_cfg.get("processed_filename", f"processed_{task}_dataset.csv")
    processed_path = processed_dir / processed_name
    results_dir = _resolve_project_path(outputs_cfg["results_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    id_col = data_cfg["id_column"]
    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]

    df = _read_dataset(raw_dataset_path, [id_col, text_col, label_col])
    total_rows = int(len(df))
    missing_raw_labels = int(df[label_col].isna().sum())

    save_label_mapping_summary(
        df,
        label_column=label_col,
        task=task,
        output_path=results_dir / "label_mapping_summary.json",
    )

    df = df.rename(columns={id_col: "id", text_col: "raw_text", label_col: "raw_label"})
    df["normalized_label"] = df["raw_label"].map(normalize_label)
    df["task_label"] = df["raw_label"].map(lambda value: normalize_task_label(value, task))

    missing_required_labels = int(df["task_label"].isna().sum())
    if labels_cfg.get("drop_unknown_labels", True):
        df = df[df["task_label"].notna()].copy()

    df["clean_text"] = preprocess_series(df["raw_text"], preprocessing_cfg)
    df["text_length"] = df["clean_text"].map(token_count)

    min_text_length = int(preprocessing_cfg.get("min_text_length", 1))
    rows_before_empty_filter = int(len(df))
    df = df[df["clean_text"].map(lambda value: is_valid_clean_text(value, min_text_length))].copy()
    rows_removed_empty_or_short = rows_before_empty_filter - int(len(df))

    output_columns = [
        "id",
        "raw_text",
        "clean_text",
        "raw_label",
        "normalized_label",
        "task_label",
        "text_length",
    ]
    df = df[output_columns].reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows remain after label and text filtering.")

    df, group_summary = build_group_safe_frame(df)
    df[output_columns].to_csv(processed_path, index=False, encoding="utf-8")

    train_df, validation_df, test_df = _split_data(df, cfg)

    train_df[output_columns].to_csv(split_dir / "train.csv", index=False, encoding="utf-8")
    validation_df[output_columns].to_csv(split_dir / "validation.csv", index=False, encoding="utf-8")
    test_df[output_columns].to_csv(split_dir / "test.csv", index=False, encoding="utf-8")

    summary = {
        "total_rows_loaded": total_rows,
        "rows_with_missing_raw_labels": missing_raw_labels,
        "rows_with_missing_required_labels": missing_required_labels,
        "rows_removed_empty_or_short_clean_text": rows_removed_empty_or_short,
        "rows_after_filtering": int(len(df)),
        "processed_dataset_path": str(processed_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "processed_dataset_rows": int(len(df)),
        "number_of_classes": int(df["task_label"].nunique()),
        "class_distribution_before_split": _distribution(df["task_label"]),
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "train_class_distribution": _distribution(train_df["task_label"]),
        "validation_class_distribution": _distribution(validation_df["task_label"]),
        "test_class_distribution": _distribution(test_df["task_label"]),
        "random_seed": int(cfg["project"]["random_seed"]),
        "split_ratios": {
            "train_size": float(cfg["split"]["train_size"]),
            "validation_size": float(cfg["split"]["validation_size"]),
            "test_size": float(cfg["split"]["test_size"]),
        },
        "task": task,
        "min_text_length_tokens": min_text_length,
        "group_safety": group_summary,
    }

    (results_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final-project dataset splits.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    create_splits(args.config)


if __name__ == "__main__":
    main()
