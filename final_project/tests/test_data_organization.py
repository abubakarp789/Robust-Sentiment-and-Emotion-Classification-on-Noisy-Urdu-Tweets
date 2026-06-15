from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _write_test_config(root: Path, raw_path: Path) -> Path:
    config = {
        "project": {"name": "test", "random_seed": 42},
        "data": {
            "raw_dataset_path": str(raw_path),
            "text_column": "Text",
            "id_column": "Id",
            "label_column": "Category",
            "output_dir": "data/splits",
        },
        "preprocessing": {
            "normalize_unicode": True,
            "remove_urls": True,
            "remove_mentions": True,
            "clean_hashtags": True,
            "remove_emojis": True,
            "remove_numbers": False,
            "remove_punctuation": True,
            "normalize_whitespace": True,
            "min_text_length": 2,
        },
        "labels": {"task": "sentiment", "drop_unknown_labels": True},
        "split": {
            "train_size": 0.7,
            "validation_size": 0.15,
            "test_size": 0.15,
            "stratify": True,
        },
        "outputs": {"results_dir": "outputs/results"},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_create_splits_writes_full_processed_dataset(tmp_path, monkeypatch) -> None:
    import create_splits

    labels = ["Joy", "Sad", "Surprise"] * 20
    raw = pd.DataFrame(
        {
            "Id": range(1, 61),
            "Text": [f"اردو متن {index} مثال" for index in range(1, 61)],
            "Category": labels,
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    config_path = _write_test_config(tmp_path, raw_path)
    monkeypatch.setattr(create_splits, "PROJECT_ROOT", tmp_path)

    summary = create_splits.create_splits(config_path)

    processed_path = tmp_path / "data" / "processed" / "processed_sentiment_dataset.csv"
    assert processed_path.is_file()
    processed = pd.read_csv(processed_path, encoding="utf-8")
    assert list(processed.columns) == [
        "id",
        "raw_text",
        "clean_text",
        "raw_label",
        "normalized_label",
        "task_label",
        "text_length",
    ]
    assert len(processed) == summary["rows_after_filtering"]
    assert summary["processed_dataset_path"] == "data/processed/processed_sentiment_dataset.csv"
    assert summary["processed_dataset_rows"] == len(processed)

    saved_summary = json.loads(
        (tmp_path / "outputs" / "results" / "split_summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary["processed_dataset_rows"] == len(processed)


def test_annotation_sample_is_balanced_deterministic_and_unlabelled() -> None:
    from create_annotation_sample import build_annotation_sample

    test_split = pd.DataFrame(
        {
            "id": range(450),
            "raw_text": [f"raw {index}" for index in range(450)],
            "clean_text": [f"clean {index}" for index in range(450)],
            "task_label": ["Positive"] * 150 + ["Negative"] * 150 + ["Neutral"] * 150,
        }
    )

    first = build_annotation_sample(test_split, per_class=100, random_seed=42)
    second = build_annotation_sample(test_split, per_class=100, random_seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["current_label"].value_counts().to_dict() == {
        "Positive": 100,
        "Negative": 100,
        "Neutral": 100,
    }
    assert first["manual_label"].fillna("").eq("").all()
    assert first["annotator_notes"].fillna("").eq("").all()


def test_final_validator_includes_data_asset_checks() -> None:
    final_source = (SRC_DIR / "validate_final_project.py").read_text(encoding="utf-8")
    pipeline_source = (SRC_DIR / "validate_pipeline.py").read_text(encoding="utf-8")

    assert "check_data_assets" in final_source
    assert "validate_data_assets(config)" in final_source
    assert "processed_sentiment_dataset.csv" in pipeline_source
    assert "annotation_sample.csv" in pipeline_source
