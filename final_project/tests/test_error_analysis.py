from __future__ import annotations

import json

import pandas as pd
import pytest

from src.error_analysis import (
    REQUIRED_PREDICTION_COLUMNS,
    add_error_categories,
    categorize_error_type,
    compute_error_summary,
    export_error_report,
    get_misclassified_examples,
    load_predictions,
    sample_errors_by_class,
    sample_errors_by_confusion_pair,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["1", "اچھا", "اچھا", "Positive", "Positive", 0.90, "test", "linear_svm", True, 1],
            ["2", "برا متن", "برا متن", "Negative", "Positive", 0.85, "test", "linear_svm", False, 2],
            ["3", "عام متن", "عام متن", "Neutral", "Positive", 0.75, "test", "linear_svm", False, 2],
            ["4", "بہت لمبا متن", "بہت لمبا متن", "Positive", "Negative", 0.60, "test", "linear_svm", False, 40],
        ],
        columns=REQUIRED_PREDICTION_COLUMNS,
    )


def test_load_predictions_validates_required_schema(tmp_path) -> None:
    valid_path = tmp_path / "valid.csv"
    _prediction_frame().to_csv(valid_path, index=False, encoding="utf-8")
    loaded = load_predictions(valid_path)
    assert len(loaded) == 4
    assert loaded.loc[0, "raw_text"] == "اچھا"

    invalid_path = tmp_path / "invalid.csv"
    pd.DataFrame({"true_label": ["Positive"]}).to_csv(invalid_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_predictions(invalid_path)


def test_misclassified_sampling_and_summary_are_deterministic() -> None:
    errors = get_misclassified_examples(_prediction_frame())
    summary = compute_error_summary(_prediction_frame())
    by_class = sample_errors_by_class(errors, n=1)
    by_pair = sample_errors_by_confusion_pair(errors, n=1)

    assert len(errors) == 3
    assert summary["total_rows"] == 4
    assert summary["total_errors"] == 3
    assert summary["error_rate"] == 0.75
    assert summary["class_errors"]["Negative"]["errors"] == 1
    assert summary["confusion_pairs"][0]["count"] == 1
    assert len(by_class) == 3
    assert len(by_pair) == 3


def test_error_categories_apply_specific_rules_before_generic_pairs() -> None:
    frame = _prediction_frame()
    assert categorize_error_type(frame.iloc[2]) == "neutral_to_positive"
    assert categorize_error_type(frame.iloc[1]) == "short_text_ambiguity"
    assert categorize_error_type(frame.iloc[3]) == "long_text_noise"

    categorized = add_error_categories(frame)
    assert "error_category" in categorized.columns
    assert categorized["is_correct"].eq(False).all()


def test_export_error_report_writes_utf8_tables_and_summary(tmp_path) -> None:
    outputs = export_error_report(
        _prediction_frame(), tmp_path, model_name="linear_svm", split="test"
    )

    assert set(outputs) == {"misclassified", "class_errors", "confusion_pairs", "summary"}
    assert pd.read_csv(outputs["misclassified"], encoding="utf-8").shape[0] == 3
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["model_name"] == "linear_svm"
    assert summary["split"] == "test"
