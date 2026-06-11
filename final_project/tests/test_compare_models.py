from __future__ import annotations

from src.compare_models import build_comparison_rows


def _metrics(model_name: str, macro_f1: float) -> dict:
    values = {
        "accuracy": 0.8,
        "macro_f1": macro_f1,
        "weighted_f1": 0.82,
        "per_class": {
            "Negative": {"f1": 0.4},
            "Neutral": {"f1": 0.2},
            "Positive": {"f1": 0.9},
        },
    }
    return {
        "models": {
            model_name: {
                "validation": {**values, "macro_f1": macro_f1 - 0.01},
                "test": values,
            }
        }
    }


def test_build_comparison_rows_combines_families_and_sorts_test_macro_f1() -> None:
    rows = build_comparison_rows(
        _metrics("linear_svm", 0.50), _metrics("text_cnn", 0.55)
    )

    assert rows[0]["model_family"] == "neural"
    assert rows[0]["model_name"] == "text_cnn"
    assert rows[0]["test_macro_f1"] == 0.55
    assert rows[0]["neutral_f1"] == 0.2
    assert rows[1]["model_family"] == "baseline"


def test_beats_linear_svm_is_correct() -> None:
    rows = build_comparison_rows(
        _metrics("linear_svm", 0.50), _metrics("text_cnn", 0.55), _metrics("mbert", 0.45)
    )
    for r in rows:
        if r["model_name"] == "text_cnn":
            assert r["beats_linear_svm"] is True
        elif r["model_name"] == "mbert":
            assert r["beats_linear_svm"] is False
        elif r["model_name"] == "linear_svm":
            assert r["beats_linear_svm"] is False

