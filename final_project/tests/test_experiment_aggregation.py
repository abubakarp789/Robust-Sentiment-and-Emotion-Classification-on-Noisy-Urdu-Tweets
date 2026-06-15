from __future__ import annotations

import pytest

from src.aggregate_experiments import summarize_records


def _record(model: str, family: str, seed: int, validation: float, test: float) -> dict:
    return {
        "task": "sentiment",
        "family": family,
        "model": model,
        "seed": seed,
        "validation": {"accuracy": 0.8, "macro_f1": validation, "weighted_f1": 0.82},
        "test": {"accuracy": 0.79, "macro_f1": test, "weighted_f1": 0.81},
    }


def test_summarize_records_ranks_by_mean_validation_macro_f1() -> None:
    records = [
        _record("linear_svm", "baseline", 42, 0.40, 0.80),
        _record("linear_svm", "baseline", 52, 0.42, 0.79),
        _record("text_cnn", "neural", 42, 0.50, 0.45),
        _record("text_cnn", "neural", 52, 0.52, 0.44),
    ]

    leaderboard, summary = summarize_records(records)

    assert leaderboard.iloc[0]["model_name"] == "text_cnn"
    assert summary["selected_model"]["model_name"] == "text_cnn"
    assert leaderboard.iloc[0]["validation_macro_f1_mean"] == pytest.approx(0.51)
    assert leaderboard.iloc[0]["test_macro_f1_mean"] == pytest.approx(0.445)


def test_summarize_records_uses_sample_standard_deviation() -> None:
    records = [
        _record("linear_svm", "baseline", 42, 0.40, 0.50),
        _record("linear_svm", "baseline", 52, 0.60, 0.70),
    ]

    leaderboard, _ = summarize_records(records)

    assert leaderboard.iloc[0]["validation_macro_f1_std"] == pytest.approx(0.1414213562)
    assert leaderboard.iloc[0]["test_macro_f1_std"] == pytest.approx(0.1414213562)


def test_summarize_records_selects_canonical_seed_by_validation() -> None:
    records = [
        _record("linear_svm", "baseline", 42, 0.50, 0.55),
        _record("linear_svm", "baseline", 52, 0.60, 0.40),
        _record("linear_svm", "baseline", 62, 0.55, 0.90),
    ]

    _, summary = summarize_records(records)

    assert summary["selected_model"]["canonical_seed"] == 52

