from __future__ import annotations

import pytest

from src.evaluate import compute_classification_metrics


def test_metrics_on_toy_predictions() -> None:
    metrics = compute_classification_metrics(
        ["Positive", "Negative", "Neutral", "Positive"],
        ["Positive", "Negative", "Positive", "Positive"],
        labels=["Negative", "Neutral", "Positive"],
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["per_class"]["Neutral"]["recall"] == 0.0
