from __future__ import annotations

import numpy as np
import pytest
import torch
import pandas as pd

from src.explanation_assistant import (
    explain_prediction,
    explain_error,
    generate_model_summary,
    generate_final_insight,
)


from src.train_transformer import compute_class_weights_with_smoothing


def test_class_weights_with_smoothing() -> None:
    # 3 classes: class 0 is majority, class 1 is minority (rare)
    labels = [0] * 80 + [1] * 5 + [2] * 15
    
    # Without smoothing (standard balanced weights)
    weights_raw = compute_class_weights_with_smoothing(labels, num_classes=3, smoothing=0.0)
    # class 1 weight should be much larger than class 0 weight
    assert weights_raw[1] > weights_raw[0]
    assert pytest.approx(float(weights_raw[1] / weights_raw[0]), 0.01) == 16.0
    
    # With smoothing = 0.5
    weights_smoothed = compute_class_weights_with_smoothing(labels, num_classes=3, smoothing=0.5)
    # The ratio of class 1 weight to class 0 weight should be significantly reduced (less extreme)
    assert weights_smoothed[1] > weights_smoothed[0]
    assert float(weights_smoothed[1] / weights_smoothed[0]) < 5.0


def test_explanation_assistant_predicts_correctly() -> None:
    text = "یہ ایک بہت ہی اچھا ٹویٹ ہے۔"
    explanation = explain_prediction(
        text=text,
        true_label="Positive",
        predicted_label="Positive",
        confidence=0.85,
        model_name="mbert"
    )
    assert isinstance(explanation, str)
    assert "Positive" in explanation
    assert "0.85" in explanation


def test_explanation_assistant_explains_error() -> None:
    row = {
        "clean_text": "میں بہت مایوس ہوں۔",
        "true_label": "Negative",
        "predicted_label": "Positive",
        "confidence": 0.90,
        "text_length": 5,
        "is_correct": False,
        "is_high_confidence_error": True,
        "is_short_text_error": False,
        "is_minority_class_error": False,
    }
    explanation = explain_error(row)
    assert isinstance(explanation, str)
    assert "Negative" in explanation
    assert "Positive" in explanation
    assert "high-confidence" in explanation.lower()


def test_explanation_assistant_model_summary() -> None:
    metrics = {
        "validation": {"macro_f1": 0.52, "accuracy": 0.84},
        "test": {"macro_f1": 0.51, "accuracy": 0.83, "per_class": {"Neutral": {"f1": 0.15}}},
    }
    summary = generate_model_summary("mbert", metrics)
    assert isinstance(summary, str)
    assert "mbert" in summary
    assert "0.51" in summary


def test_explanation_assistant_final_insight() -> None:
    df = pd.DataFrame([
        {"model_name": "linear_svm", "test_macro_f1": 0.504, "neutral_f1": 0.13},
        {"model_name": "mbert", "test_macro_f1": 0.521, "neutral_f1": 0.18},
    ])
    insight = generate_final_insight(df)
    assert isinstance(insight, str)
    assert "linear_svm" in insight
