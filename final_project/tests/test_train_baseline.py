from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.train_baseline import (
    REQUIRED_PREDICTION_COLUMNS,
    build_prediction_frame,
    confidence_scores,
    create_classifiers,
    mark_training_only_fit,
)


def _fitted_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(["اچھا دن", "برا دن", "عام دن"], ["Positive", "Negative", "Neutral"])
    return pipeline


def test_confidence_scores_use_classifier_probabilities() -> None:
    pipeline = _fitted_pipeline()
    scores = confidence_scores(pipeline, ["اچھا دن", "برا دن"])

    assert scores.shape == (2,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_prediction_frame_has_required_schema() -> None:
    source = pd.DataFrame(
        {
            "id": [1],
            "raw_text": ["بہت اچھا"],
            "clean_text": ["بہت اچھا"],
            "task_label": ["Positive"],
            "text_length": [2],
        }
    )
    frame = build_prediction_frame(
        source,
        predicted_labels=np.array(["Positive"]),
        confidence=np.array([0.8]),
        split="validation",
        model_name="logistic_regression",
    )

    assert list(frame.columns) == list(REQUIRED_PREDICTION_COLUMNS)
    assert bool(frame.loc[0, "is_correct"])


def test_training_only_fit_audit_is_attached_to_pipeline() -> None:
    pipeline = _fitted_pipeline()
    mark_training_only_fit(pipeline, train_rows=3, text_column="clean_text")

    assert pipeline.fit_audit_["tfidf_fit_split"] == "train"
    assert pipeline.fit_audit_["train_rows"] == 3
    assert pipeline.fit_audit_["validation_rows_used_for_fit"] == 0
    assert pipeline.fit_audit_["test_rows_used_for_fit"] == 0


def test_liblinear_logistic_regression_uses_explicit_one_vs_rest() -> None:
    config = {
        "baseline_models": {
            "models": {
                "logistic_regression": {
                    "enabled": True,
                    "solver": "liblinear",
                    "max_iter": 10,
                },
                "linear_svm": {"enabled": False},
                "multinomial_nb": {"enabled": False},
            }
        }
    }

    classifier = create_classifiers(config, seed=42)["logistic_regression"]

    assert isinstance(classifier, OneVsRestClassifier)
    assert isinstance(classifier.estimator, LogisticRegression)
