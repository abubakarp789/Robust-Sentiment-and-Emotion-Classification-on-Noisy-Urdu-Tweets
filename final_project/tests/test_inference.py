from __future__ import annotations

import json

import numpy as np
import torch

from src.inference import InferenceModel, resolve_model_run
from src.models_dl import TextCNN
from src.neural_utils import PAD_TOKEN, UNK_TOKEN


class MarginClassifier:
    classes_ = np.asarray(["Negative", "Positive"])

    def predict(self, texts):
        return np.asarray(["Positive"] * len(texts))

    def decision_function(self, texts):
        return np.asarray([[0.2, 1.2]] * len(texts))


def test_svm_margin_is_labeled_decision_score() -> None:
    result = InferenceModel("baseline", MarginClassifier()).predict("sample text")

    assert result["predicted_label"] == "Positive"
    assert result["score_kind"] == "decision_score"
    assert 0.0 <= result["score"] <= 1.0


def test_neural_inference_uses_vocab_and_label_mapping() -> None:
    model = TextCNN(
        vocab_size=4,
        embedding_dim=4,
        num_classes=2,
        padding_idx=0,
        num_filters=2,
        kernel_sizes=[2],
        dropout=0.0,
    )
    wrapper = InferenceModel(
        "neural",
        model,
        label_mapping={0: "Negative", 1: "Positive"},
        vocab={PAD_TOKEN: 0, UNK_TOKEN: 1, "good": 2, "bad": 3},
        max_sequence_length=4,
        device=torch.device("cpu"),
    )

    result = wrapper.predict("good text")

    assert result["predicted_label"] in {"Negative", "Positive"}
    assert result["score_kind"] == "probability"
    assert len(result["probabilities"]) == 2


def test_resolve_model_run_chooses_highest_validation_seed(tmp_path) -> None:
    for seed, score in ((42, 0.4), (52, 0.6), (62, 0.5)):
        result_dir = (
            tmp_path
            / "outputs"
            / "sentiment"
            / "runs"
            / "baseline"
            / "linear_svm"
            / f"seed_{seed}"
            / "results"
        )
        result_dir.mkdir(parents=True)
        payload = {
            "models": {
                "linear_svm": {
                    "validation": {"macro_f1": score, "accuracy": 0.8},
                    "test": {"macro_f1": 0.1, "accuracy": 0.8},
                }
            }
        }
        (result_dir / "baseline_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    run = resolve_model_run(tmp_path, "sentiment", "linear_svm")

    assert run["family"] == "baseline"
    assert run["seed"] == 52

