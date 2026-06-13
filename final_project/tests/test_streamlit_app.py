from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.inference import InferenceModel

try:
    from app import streamlit_app
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise
    streamlit_app = None


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_uses_unicode_safe_trimmed_input_boundary() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert "clean_input = str(user_text).strip()" in source
    assert "if not clean_input:" in source


def test_app_rejects_text_removed_by_preprocessing() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert "The input became empty after preprocessing. Please enter a longer Urdu tweet." in source


def test_app_uses_current_streamlit_width_api() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert "use_container_width" not in source


def test_multinomial_nb_inference_uses_probability_output() -> None:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )
    pipeline.fit(
        ["اچھا دن", "برا دن", "عام دن"],
        ["Positive", "Negative", "Neutral"],
    )

    result = InferenceModel("baseline", pipeline).predict("اچھا دن")

    assert result["predicted_label"] in {"Negative", "Neutral", "Positive"}
    assert len(result["probabilities"]) == 3
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.skipif(streamlit_app is None, reason="streamlit is not installed")
def test_model_availability_requires_real_checkpoint_files(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "baseline_linear_svm.joblib").write_bytes(b"saved")
    transformer_dir = models_dir / "transformer_mbert" / "best"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")

    assert streamlit_app.model_artifact_available("linear_svm", models_dir)
    assert not streamlit_app.model_artifact_available("mbert", models_dir)

    (transformer_dir / "model.safetensors").write_bytes(b"saved")
    assert streamlit_app.model_artifact_available("mbert", models_dir)


@pytest.mark.skipif(streamlit_app is None, reason="streamlit is not installed")
def test_selected_model_failure_falls_back_to_linear_svm(monkeypatch) -> None:
    fallback_model = object()

    def fake_cached_model(model_key: str, project_root: str):
        if model_key == "mbert":
            raise OSError("checkpoint is incomplete")
        assert model_key == "linear_svm"
        return fallback_model

    monkeypatch.setattr(streamlit_app, "cached_model", fake_cached_model)

    model, actual_key, warning = streamlit_app.load_selected_model("mbert")

    assert model is fallback_model
    assert actual_key == "linear_svm"
    assert warning is not None and "Linear SVM was used instead" in warning
