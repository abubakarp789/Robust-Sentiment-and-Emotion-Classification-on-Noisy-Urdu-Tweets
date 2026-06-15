from __future__ import annotations

from pathlib import Path

from src.run_experiments import build_run_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_config_isolates_task_family_model_and_seed() -> None:
    first = build_run_config(
        PROJECT_ROOT / "config_sentiment.yaml",
        family="baseline",
        model="linear_svm",
        seed=42,
    )
    second = build_run_config(
        PROJECT_ROOT / "config_emotion.yaml",
        family="neural",
        model="text_cnn",
        seed=52,
    )

    assert first["project"]["random_seed"] == 42
    assert second["project"]["random_seed"] == 52
    assert first["labels"]["task"] == "sentiment"
    assert second["labels"]["task"] == "emotion"
    assert first["outputs"]["results_dir"] != second["outputs"]["results_dir"]
    assert first["outputs"]["results_dir"].endswith(
        "outputs/sentiment/runs/baseline/linear_svm/seed_42/results"
    )
    assert second["outputs"]["models_dir"].endswith(
        "outputs/emotion/runs/neural/text_cnn/seed_52/models"
    )


def test_run_config_enables_only_requested_model() -> None:
    config = build_run_config(
        PROJECT_ROOT / "config_sentiment.yaml",
        family="transformer",
        model="urdu_roberta",
        seed=42,
    )

    enabled = {
        name
        for name, values in config["transformer_models"]["models"].items()
        if values["enabled"]
    }
    assert enabled == {"urdu_roberta"}

