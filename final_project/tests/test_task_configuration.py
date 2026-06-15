from __future__ import annotations

from pathlib import Path

from src.utils import resolve_task_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_task_configs_resolve_to_isolated_artifact_trees() -> None:
    sentiment = resolve_task_paths(PROJECT_ROOT / "config_sentiment.yaml")
    emotion = resolve_task_paths(PROJECT_ROOT / "config_emotion.yaml")

    assert sentiment["task"] == "sentiment"
    assert emotion["task"] == "emotion"

    for key in (
        "processed_dir",
        "split_dir",
        "results_dir",
        "figures_dir",
        "predictions_dir",
        "models_dir",
        "error_analysis_dir",
    ):
        assert sentiment[key] != emotion[key]
        assert "sentiment" in sentiment[key].parts
        assert "emotion" in emotion[key].parts

