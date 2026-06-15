from __future__ import annotations

from pathlib import Path

from src.validate_official_benchmark import validate_official_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_official_benchmark_validator_accepts_complete_dual_task_run() -> None:
    result = validate_official_benchmark(PROJECT_ROOT)

    assert result["valid"] is True
    assert result["run_count"] == 36
    assert result["tasks"] == ["sentiment", "emotion"]
    assert result["overlap_checks"] == 12
    assert result["selected_models"] == {
        "sentiment": "linear_svm",
        "emotion": "linear_svm",
    }
    assert result["errors"] == []
