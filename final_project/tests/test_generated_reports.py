from __future__ import annotations

import json
from pathlib import Path

from src.generate_reports import build_report_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_report_bundle_matches_official_aggregate_artifacts() -> None:
    bundle = build_report_bundle(PROJECT_ROOT)
    sentiment = json.loads(
        (PROJECT_ROOT / "outputs/sentiment/results/aggregate_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    emotion = json.loads(
        (PROJECT_ROOT / "outputs/emotion/results/aggregate_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    report = bundle["final_report.md"]
    assert "## Abstract" in report
    assert "## Related Work" in report
    assert "## Proposed Methodology" in report
    assert "## Dataset and Experimental Setup" in report
    assert "## Results and Discussion" in report
    assert "## Conclusion" in report
    assert "## References" in report
    assert f"{sentiment['selected_model']['test_macro_f1_mean']:.4f}" in report
    assert f"{emotion['selected_model']['test_macro_f1_mean']:.4f}" in report
    assert "No human-gold evaluation is claimed" in report
    assert "leakage-free" not in report.lower()


def test_report_bundle_covers_both_tasks_and_all_models() -> None:
    bundle = build_report_bundle(PROJECT_ROOT)
    combined = "\n".join(bundle.values())

    for task in ("Sentiment", "Emotion"):
        assert task in combined
    for model in (
        "linear_svm",
        "logistic_regression",
        "multinomial_nb",
        "text_cnn",
        "bilstm_attention",
        "mbert",
        "xlm_roberta",
        "urdu_roberta",
    ):
        assert model in combined


def test_experiment_manifest_records_all_completed_runs() -> None:
    bundle = build_report_bundle(PROJECT_ROOT)
    manifest = json.loads(bundle["experiment_manifest.json"])

    assert manifest["protocol"]["selection_metric"] == "validation_macro_f1_mean"
    assert manifest["protocol"]["human_gold_evaluation"] is False
    assert len(manifest["runs"]) == 36
    assert {run["task"] for run in manifest["runs"]} == {"sentiment", "emotion"}
