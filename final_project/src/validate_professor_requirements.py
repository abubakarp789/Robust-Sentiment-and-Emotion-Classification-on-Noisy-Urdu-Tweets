"""Validate evidence coverage for the six-stage, 50-mark project brief."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "stage_1_problem_and_proposal": ["README.md", "reports/final_report.md"],
    "stage_2_literature_and_gap": ["reports/final_report.md", "docs/independent_rubric_gap_analysis_2026-06-15.md"],
    "stage_3_design_and_methodology": ["docs/methodology.md", "src/preprocessing.py", "src/create_splits.py"],
    "stage_4_implementation": ["src/run_experiments.py", "config_sentiment.yaml", "config_emotion.yaml", "reports/experiment_manifest.json"],
    "stage_5_evaluation_and_analysis": ["src/aggregate_experiments.py", "outputs/sentiment/results/aggregate_metrics.json", "outputs/emotion/results/aggregate_metrics.json", "reports/final_evaluation_summary.md"],
    "stage_6_report_and_demonstration": ["reports/final_report.pdf", "reports/final_presentation.pptx", "app/streamlit_app.py", "docs/demonstration_guide.md"],
    "deliverables": ["src", "scripts", "tests", "data", "outputs", "reports", "notebooks", "docs/submission_manifest.md"],
}


def validate_requirements() -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for name, relative_paths in GROUPS.items():
        missing = [path for path in relative_paths if not (PROJECT_ROOT / path).exists()]
        groups[name] = {"ready": not missing, "evidence_count": len(relative_paths), "missing": missing}
    ready = all(group["ready"] for group in groups.values())
    return {
        "ready": ready,
        "rubric_components_mapped": 6,
        "rubric_marks_mapped": 50 if ready else None,
        "groups": groups,
        "note": "Evidence coverage does not predict or guarantee the awarded grade.",
    }


def main() -> None:
    result = validate_requirements()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
