"""Validate evidence for every stage and deliverable in the professor's brief."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORT = PROJECT_ROOT / "outputs" / "reports" / "final_nlp_project_report.pdf"


REQUIREMENT_GROUPS: dict[str, list[Path]] = {
    "stage_1_problem_and_proposal": [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "methodology.md",
        FINAL_REPORT,
    ],
    "stage_2_literature_and_gap": [
        FINAL_REPORT,
        PROJECT_ROOT / "docs" / "professor_requirements_alignment.md",
    ],
    "stage_3_design_and_methodology": [
        FINAL_REPORT,
        PROJECT_ROOT / "docs" / "methodology.md",
        PROJECT_ROOT / "src" / "preprocessing.py",
        PROJECT_ROOT / "src" / "label_mapping.py",
    ],
    "stage_4_implementation": [
        PROJECT_ROOT / "scripts" / "01_prepare_data.py",
        PROJECT_ROOT / "scripts" / "02_train_classical.py",
        PROJECT_ROOT / "scripts" / "03_train_neural.py",
        PROJECT_ROOT / "scripts" / "04_train_transformers.py",
        PROJECT_ROOT / "data" / "splits" / "train.csv",
        PROJECT_ROOT / "outputs" / "models" / "baseline_linear_svm.joblib",
    ],
    "stage_5_evaluation_and_analysis": [
        PROJECT_ROOT / "scripts" / "05_evaluate_models.py",
        PROJECT_ROOT / "outputs" / "report_snapshot" / "leaderboard_sentiment.csv",
        PROJECT_ROOT / "outputs" / "report_snapshot" / "leaderboard_emotion.csv",
        PROJECT_ROOT / "outputs" / "results" / "model_comparison_leaderboard.csv",
        PROJECT_ROOT / "outputs" / "error_analysis" / "baseline_error_summary.json",
        PROJECT_ROOT / "docs" / "results_analysis.md",
    ],
    "stage_6_report_and_demo": [
        FINAL_REPORT,
        PROJECT_ROOT / "app" / "streamlit_app.py",
        PROJECT_ROOT / "docs" / "demonstration_guide.md",
        PROJECT_ROOT / "docs" / "demo_script.md",
    ],
    "deliverables": [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "scripts" / "01_prepare_data.py",
        PROJECT_ROOT / "scripts" / "02_train_classical.py",
        PROJECT_ROOT / "scripts" / "05_evaluate_models.py",
        PROJECT_ROOT / "scripts" / "06_generate_visualizations.py",
        PROJECT_ROOT / "outputs" / "figures",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "docs" / "submission_manifest.md",
    ],
}

REQUIRED_REPORT_SECTIONS = (
    "Abstract",
    "Introduction",
    "Related Work",
    "Proposed Methodology",
    "Dataset and Experimental Setup",
    "Results and Discussion",
    "Conclusion",
    "References",
)


def _missing(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def validate_requirements() -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for name, paths in REQUIREMENT_GROUPS.items():
        missing = _missing(paths)
        groups[name] = {
            "ready": not missing,
            "evidence_count": len(paths),
            "missing": missing,
        }

    report_source = PROJECT_ROOT / "outputs" / "reports" / "README.md"
    report_missing: list[str] = []
    if not report_source.exists():
        report_missing.append(str(report_source))
    else:
        content = report_source.read_text(encoding="utf-8")
        report_missing.extend(section for section in REQUIRED_REPORT_SECTIONS if section not in content)

    groups["required_report_sections"] = {
        "ready": not report_missing,
        "evidence_count": len(REQUIRED_REPORT_SECTIONS),
        "missing": report_missing,
    }

    ready = all(bool(group["ready"]) for group in groups.values())
    return {
        "ready": ready,
        "rubric_components_mapped": 6,
        "rubric_marks_mapped": 50 if ready else None,
        "groups": groups,
        "note": "Coverage validation does not predict the awarded grade.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()
    result = validate_requirements()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for name, group in result["groups"].items():
            status = "READY" if group["ready"] else "MISSING"
            print(f"[{status}] {name}: {group['evidence_count']} evidence checks")
            for missing in group["missing"]:
                print(f"  - {missing}")
        print("\nProfessor brief alignment:", "READY" if result["ready"] else "INCOMPLETE")

    sys.exit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
