"""Regression checks for submission-facing documentation."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_documentation_is_self_contained() -> None:
    for path in PROJECT_ROOT.rglob("*.md"):
        assert "Assignment#0" not in path.read_text(encoding="utf-8"), path


def test_data_and_prediction_schemas_are_documented_accurately() -> None:
    raw_readme = read("data/raw/README.md")
    assert "`Id`, `Text`, `Emotions`, `Category`" in raw_readme

    processed_readme = read("data/processed/README.md")
    assert "token" in processed_readme.lower()

    prediction_readme = read("outputs/predictions/README.md")
    for field in ("`raw_text`", "`is_correct`", "`text_length`"):
        assert field in prediction_readme
    assert "- `text`:" not in prediction_readme
    assert "- `correct`:" not in prediction_readme


def test_upload_manifest_covers_professor_deliverables() -> None:
    manifest = read("docs/submission_manifest.md")
    for item in (
        "Complete source code",
        "Data processing pipeline",
        "Model training scripts",
        "Evaluation scripts",
        "Visualizations",
        "Documentation",
        "Final report",
        "Demonstration",
        "repository URL",
    ):
        assert item.lower() in manifest.lower()


def test_current_submission_materials_describe_the_official_dual_task_run() -> None:
    current_files = [
        PROJECT_ROOT / "README.md",
        *PROJECT_ROOT.joinpath("docs").glob("*.md"),
        *PROJECT_ROOT.joinpath("notebooks").glob("*.ipynb"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in current_files)

    assert "517,966" not in combined
    assert "0.5040" not in combined
    assert "seven-model" not in combined.lower()
    assert "sentiment-only" not in combined.lower()
    assert "36 official runs" in combined
    assert "validation macro-f1" in combined.lower()
    assert "weak" in combined.lower()

    readme = read("README.md")
    for required in (
        "config_sentiment.yaml",
        "config_emotion.yaml",
        "0.4590",
        "0.2854",
        "data/splits/sentiment",
        "data/splits/emotion",
    ):
        assert required in readme
