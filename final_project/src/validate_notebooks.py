"""Validate the final-project analysis notebooks and their artifact references."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


EXPECTED_NOTEBOOKS = {
    "01_dataset_analysis.ipynb": {
        "title": "# 01 Dataset Analysis and Preprocessing",
        "artifacts": [
            "outputs/sentiment/results/split_summary.json",
            "outputs/emotion/results/split_summary.json",
            "data/splits/sentiment/train.csv",
            "data/splits/sentiment/validation.csv",
            "data/splits/sentiment/test.csv",
            "data/splits/emotion/train.csv",
            "data/splits/emotion/validation.csv",
            "data/splits/emotion/test.csv",
        ],
    },
    "02_baseline_models.ipynb": {
        "title": "# 02 Baseline Statistical Models",
        "artifacts": [
            "outputs/sentiment/results/aggregate_metrics.json",
            "outputs/emotion/results/aggregate_metrics.json",
            "outputs/sentiment/results/model_comparison_leaderboard.csv",
            "outputs/emotion/results/model_comparison_leaderboard.csv",
        ],
    },
    "03_neural_models.ipynb": {
        "title": "# 03 Neural Models",
        "artifacts": [
            "outputs/sentiment/results/aggregate_metrics.json",
            "outputs/emotion/results/aggregate_metrics.json",
            "outputs/sentiment/runs/neural/text_cnn/seed_42/results/neural_metrics.json",
            "outputs/emotion/runs/neural/bilstm_attention/seed_42/results/neural_metrics.json",
        ],
    },
    "04_transformer_models.ipynb": {
        "title": "# 04 Transformer Models and Explanation Assistant",
        "artifacts": [
            "outputs/sentiment/results/model_comparison_leaderboard.csv",
            "outputs/emotion/results/model_comparison_leaderboard.csv",
            "outputs/sentiment/runs/transformer/urdu_roberta/seed_42/results/transformer_metrics.json",
            "outputs/emotion/runs/transformer/urdu_roberta/seed_42/results/transformer_metrics.json",
        ],
    },
    "05_error_analysis.ipynb": {
        "title": "# 05 Error Analysis",
        "artifacts": [
            "outputs/sentiment/runs/baseline/linear_svm/seed_42/predictions/baseline_linear_svm_test_predictions.csv",
            "outputs/emotion/runs/baseline/linear_svm/seed_42/predictions/baseline_linear_svm_test_predictions.csv",
            "outputs/sentiment/runs/baseline/linear_svm/seed_42/results/confusion_matrix_baseline_linear_svm_test.csv",
            "outputs/emotion/runs/baseline/linear_svm/seed_42/results/confusion_matrix_baseline_linear_svm_test.csv",
        ],
    },
}

PLACEHOLDER_RE = re.compile(r"\b(?:placeholder|todo|tbd|dummy|coming\s+soon)\b", re.IGNORECASE)
TRAINING_RE = re.compile(
    r"(?:\.fit\s*\(|train_baseline\.py|train_neural\.py|train_transformer\.py|create_splits\.py)",
    re.IGNORECASE,
)


def cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def validate_notebook(path: Path, expected: dict[str, Any], project_root: Path) -> list[str]:
    """Return validation errors for one notebook."""
    errors: list[str] = []
    if not path.is_file():
        return [f"Notebook is missing: {path}"]

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"Notebook is not valid UTF-8 JSON: {path.name}: {exc}"]

    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        errors.append(f"{path.name}: expected nbformat 4 with a cells list")
        return errors

    cells = notebook["cells"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if len(markdown_cells) < 5:
        errors.append(f"{path.name}: needs at least 5 markdown cells; found {len(markdown_cells)}")
    if len(code_cells) < 3:
        errors.append(f"{path.name}: needs at least 3 code cells; found {len(code_cells)}")

    first_markdown = cell_text(markdown_cells[0]).strip() if markdown_cells else ""
    first_line = first_markdown.splitlines()[0] if first_markdown else ""
    if first_line != expected["title"]:
        errors.append(
            f"{path.name}: title must be '{expected['title']}', found '{first_line or '[missing]'}'"
        )

    complete_text = "\n".join(cell_text(cell) for cell in cells)
    placeholder = PLACEHOLDER_RE.search(complete_text)
    if placeholder:
        errors.append(f"{path.name}: contains prohibited placeholder term '{placeholder.group(0)}'")
    if not any(cell_text(cell).strip() for cell in code_cells):
        errors.append(f"{path.name}: contains no meaningful code")
    training_call = TRAINING_RE.search(complete_text)
    if training_call:
        errors.append(f"{path.name}: contains a training or split-generation call '{training_call.group(0)}'")

    for relative_path in expected["artifacts"]:
        artifact_path = project_root / relative_path
        if not artifact_path.is_file():
            errors.append(f"{path.name}: required artifact is missing: {relative_path}")

        if relative_path not in complete_text:
            errors.append(f"{path.name}: does not reference required artifact: {relative_path}")

    return errors


def validate_notebooks(config_path: str | Path, verbose: bool = True) -> bool:
    """Validate every required notebook relative to the supplied project config."""
    resolved_config = Path(config_path).resolve()
    if not resolved_config.is_file():
        if verbose:
            print(f"[FAILURE] Config file is missing: {resolved_config}")
        return False

    project_root = resolved_config.parent
    notebooks_dir = project_root / "notebooks"
    all_errors: list[str] = []

    for filename, expected in EXPECTED_NOTEBOOKS.items():
        errors = validate_notebook(notebooks_dir / filename, expected, project_root)
        if errors:
            all_errors.extend(errors)
        elif verbose:
            print(f"[SUCCESS] {filename} is complete and references existing artifacts.")

    if verbose:
        for error in all_errors:
            print(f"[FAILURE] {error}")
        if not all_errors:
            print("[SUCCESS] ALL NOTEBOOK VALIDATIONS PASSED SUCCESSFULLY!")

    return not all_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final-project analysis notebooks.")
    parser.add_argument("--config", default="config.yaml", help="Path to the project config file")
    args = parser.parse_args()
    sys.exit(0 if validate_notebooks(args.config) else 1)


if __name__ == "__main__":
    main()
