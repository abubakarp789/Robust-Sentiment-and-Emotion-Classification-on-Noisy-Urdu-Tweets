from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_notebook_validator_accepts_submission_notebooks() -> None:
    result = subprocess.run(
        [sys.executable, "src/validate_notebooks.py", "--config", "config.yaml"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_final_validator_runs_notebook_validation() -> None:
    source = (PROJECT_ROOT / "src" / "validate_final_project.py").read_text(encoding="utf-8")

    assert "check_notebooks" in source
    assert "validate_notebooks" in source


def test_final_validator_keeps_deployment_compile_check() -> None:
    from src.validate_final_project import check_deployment

    assert check_deployment() is True
