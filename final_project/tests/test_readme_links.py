"""Test suite to run markdown link validation."""

import sys
from pathlib import Path

# Ensure src is in the system path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from validate_readme_links import scan_project


def test_readme_links():
    """Verify that there are no broken local relative links in the workspace Markdown files."""
    errors = scan_project()
    assert not errors, f"Broken relative links detected:\n{errors}"
