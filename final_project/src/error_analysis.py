"""Error-analysis stubs for the final NLP semester project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def save_misclassified_examples(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """Save rows where the predicted label differs from the true label."""
    raise NotImplementedError("Misclassified-example export is not implemented yet.")


def generate_error_summary(errors: pd.DataFrame) -> Dict[str, Any]:
    """Generate aggregate statistics over a misclassification table."""
    raise NotImplementedError("Error summary generation is not implemented yet.")


def categorize_error_type(row: pd.Series) -> str:
    """Assign a human-readable error category to one misclassified example."""
    raise NotImplementedError("Error categorization is not implemented yet.")


def export_error_report(
    errors: pd.DataFrame,
    summary: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """Export an error-analysis report as Markdown, CSV, or JSON."""
    raise NotImplementedError("Error report export is not implemented yet.")
