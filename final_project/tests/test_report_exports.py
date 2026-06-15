from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_report_pdf_is_present_and_nontrivial() -> None:
    path = PROJECT_ROOT / "reports" / "final_report.pdf"
    assert path.read_bytes().startswith(b"%PDF")
    assert path.stat().st_size > 100_000


def test_final_presentation_contains_at_least_ten_slides() -> None:
    path = PROJECT_ROOT / "reports" / "final_presentation.pptx"
    assert path.stat().st_size > 100_000
    with ZipFile(path) as archive:
        slides = [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
    assert len(slides) >= 10
