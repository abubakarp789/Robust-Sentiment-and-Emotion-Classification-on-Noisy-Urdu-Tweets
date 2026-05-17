"""Compile the Markdown technical report into LaTeX, DOCX, and PDF.

Workflow:

1. Read `report/Assignment3_TechnicalReport.md`.
2. Substitute the two `<!-- BEGIN_*_LEADERBOARD -->` placeholders with the
   contents of `outputs/results/leaderboard_{sentiment,emotion}.csv`
   (produced by `03_evaluation.ipynb`).
3. Call `pandoc` to emit `.tex`, `.docx`, and `.pdf` next to the source.

Requirements (run once):

    # Markdown → DOCX / TeX: pandoc
    winget install --id JohnMacFarlane.Pandoc -e
    # TeX → PDF (only needed if you want a PDF from pandoc directly):
    winget install --id MiKTeX.MiKTeX -e

You can run the script even without pandoc — in that case it will produce
the substituted Markdown and a stand-alone LaTeX file via a built-in
exporter, and skip the PDF/DOCX outputs with a friendly warning.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "outputs" / "results"
MASTER = HERE / "Assignment3_TechnicalReport.md"
OUT_FILLED = HERE / "Assignment3_TechnicalReport.filled.md"


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in row.tolist():
            if isinstance(c, float):
                cells.append(f"{c:.4f}")
            else:
                cells.append(str(c))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def substitute_leaderboards(src: str) -> str:
    out = src
    for task in ("sentiment", "emotion"):
        path = RESULTS / f"leaderboard_{task}.csv"
        if not path.exists():
            print(f"  ⚠  leaderboard_{task}.csv not found — leaving placeholder.")
            continue
        df = pd.read_csv(path)
        table = _df_to_md(df)
        tag = task.upper()
        begin = f"<!-- BEGIN_{tag}_LEADERBOARD -->"
        end = f"<!-- END_{tag}_LEADERBOARD -->"
        if begin in out and end in out:
            before = out.split(begin, 1)[0] + begin + "\n"
            after = "\n" + end + out.split(end, 1)[1]
            out = before + table + after
            print(f"  ✓  substituted {task} leaderboard ({len(df)} rows)")
        else:
            print(f"  ⚠  could not find {tag} placeholder markers")
    return out


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def main() -> int:
    src = MASTER.read_text(encoding="utf-8")
    src = substitute_leaderboards(src)
    OUT_FILLED.write_text(src, encoding="utf-8")
    print(f"\nWrote filled Markdown -> {OUT_FILLED.name}")

    out_tex = HERE / "Assignment3_TechnicalReport.tex"
    out_docx = HERE / "Assignment3_TechnicalReport.docx"
    out_pdf = HERE / "Assignment3_TechnicalReport.pdf"

    if not _have("pandoc"):
        print("\n⚠  pandoc not installed — skipping DOCX / PDF / TeX outputs.")
        print("    Install with:  winget install --id JohnMacFarlane.Pandoc -e")
        return 0

    common = [
        "pandoc",
        str(OUT_FILLED),
        "--from", "markdown+pipe_tables+raw_html",
        "--standalone",
        "--metadata", "lang=en",
    ]

    # --- LaTeX ---
    print("\nRunning pandoc → LaTeX ...")
    subprocess.run(common + ["--to", "latex", "-o", str(out_tex)], check=True)
    print(f"  wrote {out_tex.name}")

    # --- DOCX ---
    print("Running pandoc → DOCX ...")
    subprocess.run(common + ["--to", "docx", "-o", str(out_docx)], check=True)
    print(f"  wrote {out_docx.name}")

    # --- PDF (via xelatex if present, else default) ---
    print("Running pandoc → PDF ...")
    pdf_cmd = common + ["--to", "pdf", "-o", str(out_pdf)]
    if _have("xelatex"):
        pdf_cmd += ["--pdf-engine=xelatex"]
    try:
        subprocess.run(pdf_cmd, check=True)
        print(f"  wrote {out_pdf.name}")
    except subprocess.CalledProcessError:
        print("  ⚠  PDF generation failed — install MiKTeX or TeX Live, "
              "or open the .docx in Word and use 'Save As PDF'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
