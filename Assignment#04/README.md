# Assignment 4 Final Report Package

This folder consolidates the permitted work from `Assignment#01`, `Assignment#02`, and `Assignment#03` into an IEEE-style final technical report. The original assignment folders were treated as read-only sources. The `final_project` folder was not inspected or modified.

## Contents

- `source_analysis.md`: evidence extracted from each earlier assignment, including corrections and unresolved gaps.
- `report/main.tex`: IEEE conference paper source.
- `report/references.bib`: BibTeX records derived from Assignment 2 and the model references already listed in Assignment 3.
- `report/figures/`: architecture, dataset, leakage, F1, and confusion-matrix figures.
- `report/tables/result_tables.tex`: reusable result tables.
- `submission_checklist.md`: final academic and formatting checks.
- `build_notes.md`: compilation outcome, warnings, and remaining review items.

## Compile

From `Assignment#04/report`:

```powershell
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This package was successfully compiled with the bundled Tectonic executable, which also processed `references.bib`:

```powershell
tectonic main.tex
```

The expected output is `Assignment#04/report/main.pdf`.

## Important Evidence Decisions

- The local CSV contains 1,048,000 rows, although the source paper reports 1,140,821 collected tweets. The report states both values and makes clear that experiments used the local CSV.
- Only 533,429 local rows have usable `Category` labels after canonicalization; 514,571 rows have missing categories.
- Assignment 3 contains saved predictions, leaderboards, figures, and report results for seven models on two tasks. These are reported as project results.
- The manually verified 500-1,000 tweet test set and deployed demo proposed in Assignment 1 were not found. They are limitations/future work, not completed contributions.

## Remaining Review

See `build_notes.md` and `submission_checklist.md` for items that still require human confirmation, especially author contact details and whether the instructor requires an AI-use or similarity report with Assignment 4.
