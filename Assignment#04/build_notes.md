# Build Notes

## Source Status

The report is based only on artifacts in `Assignment#01`, `Assignment#02`, and `Assignment#03`. The report distinguishes observed local data/results from proposal-only goals.

## Compilation

Compilation succeeded on June 13, 2026 using bundled Tectonic 0.16.9 from inside `Assignment#04/report`:

```powershell
tectonic main.tex
```

Tectonic processed the BibTeX database with the IEEEtran bibliography style and created `report/main.pdf`.

- Final page count: 9 pages.
- Output size after architecture redesign: 504,757 bytes (approximately 493 KiB).
- Citation audit: 29 BibTeX entries, 29 cited, 0 missing, 0 unused.
- Abstract length: 251 words.
- Visual inspection: all nine pages rendered; tables, figures, two-column text, and references are visible without clipping.

The generic compilation helper initially skipped the build because no TeX Live installation was found and it conservatively classified BibTeX as unsuitable for Tectonic. Running the bundled Tectonic binary directly succeeded, including BibTeX processing.

Tectonic emitted a non-fatal Windows Fontconfig warning and minor underfull-box warnings in the software/hardware paragraph. The PDF was still generated correctly, and visual inspection found no clipped content.

## Known Review Items

- Author email addresses were not available in the inspected sources.
- The proposed manually verified gold test set and deployed demo were not found.
- No repeated-seed uncertainty estimates or statistical significance tests were found.
- Hardware and runtime values are transcribed from Assignment 3 rather than re-benchmarked.
- Confirm whether Assignment 4 needs separate plagiarism/similarity and AI-use reports.

## Figure Provenance

The architecture figure was redesigned on June 13, 2026 as a vector-first Matplotlib composition. The new design uses a strict three-row grid, numbered stage badges, pastel academic color coding, subtle depth, symmetrical branch routing, metric chips, and a separate leakage-prevention callout. The report now embeds the vector PDF rather than the raster preview.

- `emotion_distribution.png` is copied from Assignment 3 EDA output.
- `f1_comparison.png` combines the two Assignment 3 F1 comparison figures without changing values.
- `best_model_confusions.png` is regenerated from the saved Assignment 3 prediction CSVs for Urdu-RoBERTa sentiment and mBERT emotion.
- `label_leakage_heatmap.png` recreates the conditional values recorded in the Assignment 3 leakage diagnostic.
- `system_architecture.svg` is the editable vector export of the redesigned publication figure.
- `system_architecture.pdf` is the vector version embedded by `report/main.tex`.
- `system_architecture.png` is a 300-DPI preview version.
- `generate_system_architecture.py` is the primary editable source. Regenerate all three formats from `Assignment#04/report/figures` with `python generate_system_architecture.py`.
