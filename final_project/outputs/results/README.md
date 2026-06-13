# Results Folder

## Purpose

This folder contains the later 517,966-row packaged sentiment rerun: quantitative metrics, leaderboards, classification reports, training histories, and confusion matrices. The earlier dual-task report results are kept separately in `../report_snapshot/`.

## Contents

This folder contains metrics and reports for all models. Key summary files include:

| File/Folder | Description |
|---|---|
| [model_comparison_leaderboard.csv](model_comparison_leaderboard.csv) | Master leaderboard comparing accuracy and F1 metrics across all models |
| [final_evaluation_summary.json](final_evaluation_summary.json) | High-level summary JSON documenting project metadata and rankings |
| [split_summary.json](split_summary.json) | Documented row counts and class distributions for the splits |
| [label_mapping_summary.json](label_mapping_summary.json) | Statistics on raw label mappings and normalized surface forms |

## How It Is Used

The files in this directory are generated and referenced by the evaluation pipeline:
- **Baseline/Neural/Transformer Results**: Metrics are written here as JSON/CSV logs after training finishes.
- **Leaderboard compilation**: `python src/compare_models.py --config config.yaml` aggregates metrics into [model_comparison_leaderboard.csv](model_comparison_leaderboard.csv).
- **Streamlit Integration**: The interactive dashboard reads [model_comparison_leaderboard.csv](model_comparison_leaderboard.csv) to display the model leaderboard.
- **Jupyter Notebooks**: Notebooks under `notebooks/` load these logs to plot graphs.

## Related Files

- [../README.md](../README.md)
- [../predictions/](../predictions/)
- [../../src/compare_models.py](../../src/compare_models.py)

## Notes

- **Warning**: Do not modify any JSON or CSV files in this folder manually. If evaluation metrics need to be updated, run evaluation scripts or compile scripts in `src/`.
