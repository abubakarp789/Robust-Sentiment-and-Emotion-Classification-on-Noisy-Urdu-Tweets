# Submission Upload Manifest

This is the authoritative hand-in list for the CSC-355 final project.

## Preferred Upload

Upload one archive named `NLP_Final_Project.zip` containing the complete `final_project/` folder. The archive should retain this structure:

The current uncompressed package is approximately 892 MB, mostly because it includes the dataset and saved prediction tables.

- `README.md`, `requirements.txt`, and `config.yaml`
- `src/`: reusable preprocessing, modeling, evaluation, inference, and validation code
- `scripts/`: numbered data preparation, training, evaluation, and visualization commands
- `tests/`: automated regression and requirement checks
- `app/`: Streamlit demonstration
- `notebooks/`: five reviewer notebooks
- `docs/`: methodology, dataset, setup, results, ethics, rubric alignment, and demo material
- `data/`: raw, processed, split, and annotation data
- `outputs/`: report, models, predictions, metrics, figures, and error analysis

Also upload these separately when the LMS provides dedicated fields:

1. `outputs/reports/final_nlp_project_report.pdf` as the final report.
2. The Git repository URL as the source-code repository after the final local changes have been committed and pushed: `https://github.com/abubakarp789/Robust-Sentiment-and-Emotion-Classification-on-Noisy-Urdu-Tweets`
3. Presentation material only if the instructor requests a separate presentation file; the prepared content is in `docs/presentation_outline.md`.

## Required Evidence Inside the Archive

| Professor deliverable | Submission path |
|---|---|
| Complete source code | `src/`, `scripts/`, `app/` |
| Data processing pipeline | `scripts/01_prepare_data.py`, `src/preprocessing.py`, `src/label_mapping.py`, `src/create_splits.py` |
| Model training scripts | `scripts/02_train_classical.py`, `scripts/03_train_neural.py`, `scripts/04_train_transformers.py`, corresponding `src/train_*.py` modules |
| Evaluation scripts | `scripts/05_evaluate_models.py`, `src/evaluate.py`, `src/evaluation_workflow.py`, error-analysis modules |
| Visualizations | `outputs/figures/`, `outputs/report_snapshot/`, `scripts/06_generate_visualizations.py` |
| Documentation | Root README, all folder READMEs, `docs/`, notebooks |
| Final report | `outputs/reports/final_nlp_project_report.pdf` |
| Demonstration | `app/streamlit_app.py`, included Linear SVM model, demo guide and script |

## If the Portal Has a Size Limit

Do not silently remove evidence. Split the upload into two archives:

1. `NLP_Final_Project_Code_Report.zip`: everything except `data/raw/`, `data/processed/`, `data/splits/`, and `outputs/predictions/`.
2. `NLP_Final_Project_Data_Evidence.zip`: `data/raw/`, `data/processed/`, `data/splits/`, and `outputs/predictions/`.

Before compression, these groups are approximately 71 MB and 821 MB respectively.

Keep `data/annotation/`, `outputs/results/`, `outputs/report_snapshot/`, `outputs/error_analysis/`, `outputs/figures/`, `outputs/models/`, and `outputs/reports/` in the code/report archive because they are important grading evidence. If only one archive is accepted, confirm the upload limit with the instructor before omitting the large data-evidence archive.

## Do Not Upload

- `.venv/` or other local virtual environments
- `__pycache__/`, `.pytest_cache/`, or notebook checkpoint folders
- Temporary files or regenerated `outputs/metrics/` unless specifically requested
- Duplicate copies of the final report
- Earlier assignment folders unless the instructor asks for milestone history

## Final Verification

Run from `final_project` before creating the archive:

```powershell
python -m pytest tests
python src\validate_pipeline.py --config config.yaml
python src\validate_professor_requirements.py
python src\validate_readme_links.py
python src\validate_final_project.py --config config.yaml
```

The research-paper results in `outputs/report_snapshot/` and the later packaged sentiment results in `outputs/results/` are separate experiment snapshots. Keep both and do not combine their metrics.
