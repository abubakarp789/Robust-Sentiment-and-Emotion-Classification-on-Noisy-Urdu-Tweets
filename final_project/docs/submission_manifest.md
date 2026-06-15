# Submission Upload Manifest

## Required Package

- Complete source code: `src/`, `scripts/`, `app/`
- Data processing pipeline: task configs, preprocessing, label mapping, group-safe split generator
- Model training scripts: classical, neural, and Transformer workflows
- Evaluation scripts: aggregation, metrics, error analysis, official benchmark validator
- Visualizations: `reports/figures/` and task-specific output figures
- Documentation: README, `docs/`, `reports/`, and notebooks
- Final report: `reports/final_report.pdf`
- Presentation: `reports/final_presentation.pptx`
- Demonstration: `app/streamlit_app.py` plus demo guide
- Source repository URL: submit the final repository link with the archive

The archive contains 36 official runs and all saved model artifacts. If the portal has a size limit, preserve source, reports, task aggregates, selected model artifacts, and the experiment manifest; place large non-selected run artifacts in a separately labeled evidence archive rather than silently deleting them.

## Final Verification

```powershell
python -m pytest tests
python src/validate_official_benchmark.py
python src/validate_notebooks.py --config config.yaml
python src/validate_readme_links.py
python src/validate_final_project.py --config config_sentiment.yaml
```
