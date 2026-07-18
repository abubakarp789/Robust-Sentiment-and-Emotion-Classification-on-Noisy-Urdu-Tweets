# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python NLP project for Urdu sentiment and emotion classification. Core pipeline code lives in `src/`, including preprocessing, model training, inference, evaluation, reporting, and validation modules. Stable wrapper scripts are in `scripts/`, the Streamlit demo is in `app/`, and review notebooks are in `notebooks/`. Data is organized under `data/` with `raw/`, `processed/`, `splits/`, and `annotation/` subfolders. Generated models, metrics, predictions, figures, and reports are stored under `outputs/`. Regression tests live in `tests/`.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the full test suite:

```powershell
python -m pytest tests
```

Run task pipelines and aggregation:

```powershell
python src/run_experiments.py --config config_sentiment.yaml
python src/run_experiments.py --config config_emotion.yaml
python src/aggregate_experiments.py --config config_sentiment.yaml
```

Launch the local demo:

```powershell
streamlit run app/streamlit_app.py
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and clear `snake_case` names for modules, functions, variables, and test files. Keep reusable logic in `src/`; keep `scripts/` as thin command wrappers. Prefer explicit config-driven behavior using `config.yaml`, `config_sentiment.yaml`, or `config_emotion.yaml` instead of hardcoded paths. Write functions that are deterministic where possible, especially for split creation, metrics, and validation.

## Testing Guidelines

Tests use `pytest` and are named `tests/test_*.py`. Add focused regression tests beside the closest existing test module when changing preprocessing, label mapping, training helpers, evaluation, reporting, or app behavior. Prefer small fixtures and saved metadata over retraining models. Before submitting changes, run `python -m pytest tests`; for targeted work, run a single file such as `python -m pytest tests/test_preprocessing.py`.

## Commit & Pull Request Guidelines

Recent history uses short imperative commits, often Conventional Commit prefixes such as `feat:`, `fix:`, and `docs:`. Follow that style, for example `fix: preserve group-safe split isolation` or `docs: update benchmark notes`. Pull requests should summarize the change, list validation commands run, note any generated artifacts changed under `outputs/`, and include screenshots when the Streamlit UI changes.

## Security & Configuration Tips

Do not commit secrets, local credentials, or machine-specific absolute paths. Keep large regenerated artifacts intentional and documented. When modifying data handling, preserve group-safe split guarantees and avoid leakage between train, validation, and test partitions.
