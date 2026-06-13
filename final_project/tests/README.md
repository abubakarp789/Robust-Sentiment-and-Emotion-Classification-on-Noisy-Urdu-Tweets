# Regression Tests

## Purpose

This folder contains pytest test files used for validation, regression testing, and code quality control. These tests verify the pipeline modules, data partitions, and deployment code to ensure stability.

## Contents

| File/Folder | Description |
|---|---|
| [test_compare_models.py](test_compare_models.py) | Tests model comparison aggregation and leaderboard formatting |
| [test_data_organization.py](test_data_organization.py) | Validates rows, column schema, and properties of data splits |
| [test_error_analysis.py](test_error_analysis.py) | Tests correct categorization of model errors |
| [test_evaluate.py](test_evaluate.py) | Validates precision, recall, and F1 calculations |
| [test_neural_utils.py](test_neural_utils.py) | Verifies tokenization and vocabulary mapping utilities |
| [test_streamlit_app.py](test_streamlit_app.py) | Tests Streamlit UI and inference flow when Streamlit is installed; otherwise skips those optional checks |
| [test_train_baseline.py](test_train_baseline.py) | Verifies baseline model training runs and joblib exports |
| [test_transformer.py](test_transformer.py) | Tests tokenization and classification helpers for transformers |
| [test_validate_notebooks.py](test_validate_notebooks.py) | Checks notebook structures and validation routines |
| [test_preprocessing.py](test_preprocessing.py) | Tests emoji, URL, mention, and hashtag cleaning |
| [test_label_mapping.py](test_label_mapping.py) | Tests noisy emotion and sentiment mappings |
| [test_metrics.py](test_metrics.py) | Tests aggregate and per-class metric calculation |
| [test_professor_requirements.py](test_professor_requirements.py) | Ensures every professor requirement has evidence |
| [test_documentation_alignment.py](test_documentation_alignment.py) | Checks self-contained paths, documented schemas, and upload-manifest coverage |

## How It Is Used

Tests are run automatically during development or before final submission to protect code and data integrity.
To run the full suite:
```powershell
python -m pytest tests
```

## Related Files

- [../README.md](../README.md)
- [../src/validate_final_project.py](../src/validate_final_project.py)

## Notes

- Tests use small fixtures or saved metadata rather than retraining models. The suite is intended to finish in under a minute on the project machine.
