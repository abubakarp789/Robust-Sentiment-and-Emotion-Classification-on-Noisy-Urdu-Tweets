# Robust Sentiment and Emotion Classification on Noisy Urdu Tweets

Group-safe, dual-task evaluation on the local SentiUrdu-1M corpus for the CSC-355 Natural Language Processing Design Project, Namal University Mianwali.

**Students:** M. Raqib Hayat (NUM-BSCS-2022-40) and Abu Bakar (NUM-BSCS-2022-41)<br>
**Instructor:** Dr. Muzamil Ahmed

## Official Benchmark

The repository contains 36 official runs across two separate pipelines and eight models. Classical and neural models use seeds 42, 52, and 62; resource-limited Transformers use seed 42, 50,000 training rows, and one epoch. Mean validation macro-F1 determines selection. Test results are reported only after selection.

| Task | Retained rows | Train / validation / test | Selected model | Test macro-F1 | Bootstrap 95% interval |
|---|---:|---:|---|---:|---:|
| Sentiment | 446,745 | 312,710 / 67,016 / 67,019 | Linear SVM | 0.4590 | [0.4456, 0.4742] |
| Emotion | 446,640 | 312,643 / 66,995 / 67,002 | Linear SVM | 0.2854 | [0.2749, 0.2962] |

Labels are weak references derived from the source dataset. No human-gold evaluation is claimed.

## Data Integrity

- Emojis are removed before feature extraction to reduce direct weak-label shortcut learning.
- Rows connected by a shared tweet ID or normalized text are assigned to only one partition.
- Connected groups with conflicting task labels are excluded.
- Automated checks confirm zero shared IDs and zero shared normalized texts across train, validation, and test for both tasks.
- Official splits are in `data/splits/sentiment/` and `data/splits/emotion/`.

## Models

The benchmark evaluates Logistic Regression, Linear SVM, Multinomial Naive Bayes, Text-CNN, BiLSTM-Attention, mBERT, XLM-RoBERTa, and Urdu-RoBERTa. Every run stores its model checkpoint, validation/test predictions, metrics, and metadata under `outputs/{task}/runs/`.

## Key Files

- Configurations: `config_sentiment.yaml`, `config_emotion.yaml`
- Final report source: [reports/final_report.md](reports/final_report.md)
- Final report PDF: [reports/final_report.pdf](reports/final_report.pdf)
- Presentation: [reports/final_presentation.pptx](reports/final_presentation.pptx)
- Evaluation summary: [reports/final_evaluation_summary.md](reports/final_evaluation_summary.md)
- Dataset card: [reports/dataset_card.md](reports/dataset_card.md)
- Model card: [reports/model_card.md](reports/model_card.md)
- Experiment manifest: [reports/experiment_manifest.json](reports/experiment_manifest.json)
- Demo guide: [docs/demonstration_guide.md](docs/demonstration_guide.md)
- Submission checklist: [docs/final_submission_checklist.md](docs/final_submission_checklist.md)

## Commands

```powershell
python src/run_experiments.py --config config_sentiment.yaml
python src/run_experiments.py --config config_emotion.yaml
python src/aggregate_experiments.py --config config_sentiment.yaml
python src/aggregate_experiments.py --config config_emotion.yaml
python src/generate_reports.py
python src/validate_official_benchmark.py
python -m pytest tests
streamlit run app/streamlit_app.py
```

The saved official run is complete; these training commands are for reproducibility and will create or overwrite run artifacts according to the supplied task configuration.

## Structure

```text
final_project/
|-- app/                 Dual-task Streamlit demo
|-- data/                Raw data and task-specific processed/split datasets
|-- docs/                Methodology, results, ethics, demo, and submission guidance
|-- notebooks/           Five artifact-review notebooks; no retraining
|-- outputs/             Task-specific runs, aggregates, predictions, and figures
|-- reports/             Final report, cards, manifest, figures, PDF, and presentation
|-- scripts/             Stable command wrappers
|-- src/                 Pipeline, training, inference, reporting, and validation code
|-- tests/               Regression and integrity tests
|-- config_sentiment.yaml
`-- config_emotion.yaml
```

## Limitations

- The labels are weak and highly imbalanced, especially Neutral and rare emotions.
- Transformers are one-seed, one-epoch, 50,000-row pilot runs, so family comparisons are not compute-matched.
- Linear SVM decision scores are not calibrated probabilities.
- Native-speaker human annotation remains the highest-priority future evaluation step.

## Rubric Scope

The package maps evidence to all six stages and 50 available rubric marks. That is a coverage audit, not a guarantee of the awarded grade.
