"""Refresh submission documents and review notebooks from official results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(relative: str, content: str) -> None:
    path = PROJECT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content.strip() + "\n")


def _task_summary(task: str) -> dict[str, Any]:
    results = PROJECT_ROOT / "outputs" / task / "results"
    return {
        "aggregate": _load(results / "aggregate_metrics.json"),
        "split": _load(results / "split_summary.json"),
    }


def _notebook(title: str, sections: list[tuple[str, str]], code_cells: list[str]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        {"cell_type": "markdown", "metadata": {}, "source": [title + "\n"]}
    ]
    for heading, body in sections:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {heading}\n\n{body}\n"],
            }
        )
    cells.extend(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code + "\n"],
        }
        for code in code_cells
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_notebooks() -> None:
    common_setup = "from pathlib import Path\nimport json\nimport pandas as pd\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()"
    notebooks = {
        "01_dataset_analysis.ipynb": _notebook(
            "# 01 Dataset Analysis and Preprocessing",
            [
                ("Scope", "Review the official group-safe sentiment and emotion datasets without regenerating them."),
                ("Weak Labels", "SentiUrdu-1M labels are weak references. No human-gold evaluation is claimed."),
                ("Preprocessing", "Unicode normalization, URL/mention removal, hashtag text preservation, emoji removal, punctuation/number removal, and a two-token minimum are shared."),
                ("Split Safety", "Connected rows sharing an ID or normalized text stay in one split; conflicting-label groups are excluded."),
                ("Artifacts", "Official summaries: `outputs/sentiment/results/split_summary.json` and `outputs/emotion/results/split_summary.json`. Official splits: `data/splits/sentiment/train.csv`, `data/splits/sentiment/validation.csv`, `data/splits/sentiment/test.csv`, `data/splits/emotion/train.csv`, `data/splits/emotion/validation.csv`, and `data/splits/emotion/test.csv`."),
            ],
            [
                common_setup,
                "summaries = {task: json.loads((ROOT / f'outputs/{task}/results/split_summary.json').read_text(encoding='utf-8')) for task in ('sentiment', 'emotion')}\npd.DataFrame({task: {'rows': value['rows_after_filtering'], 'train': value['train_size'], 'validation': value['validation_size'], 'test': value['test_size']} for task, value in summaries.items()}).T",
                "for task in ('sentiment', 'emotion'):\n    frame = pd.read_csv(ROOT / f'data/splits/{task}/test.csv', usecols=['task_label'])\n    print(task, frame['task_label'].value_counts().to_dict())",
            ],
        ),
        "02_baseline_models.ipynb": _notebook(
            "# 02 Baseline Statistical Models",
            [
                ("Models", "TF-IDF Logistic Regression, Linear SVM, and Multinomial Naive Bayes were run with seeds 42, 52, and 62 for both tasks."),
                ("Selection", "Models are ranked by mean validation macro-F1, never by test performance."),
                ("Sentiment", "Linear SVM is selected with test macro-F1 0.4590."),
                ("Emotion", "Linear SVM is selected with test macro-F1 0.2854."),
                ("Artifacts", "Aggregate files: `outputs/sentiment/results/aggregate_metrics.json`, `outputs/emotion/results/aggregate_metrics.json`, `outputs/sentiment/results/model_comparison_leaderboard.csv`, and `outputs/emotion/results/model_comparison_leaderboard.csv`. Baseline run metrics are stored under `outputs/{task}/runs/baseline/{model}/seed_{seed}/results/baseline_metrics.json`."),
            ],
            [
                common_setup,
                "sentiment = pd.read_csv(ROOT / 'outputs/sentiment/results/model_comparison_leaderboard.csv')\nemotion = pd.read_csv(ROOT / 'outputs/emotion/results/model_comparison_leaderboard.csv')\nsentiment[sentiment.model_family == 'baseline']",
                "emotion[emotion.model_family == 'baseline']",
            ],
        ),
        "03_neural_models.ipynb": _notebook(
            "# 03 Neural Models",
            [
                ("Models", "Text-CNN and BiLSTM-Attention use task-specific vocabularies, class weights, early stopping, and validation macro-F1 checkpoints."),
                ("Repeated Seeds", "Both neural models were trained with seeds 42, 52, and 62 on both tasks."),
                ("Uncertainty", "Mean and standard deviation across seeds are retained in each task leaderboard."),
                ("Interpretation", "Neural models did not exceed the selected Linear SVM under the official group-safe protocol."),
                ("Artifacts", "Representative run metrics are `outputs/sentiment/runs/neural/text_cnn/seed_42/results/neural_metrics.json` and `outputs/emotion/runs/neural/bilstm_attention/seed_42/results/neural_metrics.json`; task aggregates are `outputs/sentiment/results/aggregate_metrics.json` and `outputs/emotion/results/aggregate_metrics.json`."),
            ],
            [
                common_setup,
                "frames = {task: pd.read_csv(ROOT / f'outputs/{task}/results/model_comparison_leaderboard.csv') for task in ('sentiment', 'emotion')}\nframes['sentiment'][frames['sentiment'].model_family == 'neural']",
                "frames['emotion'][frames['emotion'].model_family == 'neural']",
            ],
        ),
        "04_transformer_models.ipynb": _notebook(
            "# 04 Transformer Models and Explanation Assistant",
            [
                ("Models", "mBERT, XLM-RoBERTa, and Urdu-RoBERTa were evaluated for both tasks."),
                ("Compute Budget", "Transformer runs use seed 42, 50,000 training rows, and one epoch because of the 24-36 hour compute window."),
                ("Status", "These are explicitly resource-limited pilot results, not matched-budget claims about Transformer capability."),
                ("Selection", "Urdu-RoBERTa is the strongest Transformer by validation macro-F1 on both tasks, but Linear SVM remains the official selected model."),
                ("Artifacts", "Representative Urdu-RoBERTa metrics are `outputs/sentiment/runs/transformer/urdu_roberta/seed_42/results/transformer_metrics.json` and `outputs/emotion/runs/transformer/urdu_roberta/seed_42/results/transformer_metrics.json`; task leaderboards are `outputs/sentiment/results/model_comparison_leaderboard.csv` and `outputs/emotion/results/model_comparison_leaderboard.csv`."),
            ],
            [
                common_setup,
                "sentiment = pd.read_csv(ROOT / 'outputs/sentiment/results/model_comparison_leaderboard.csv')\nsentiment[sentiment.model_family == 'transformer']",
                "emotion = pd.read_csv(ROOT / 'outputs/emotion/results/model_comparison_leaderboard.csv')\nemotion[emotion.model_family == 'transformer']",
            ],
        ),
        "05_error_analysis.ipynb": _notebook(
            "# 05 Error Analysis",
            [
                ("Selected Runs", "Canonical seed-42 Linear SVM predictions are analyzed for sentiment and emotion."),
                ("Class Imbalance", "Rare Neutral and rare emotion classes remain difficult despite macro-F1 selection."),
                ("Weak Reference Labels", "Disagreement with a weak label is not automatically a linguistic model error."),
                ("Qualitative Risks", "Short context, sarcasm, negation, spelling variation, and code-mixing remain important error sources."),
                ("Artifacts", "Predictions: `outputs/sentiment/runs/baseline/linear_svm/seed_42/predictions/baseline_linear_svm_test_predictions.csv` and `outputs/emotion/runs/baseline/linear_svm/seed_42/predictions/baseline_linear_svm_test_predictions.csv`. Confusion matrices: `outputs/sentiment/runs/baseline/linear_svm/seed_42/results/confusion_matrix_baseline_linear_svm_test.csv` and `outputs/emotion/runs/baseline/linear_svm/seed_42/results/confusion_matrix_baseline_linear_svm_test.csv`."),
            ],
            [
                common_setup,
                "paths = {task: ROOT / f'outputs/{task}/runs/baseline/linear_svm/seed_42/predictions/baseline_linear_svm_test_predictions.csv' for task in ('sentiment', 'emotion')}\nerrors = {task: pd.read_csv(path) for task, path in paths.items()}\n{task: int((~frame.is_correct).sum()) for task, frame in errors.items()}",
                "for task, frame in errors.items():\n    display(frame.loc[~frame.is_correct, ['raw_text', 'true_label', 'predicted_label']].head(10))",
            ],
        ),
    }
    for filename, payload in notebooks.items():
        with (PROJECT_ROOT / "notebooks" / filename).open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def refresh() -> None:
    sentiment = _task_summary("sentiment")
    emotion = _task_summary("emotion")
    sent = sentiment["aggregate"]["selected_model"]
    emo = emotion["aggregate"]["selected_model"]
    sent_split = sentiment["split"]
    emo_split = emotion["split"]

    readme = f"""# Robust Sentiment and Emotion Classification on Noisy Urdu Tweets

Group-safe, dual-task evaluation on the local SentiUrdu-1M corpus for the CSC-355 Natural Language Processing Design Project, Namal University Mianwali.

**Students:** M. Raqib Hayat (NUM-BSCS-2022-40) and Abu Bakar (NUM-BSCS-2022-41)<br>
**Instructor:** Dr. Muzamil Ahmed

## Official Benchmark

The repository contains 36 official runs across two separate pipelines and eight models. Classical and neural models use seeds 42, 52, and 62; resource-limited Transformers use seed 42, 50,000 training rows, and one epoch. Mean validation macro-F1 determines selection. Test results are reported only after selection.

| Task | Retained rows | Train / validation / test | Selected model | Test macro-F1 | Bootstrap 95% interval |
|---|---:|---:|---|---:|---:|
| Sentiment | {sent_split['rows_after_filtering']:,} | {sent_split['train_size']:,} / {sent_split['validation_size']:,} / {sent_split['test_size']:,} | Linear SVM | {sent['test_macro_f1_mean']:.4f} | [{sent['bootstrap_95']['lower_95']:.4f}, {sent['bootstrap_95']['upper_95']:.4f}] |
| Emotion | {emo_split['rows_after_filtering']:,} | {emo_split['train_size']:,} / {emo_split['validation_size']:,} / {emo_split['test_size']:,} | Linear SVM | {emo['test_macro_f1_mean']:.4f} | [{emo['bootstrap_95']['lower_95']:.4f}, {emo['bootstrap_95']['upper_95']:.4f}] |

Labels are weak references derived from the source dataset. No human-gold evaluation is claimed.

## Data Integrity

- Emojis are removed before feature extraction to reduce direct weak-label shortcut learning.
- Rows connected by a shared tweet ID or normalized text are assigned to only one partition.
- Connected groups with conflicting task labels are excluded.
- Automated checks confirm zero shared IDs and zero shared normalized texts across train, validation, and test for both tasks.
- Official splits are in `data/splits/sentiment/` and `data/splits/emotion/`.

## Models

The benchmark evaluates Logistic Regression, Linear SVM, Multinomial Naive Bayes, Text-CNN, BiLSTM-Attention, mBERT, XLM-RoBERTa, and Urdu-RoBERTa. Every run stores its model checkpoint, validation/test predictions, metrics, and metadata under `outputs/{{task}}/runs/`.

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
"""
    _write("README.md", readme)

    common = f"""The official project is a dual-task benchmark with 36 official runs. Sentiment retains {sent_split['rows_after_filtering']:,} rows and emotion retains {emo_split['rows_after_filtering']:,} rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 {sent['test_macro_f1_mean']:.4f} for sentiment and {emo['test_macro_f1_mean']:.4f} for emotion. Labels are weak references; no human-gold evaluation is claimed."""

    docs = {
        "docs/dataset_description.md": f"""# Dataset Description

{common}

The raw file `data/raw/Urdu Tweets Dataset.csv` has 1,048,000 rows and columns `Id`, `Text`, `Emotions`, and `Category`. Of these, 514,571 rows lack the required category label. The pipeline removes URLs, mentions, emojis, numbers, punctuation, empty/short text, conflicting connected groups, and exact normalized-text duplicates.

| Task | Train | Validation | Test | Classes |
|---|---:|---:|---:|---:|
| Sentiment | {sent_split['train_size']:,} | {sent_split['validation_size']:,} | {sent_split['test_size']:,} | 3 |
| Emotion | {emo_split['train_size']:,} | {emo_split['validation_size']:,} | {emo_split['test_size']:,} | 6 |

See `reports/dataset_card.md` and each task's `outputs/<task>/results/split_summary.json` for complete distributions and removal counts.
""",
        "docs/methodology.md": f"""# Methodology

{common}

## Protocol

1. Normalize Urdu/Arabic Unicode.
2. Remove URLs and mentions; preserve hashtag text.
3. Remove emojis before feature extraction.
4. Remove numbers/punctuation and require at least two cleaned tokens.
5. Build connected duplicate groups using shared ID or normalized text.
6. Exclude connected groups containing conflicting task labels.
7. Deduplicate normalized text and create deterministic 70/15/15 group-level splits.
8. Fit all text representations on training data only.
9. Run eight models with three classical/neural seeds and one resource-limited Transformer seed.
10. Rank by mean validation macro-F1 and bootstrap the selected test predictions.

Task configurations are `config_sentiment.yaml` and `config_emotion.yaml`. Run isolation is under `outputs/<task>/runs/<family>/<model>/seed_<seed>/`.
""",
        "docs/experiment_setup.md": f"""# Experiment Setup

{common}

Training ran on an NVIDIA GeForce RTX 5070 Ti within a 24-36 hour execution window. Classical and neural runs use seeds 42, 52, and 62. mBERT, XLM-RoBERTa, and Urdu-RoBERTa use seed 42, 50,000 training examples, one epoch, and mixed precision. Transformer results are resource-limited pilots.

The complete run count is 18 classical + 12 neural + 6 Transformer = 36. Every run includes validation and test predictions plus a saved model artifact.
""",
        "docs/results_analysis.md": f"""# Results Analysis

{common}

| Task | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|---:|
| Sentiment Linear SVM | {sent['validation_macro_f1_mean']:.4f} | {sentiment['aggregate']['models'][1]['test_accuracy_mean']:.4f} | {sent['test_macro_f1_mean']:.4f} | {sentiment['aggregate']['models'][1]['test_weighted_f1_mean']:.4f} |
| Emotion Linear SVM | {emo['validation_macro_f1_mean']:.4f} | {emotion['aggregate']['models'][1]['test_accuracy_mean']:.4f} | {emo['test_macro_f1_mean']:.4f} | {emotion['aggregate']['models'][1]['test_weighted_f1_mean']:.4f} |

The gap between accuracy and macro-F1 reflects majority-class dominance. Emotion is harder because six classes include several very rare labels. See `reports/final_evaluation_summary.md` and the task leaderboards for all eight models and seed variation.
""",
        "docs/ethics_and_limitations.md": f"""# Ethics and Limitations

{common}

- Weak labels can encode emoji heuristics and annotation noise; disagreement is not always a model error.
- Exact tweets are retained only for local academic analysis and should not be used for user profiling.
- Severe imbalance limits reliability for Neutral, Angry, Fear, Disgust, Surprise, and Sad.
- Transformer experiments are not compute-matched to classical/neural runs.
- Linear SVM output is a decision score, not a calibrated probability.
- No deployment should make consequential decisions without native-speaker review, consent, and domain-specific validation.
""",
        "docs/demonstration_guide.md": f"""# Demonstration Guide

{common}

1. Run `streamlit run app/streamlit_app.py`.
2. Select Sentiment or Emotion.
3. Select one of the eight saved models; Linear SVM is the dependable selected model for both tasks.
4. Enter an Urdu tweet and show the cleaned text after emoji/URL/mention removal.
5. Run prediction and explain that SVM shows a decision score rather than calibrated confidence.
6. Show the validation-ranked leaderboard and the weak-label/group-safe disclosures.
7. Switch tasks to demonstrate separate pipelines and artifacts.

Do not claim human-gold accuracy or that the one-epoch Transformer pilots represent their maximum capability.
""",
        "docs/demo_script.md": f"""# Demo Script

"This project evaluates sentiment and emotion separately on noisy Urdu tweets. The official benchmark contains 36 official runs across eight models. We remove emojis because the source labels are weakly related to emoji signals, and we prevent shared IDs or normalized text from crossing data splits. Models are selected using validation macro-F1 only. Linear SVM is selected for both tasks, with test macro-F1 {sent['test_macro_f1_mean']:.4f} for sentiment and {emo['test_macro_f1_mean']:.4f} for emotion. The labels are weak references and no human-gold evaluation is claimed."

During the live demo, switch between tasks, show preprocessing, predict with Linear SVM, then compare another model using the validation-ranked table.
""",
        "docs/presentation_outline.md": f"""# Presentation Outline

1. Problem: noisy, code-mixed, weakly labeled Urdu tweets.
2. Dataset: 1,048,000 raw rows and task-specific retained sets.
3. Leakage control: emoji removal and connected duplicate grouping.
4. Two tasks: three-class sentiment and six-class emotion.
5. Eight models across classical, neural, and Transformer families.
6. Protocol: 36 official runs and validation macro-F1 selection.
7. Sentiment result: Linear SVM test macro-F1 {sent['test_macro_f1_mean']:.4f}.
8. Emotion result: Linear SVM test macro-F1 {emo['test_macro_f1_mean']:.4f}.
9. Error analysis: imbalance, rare labels, context, and weak-reference noise.
10. Demo: task selector, preprocessing, model selector, prediction, leaderboard.
11. Limitations: no human-gold set and resource-limited Transformer pilots.
12. Conclusion and future native-speaker annotation.
""",
        "docs/final_project_audit.md": f"""# Final Project Audit

Audit date: June 15, 2026.

{common}

## Closed Gaps

- Separate sentiment and emotion configs, splits, runs, results, inference, and demo paths.
- Connected ID/text duplicate grouping with zero measured cross-split overlap.
- Three seeds for classical/neural models and one declared resource-limited Transformer seed.
- Validation-only selection, canonical runs, bootstrap intervals, complete predictions/checkpoints, and generated reporting.

## Remaining Scientific Limitation

No native-speaker human-gold evaluation was possible. This is disclosed throughout the package and prevents a claim of gold-standard real-world performance.
""",
        "docs/independent_rubric_gap_analysis_2026-06-15.md": f"""# Independent Rubric and Gap Analysis

Audit date: June 15, 2026. Status: post-remediation.

{common}

The earlier audit identified duplicate cross-split overlap, mixed experiment snapshots, incomplete task packaging, incomplete repeated seeds, test-based ranking risk, missing model weights, and stale documentation. These issues are closed in the official task-isolated 36-run benchmark. The remaining gap is the absence of native-speaker human-gold labels. Rubric evidence coverage reaches all 50 available marks, but no grade is guaranteed.
""",
        "docs/final_submission_checklist.md": """# Final Submission Checklist

- [x] Separate sentiment and emotion configurations.
- [x] Group-safe train/validation/test splits with zero ID/text overlap.
- [x] Eight models evaluated for both tasks.
- [x] 36 official runs saved with models, predictions, metrics, and metadata.
- [x] Validation macro-F1 used for model selection.
- [x] Bootstrap intervals generated for selected test predictions.
- [x] Dual-task Streamlit demo available.
- [x] Final Markdown report, PDF, presentation, cards, manifest, and figures generated.
- [x] Weak-label and no-human-gold limitations disclosed.
- [ ] Confirm final archive/upload limits and repository URL on the submission portal.
""",
        "docs/professor_requirements_alignment.md": f"""# Professor Requirements Alignment

The package maps evidence to all six project stages and all 50 available rubric marks. This is evidence coverage, not a promised grade.

| Stage | Evidence | Status |
|---|---|---|
| Problem and proposal | README and final report | Ready |
| Literature review and gap | Related Work and 20 references in final report | Ready |
| Design and methodology | Group-safe dual-task protocol and architecture modules | Ready |
| Implementation | 36 official runs, eight models, saved artifacts | Ready |
| Evaluation and optimization | Validation ranking, repeated seeds, bootstrap intervals, per-class/confusion outputs | Ready |
| Report and demonstration | PDF, PPTX, Streamlit app, demo guide | Ready |

Headline results are sentiment macro-F1 {sent['test_macro_f1_mean']:.4f} and emotion macro-F1 {emo['test_macro_f1_mean']:.4f}, both from selected Linear SVM runs. Labels remain weak references.
""",
        "docs/submission_manifest.md": """# Submission Upload Manifest

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
""",
    }
    for path, content in docs.items():
        _write(path, content)

    readmes = {
        "data/processed/README.md": f"""# Processed Data

Task-specific datasets are stored in `sentiment/dataset.csv` ({sent_split['rows_after_filtering']:,} rows) and `emotion/dataset.csv` ({emo_split['rows_after_filtering']:,} rows). They contain `id`, `raw_text`, `clean_text`, source labels, canonical labels, task labels, and token length. Generate them through `src/create_splits.py` with the corresponding task config.
""",
        "data/splits/README.md": f"""# Group-Safe Data Splits

Official splits are under `sentiment/` and `emotion/`. Sentiment sizes are {sent_split['train_size']:,}/{sent_split['validation_size']:,}/{sent_split['test_size']:,}; emotion sizes are {emo_split['train_size']:,}/{emo_split['validation_size']:,}/{emo_split['test_size']:,}. Connected shared-ID/shared-text groups never cross partitions, and automated checks find zero overlap.
""",
        "notebooks/README.md": """# Analysis Notebooks

The five notebooks review saved dual-task artifacts without retraining: dataset/splits, baseline models, neural models, Transformers, and error analysis. They reference the official task-specific `outputs/<task>/` and `data/splits/<task>/` paths.
""",
        "outputs/README.md": """# Outputs

Official artifacts are task-isolated under `sentiment/` and `emotion/`. Each task contains `runs/<family>/<model>/seed_<seed>/` with models, predictions, results, and metadata, plus task-level aggregate results and figures. The complete benchmark contains 36 official runs.
""",
        "outputs/predictions/README.md": """# Legacy Prediction Snapshot

This folder is retained for provenance. Official current predictions are stored inside each isolated run under `outputs/<task>/runs/<family>/<model>/seed_<seed>/predictions/`. Prediction files include `id`, `raw_text`, `clean_text`, `true_label`, `predicted_label`, `confidence`, `is_correct`, `split`, `model_name`, and `text_length`.
""",
        "outputs/results/README.md": """# Legacy Result Snapshot

This folder is retained for provenance. Official current aggregates are `outputs/sentiment/results/` and `outputs/emotion/results/`, including validation-ranked leaderboards, split summaries, and aggregate metrics.
""",
        "outputs/models/README.md": """# Legacy Model Snapshot

This folder is retained for provenance. Complete official model artifacts are saved in each task-specific run under `outputs/<task>/runs/<family>/<model>/seed_<seed>/models/`.
""",
        "outputs/figures/README.md": """# Legacy Figures

This folder is retained for provenance. Current task figures are under `outputs/sentiment/figures/`, `outputs/emotion/figures/`, and `reports/figures/`.
""",
        "outputs/error_analysis/README.md": """# Legacy Error Analysis

This folder is retained for provenance. Official selected-run predictions and confusion matrices are under each task's Linear SVM seed-42 run; the final report summarizes their class-level errors and limitations.
""",
        "outputs/reports/README.md": """# Legacy Report Location

Current submission reports are under `reports/`: Markdown source, PDF, presentation, evaluation summary, dataset card, model card, manifest, and figures.
""",
        "outputs/report_snapshot/README.md": """# Historical Snapshot

This directory preserves earlier evidence for provenance only. The official current benchmark is the task-isolated 36-run experiment under `outputs/sentiment/` and `outputs/emotion/`; do not mix historical metrics with current results.
""",
        "app/README.md": """# Streamlit Demo

Run `streamlit run app/streamlit_app.py`. The app supports sentiment and emotion, all eight saved models, task-specific preprocessing and inference, validation-ranked leaderboards, and weak-label/group-safe disclosures.
""",
    }
    for path, content in readmes.items():
        _write(path, content)

    _write_notebooks()


if __name__ == "__main__":
    refresh()
    print("Submission materials refreshed from official results.")
