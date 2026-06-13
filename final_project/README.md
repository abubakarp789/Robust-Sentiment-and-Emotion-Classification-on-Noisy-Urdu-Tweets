# Robust Sentiment and Emotion Classification on Noisy Urdu Tweets

Leak-free multi-stage evaluation on the local SentiUrdu-1M corpus for the CSC-355 Natural Language Processing Design Project, Namal University Mianwali.

**Students:** M. Raqib Hayat (NUM-BSCS-2022-40) and Abu Bakar (NUM-BSCS-2022-41)<br>
**Instructor:** Dr. Muzamil Ahmed

| Course information | Value |
|---|---|
| Course | CSC-355 Natural Language Processing |
| Session / Semester | 2022-2026, 8th Semester |
| Total marks | 50 |
| Professor-provided submission date | May 20, 2026 |
| CLO coverage | CLO-2, CLO-3, CLO-4 |

## Quick Evidence

- Final report: [final_nlp_project_report.pdf](outputs/reports/final_nlp_project_report.pdf)
- Audit: [final_project_audit.md](docs/final_project_audit.md)
- Rubric checklist: [final_submission_checklist.md](docs/final_submission_checklist.md)
- Demo guide: [demonstration_guide.md](docs/demonstration_guide.md)
- Exact professor-brief mapping: [professor_requirements_alignment.md](docs/professor_requirements_alignment.md)
- Submission manifest: [submission_manifest.md](docs/submission_manifest.md)
- Source repository: [GitHub](https://github.com/abubakarp789/Robust-Sentiment-and-Emotion-Classification-on-Noisy-Urdu-Tweets)

## Professor Brief Alignment

| Required stage | Primary evidence | Status |
|---|---|---|
| 1. Problem identification and proposal | Final report, methodology, this README | Ready |
| 2. Literature review and gap analysis | Final report Related Work and references | Ready |
| 3. System design and methodology | Architecture figure, methodology docs, modular pipeline | Ready |
| 4. Implementation and experimental development | Source, data pipeline, training scripts, tests, saved predictions | Ready |
| 5. Evaluation, analysis, and optimization | Dual-task leaderboards, metrics, confusion matrices, error analysis | Ready |
| 6. Final report and demonstration | IEEE report, report copy, Streamlit baseline demo, speaking guide | Ready |

Every required deliverable named by the professor has a repository artifact. This means the submission is structurally aligned with the 50-mark rubric; it does not guarantee a particular awarded grade.

## Problem

Urdu tweets contain spelling variation, code-mixing, short context, sarcasm, URLs, mentions, hashtags, emojis, and weak labels. SentiUrdu-1M is also severely imbalanced. Its labels are influenced by emoji-based heuristics, so retaining emojis in model input creates a shortcut: a model can imitate the labeling rule instead of learning Urdu language evidence.

The project removes emojis before feature extraction, canonicalizes noisy emotion labels, derives a three-class sentiment target, and compares classical, neural, and Transformer model families under shared stratified splits and class-aware metrics.

## Why This Is a Complex Computational Problem

- The local CSV contains 1,048,000 tweets.
- 514,571 rows have no `Category` label.
- The raw category field has inconsistent and multi-label surface forms.
- Joy/Positive dominates while Surprise/Neutral is extremely rare.
- Leakage prevention conflicts with apparent predictive performance.
- Classical, neural, and Transformer methods have different cost, robustness, and deployment trade-offs.
- Accuracy, macro-F1, weighted-F1, minority recall, and compute cost are competing objectives.

## Objectives

1. Build deterministic Urdu tweet preprocessing with emoji leakage control.
2. Normalize labels into six emotions and map them into three sentiments.
3. Create fixed 70/15/15 stratified train, validation, and test splits.
4. Compare TF-IDF, neural, and Transformer approaches.
5. Evaluate with accuracy, macro precision/recall/F1, weighted-F1, per-class reports, confusion matrices, and qualitative errors.
6. Package the code, evidence, report, and live demo for reproducible inspection.

## Dataset

The included file is `data/raw/Urdu Tweets Dataset.csv` with columns `Id`, `Text`, `Emotions`, and `Category`.

Two verified experiment snapshots exist:

| Evidence source | Cleaning threshold | Rows used | Tasks |
|---|---:|---:|---|
| Assignment 4 report / Assignment 3 outputs | Remove empty cleaned text | 532,661 | Sentiment and emotion |
| Packaged `final_project` rerun | Minimum 2 cleaned tokens | 517,966 | Sentiment |

These snapshots must not be merged. The compiled report uses Assignment 3 results. The packaged outputs use the stricter two-token filter and a later sentiment rerun. See `docs/dataset_description.md` and `docs/results_analysis.md`.

The exact dual-task leaderboards used by the final report are also copied into `outputs/report_snapshot/` so the final project package contains direct sentiment and emotion evidence.

## Pipeline

1. Read the local CSV only.
2. Normalize Urdu/Arabic Unicode variants.
3. Remove URLs and mentions.
4. Remove `#` while preserving hashtag text.
5. Remove emojis before feature extraction.
6. Remove numbers and punctuation; normalize whitespace.
7. Parse noisy `Category` values and fix variants such as `Surprice`.
8. Produce six canonical emotions: Joy, Sad, Angry, Fear, Disgust, Surprise.
9. Map emotions to Positive, Negative, and Neutral sentiment.
10. Create fixed stratified 70/15/15 splits with seed 42.
11. Train model families and evaluate saved predictions.

## Models

| Family | Implemented models | Current artifact status |
|---|---|---|
| Classical | TF-IDF Logistic Regression, Linear SVM, Multinomial NB | Runnable `.joblib` files included |
| Neural | Text-CNN, BiLSTM with additive attention | Code, predictions, metrics, histories, vocab, and label map included; `.pt` weights are not included |
| Transformer | mBERT, XLM-R, Urdu-RoBERTa configuration | Code and Assignment 3 evidence for all three; packaged rerun predictions for mBERT/XLM-R; large weight files are not included |

The neural packaged rerun used random trainable embeddings because local Urdu fastText vectors were not present. Assignment 3 and the final report document fastText-based experiments. Training scripts never download embeddings or pretrained models automatically; required resources must already be available locally.

## Results

### Assignment 4 final report

| Task | Best macro-F1 model | Macro-F1 | Highest-accuracy model | Accuracy |
|---|---|---:|---|---:|
| Sentiment | Urdu-RoBERTa | 0.4573 | Linear SVM | 0.8783 |
| Emotion | mBERT | 0.2703 | Linear SVM | 0.8773 |

### Packaged sentiment rerun

**Best final model for the packaged runnable demo:** TF-IDF + Linear SVM.

| Model | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|
| Linear SVM | 0.8531 | 0.5040 | 0.8527 |
| Logistic Regression | 0.7740 | 0.4613 | 0.8013 |
| BiLSTM-Attention | 0.7408 | 0.4506 | 0.7763 |
| Text-CNN | 0.7582 | 0.4476 | 0.7882 |
| XLM-R | 0.8528 | 0.4346 | 0.8426 |
| mBERT | 0.8520 | 0.4240 | 0.8382 |
| Multinomial NB | 0.8787 | 0.4014 | 0.8417 |

Multinomial NB has the highest packaged accuracy but collapses more strongly toward the Positive majority. Linear SVM is selected because macro-F1 is the headline metric.

## Structure

```text
final_project/
|-- app/                 Streamlit demo
|-- data/                Raw, processed, split, and annotation data
|-- docs/                Submission audit, methodology, setup, results, and demo docs
|-- notebooks/           Grader-friendly artifact analysis notebooks
|-- outputs/             Models, predictions, metrics, figures, and report copy
|-- scripts/             Stable numbered command-line entry points
|-- src/                 Preprocessing, training, evaluation, inference, validation
|-- tests/               Lightweight and regression tests
|-- config.yaml          Canonical packaged sentiment configuration
`-- requirements.txt
```

The complete folder is the self-contained grading and evidence package. Earlier assignment folders are useful provenance but are not required to inspect it. Expensive deep-model retraining still requires the external pretrained resources described under Limitations.

## Setup

Python 3.10-3.12 is recommended for the pinned ML ecosystem. From `final_project`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No secrets are required. CUDA-specific PyTorch builds are intentionally not pinned.

## Commands

Run these from `final_project`:

```powershell
python scripts\01_prepare_data.py --config config.yaml
python scripts\02_train_classical.py --config config.yaml
python scripts\03_train_neural.py --config config.yaml --sample-size 50000
python scripts\04_train_transformers.py --config config.yaml --sample-size 50000 --epochs 1
python scripts\05_evaluate_models.py
python scripts\06_generate_visualizations.py
python -m pytest tests
python src\validate_final_project.py --config config.yaml
python src\validate_professor_requirements.py
streamlit run app\streamlit_app.py
```

The neural and Transformer sample modes shorten a demonstration run. They create new experimental outputs and must not be presented as the existing final-report results.

## Expected Outputs

- Splits: `data/splits/*.csv`
- Processed data: `data/processed/processed_sentiment_dataset.csv`
- Saved predictions: `outputs/predictions/`
- Recomputed metrics: `outputs/metrics/` (created on demand by the evaluation script)
- Existing experiment metrics: `outputs/results/`
- Figures: `outputs/figures/`
- Error analysis: `outputs/error_analysis/`
- Report copy: `outputs/reports/final_nlp_project_report.pdf`

## Demo

The most reliable live path uses the included Linear SVM artifact. Start Streamlit, enter an Urdu tweet containing an emoji, show that preprocessing removes the emoji, then compare the prediction with the model leaderboard and confusion matrix. Full speaking notes and likely questions are in `docs/demonstration_guide.md` and `docs/demo_script.md`.

## Why the Notebooks Are Included

The five notebooks are useful submission artifacts, not duplicate training code. They provide short, ordered walkthroughs of dataset preparation, baseline results, neural results, Transformer results, and error analysis. They load saved artifacts instead of retraining models, so the professor can inspect the experimental story quickly and reproducibly.

## Limitations

- Test labels are weak, not independently gold-annotated.
- Extreme class imbalance makes Neutral and rare emotions unreliable.
- No repeated-seed confidence intervals or significance tests are available.
- The final report and packaged rerun use different cleaning thresholds and result snapshots.
- Neural and Transformer weight files are omitted; their saved predictions and metrics remain inspectable.
- Packaged Transformer training requires locally cached pretrained models and does not download them automatically.

## Future Work

Create a native-speaker gold test set, run repeated seeds, add calibration and bootstrap confidence intervals, evaluate noise-robust losses, preserve task-specific split/output directories, and test domain-adapted Urdu/Twitter encoders using locally approved resources.

## Academic Integrity

All counts, results, hardware notes, and claims in this folder come from files already present in this repository. No new score, checkpoint, citation, dataset fact, or hardware specification has been invented. Items that could not be verified are marked as missing or not verified in the audit and checklist.

## Submission Uploads

The exact hand-in list, including the preferred full package and a size-limited alternative, is documented in [submission_manifest.md](docs/submission_manifest.md).
