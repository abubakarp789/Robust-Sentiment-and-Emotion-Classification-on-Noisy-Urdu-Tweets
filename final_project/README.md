# Leakage-Aware Urdu Tweet Sentiment and Emotion Classification

> [!IMPORTANT]
> **Best Final Model**: **TF-IDF + Linear SVM**, selected by test Macro-F1 (`0.5040`). 
> Under SentiUrdu-1M's emoji-stripped (leakage-free) pipeline and severe class imbalance, the classical max-margin classifier outperforms deep learning and transformer architectures trained under resource constraints.

## Overview

This final semester project converts the assignment-based work into a clean, research-oriented NLP repository. The project focuses on sentiment and emotion classification for noisy Urdu tweets using the SentiUrdu-1M dataset. It compares classical machine learning, neural models, and transformer-based models under a shared preprocessing, training, and evaluation pipeline.

The current assignments already provide a strong foundation: a preprocessing pipeline, literature review, dataset exploration, baseline models, neural models, transformer experiments, evaluation results, and a technical report. This `final_project/` directory organizes that work into a final deliverable structure without deleting or overwriting the original assignment folders.

## Problem Statement

Urdu social media text is difficult to classify because it contains informal spellings, right-to-left script issues, code-mixing, emojis, hashtags, mentions, URLs, sarcasm, and weakly supervised labels. SentiUrdu-1M is large enough for modern NLP modeling, but its labels are derived partly from emoji and lexical heuristics. If emojis remain in the input, models may learn the labeling shortcut rather than Urdu sentiment and emotion semantics.

The project therefore aims to build a leakage-aware pipeline for classifying Urdu tweets into sentiment and emotion categories while evaluating model behavior under class imbalance and weak-label noise.

## Motivation

Reliable Urdu sentiment analysis can support social-media monitoring, public-opinion research, product feedback analysis, crisis communication, and low-resource language NLP research. Most strong NLP tools are built for English and do not transfer cleanly to Urdu, especially on noisy Twitter-style text. A transparent, evaluated, and deployable Urdu pipeline helps reduce this language-resource gap.

## Dataset Description

- Dataset: SentiUrdu-1M
- Domain: Urdu tweets
- Size found in the current folder: 1,048,000 rows
- Source file in current assignment work: `Assignment#01/Urdu Tweets Dataset.csv`
- Columns: `Id`, `Text`, `Emotions`, `Category`
- Label issue: `Category` is missing for 514,571 rows
- Usable category-labelled rows: 533,429 rows
- Raw label issue: `Category` has many inconsistent surface forms and requires normalization
- Weak supervision issue: labels are derived from emoji/lexicon heuristics, so label noise and leakage risk must be handled carefully

See `reports/dataset_card.md` for the full dataset card.

## Milestone Mapping

| Course Milestone | Current Evidence | Final Project Status |
| --- | --- | --- |
| Milestone 1: Problem definition + dataset exploration | Assignment 1 proposal, preprocessing notebook, Assignment 3 EDA notebook | Completed & Verified |
| Milestone 2: Baseline statistical model | TF-IDF + Logistic Regression and Linear SVM in Assignment 3 | Completed, Packaged, & Evaluated |
| Milestone 3: Neural model | Text-CNN and BiLSTM-Attention in Assignment 3 | Completed, Packaged, & Evaluated |
| Milestone 4: Transformer + Generative AI | mBERT, XLM-R, Urdu-RoBERTa in Assignment 3 | Completed, Packaged, & Evaluated |
| Milestone 5: Evaluation + deployment + final report | Evaluation notebook, Streamlit app, and technical report | Fully Completed, Verified, & Submission-Ready |

## Methodology

1. Load SentiUrdu-1M from the raw data folder.
2. Normalize labels from the raw `Category` column.
3. Apply leak-aware Urdu tweet preprocessing.
4. Create reproducible train/validation/test splits.
5. Train statistical baselines using TF-IDF features.
6. Train neural models using Urdu word embeddings.
7. Fine-tune transformer encoders.
8. Evaluate with metrics suitable for imbalanced classification.
9. Perform qualitative and quantitative error analysis.
10. Deploy a simple interactive demo.

## Preprocessing Pipeline

The final-project preprocessing module is implemented in `src/preprocessing.py`.
It is controlled by `config.yaml` and currently supports:

- Urdu/Arabic Unicode normalization
- URL removal
- `@mention` removal
- Hashtag cleanup while preserving hashtag text
- Emoji removal for label-leakage prevention
- Western and Eastern Arabic-Indic number removal
- ASCII and Urdu/Arabic punctuation removal
- Whitespace normalization
- Minimum cleaned-text token filtering

Emoji removal is intentionally enabled by default because the dataset labels are
partly derived from emoji-based weak supervision. Keeping emojis in the model
input would allow models to learn the labeling heuristic instead of Urdu text
semantics.

## Label Normalization

The label module is implemented in `src/label_mapping.py`. It converts noisy
raw `Category` values such as `" Joy"`, `"['Joy']"`, `"Joy , Joy"`, and
`"Surprice"` into canonical emotion labels:

- Joy
- Sad
- Angry
- Fear
- Disgust
- Surprise

For the sentiment task, canonical emotions are mapped as:

- Joy -> Positive
- Sad, Angry, Fear, Disgust -> Negative
- Surprise -> Neutral

The split-generation script writes the label mapping audit to
`outputs/results/label_mapping_summary.json`.

## Train/Validation/Test Split Strategy

The split pipeline is implemented in `src/create_splits.py`. It:

1. Loads the dataset path from `config.yaml`.
2. Reads the raw CSV with UTF-8 encoding.
3. Applies preprocessing.
4. Normalizes labels.
5. Removes rows with missing task labels.
6. Removes empty or too-short cleaned tweets.
7. Creates stratified 70/15/15 train/validation/test splits.
8. Saves CSV files under `data/splits/`.
9. Saves `outputs/results/split_summary.json`.

Run it with:

```bash
cd final_project
python src/create_splits.py
python src/validate_pipeline.py
```

## Models Used

Current Assignment 3 work already includes:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Text-CNN
- BiLSTM with attention
- mBERT
- XLM-RoBERTa
- Urdu-RoBERTa

The final project implements this model lineup across course milestones, keeping all original assignment folders untouched:

## Milestone 2: Statistical Baselines
- Implemented in `src/train_baseline.py` and evaluated in `src/evaluate.py`.
- Trains TF-IDF + Logistic Regression, TF-IDF + Linear SVM, and Multinomial Naive Bayes.
- Reuses a single training-only TF-IDF vocabulary fit to prevent leakage.
- Validation script: `src/validate_baseline.py`.

## Milestone 3: Neural Models
- Implemented in `src/train_neural.py` with architectures in `src/models_dl.py` and utilities in `src/neural_utils.py`.
- Trains a multi-kernel Text-CNN and a BiLSTM with additive attention.
- Uses validation macro-F1 early stopping and PyTorch/CUDA mixed precision (`fp16`).
- Validation script: `src/validate_neural.py`.

## Milestone 4: Transformer-Based Modeling
- Implemented in `src/train_transformer.py`.
- Fine-tunes multilingual pre-trained encoders `bert-base-multilingual-cased` (mBERT) and `xlm-roberta-base` (XLM-RoBERTa).
- Addresses class imbalance using smoothed class weights (`class_weight_smoothing = 0.5`) to prevent gradient instability from the rare Neutral class.
- Saves model checkpoints, classification reports, confusion heatmaps, prediction tables, and training histories.
- Validation script: `src/validate_transformer.py`.

## Generative AI / Explanation Assistant
- Implemented in `src/explanation_assistant.py`.
- A lightweight, rule-based and template-driven assistant that explains model predictions, highlights likely causes of misclassification errors, generates plain-English model performance summaries, and outputs project-level model insights.
- Integrates directly with the interactive user interface to explain model outputs in real time.

## Evaluation Metrics

The final evaluation should report:

- Accuracy
- Macro precision, recall, and F1
- Weighted precision, recall, and F1
- Per-class precision, recall, and F1
- Confusion matrices
- Misclassified examples
- Class-wise and error-type analysis

Macro-F1 should be the headline metric because the dataset is extremely imbalanced.

## Error Analysis Plan

The project should save a structured error-analysis table containing:

- Raw tweet text
- Preprocessed text
- True label
- Predicted label
- Confidence score, if available
- Text length
- Model name
- Error category
- Human-readable explanation

Planned error categories include negation, sarcasm, code-mixing, ambiguous text, weak-label issue, minority-class confusion, and preprocessing loss.

## GenAI / Agentic AI Extension

The most suitable extension is an error-analysis and explanation assistant. It does not replace the classifier. Instead, it helps interpret model predictions by producing short human-readable explanations, identifying likely error types, and summarizing failure patterns across misclassified examples.

This extension has been successfully implemented in `src/explanation_assistant.py` and integrated into the Streamlit application for real-time inference analysis.

## Ethical Considerations

Key ethical issues include weak labels, dataset bias, privacy risk from social-media text, exposure to offensive content, minority-class harm, and misuse of sentiment predictions for surveillance or unfair decision-making.

See `reports/ethics_and_limitations.md`.

## Deployment Plan

The recommended deployment is a Streamlit app because it is simple, course-demo friendly, and well-suited for showing:

- Tweet input
- Preprocessed output
- Model selection
- Prediction result
- Confidence score
- Explanation
- Error-analysis note

The app has been fully implemented in `app/streamlit_app.py` and loads trained model checkpoints with robust fallbacks to the best overall model (Linear SVM) if transformer files are missing.

## Folder Structure

```text
final_project/
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- splits/
|   `-- annotation/
|-- notebooks/
|   |-- 01_dataset_analysis.ipynb
|   |-- 02_baseline_models.ipynb
|   |-- 03_neural_models.ipynb
|   |-- 04_transformer_models.ipynb
|   `-- 05_error_analysis.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- label_mapping.py
|   |-- train_baseline.py
|   |-- train_neural.py
|   |-- train_transformer.py
|   |-- evaluate.py
|   |-- error_analysis.py
|   |-- inference.py
|   `-- utils.py
|-- app/
|   `-- streamlit_app.py
|-- outputs/
|   |-- figures/
|   |-- results/
|   |-- models/
|   |-- predictions/
|   `-- error_analysis/
|-- reports/
|   |-- final_report.md
|   |-- dataset_card.md
|   |-- ethics_and_limitations.md
|   `-- slides_outline.md
|-- README.md
|-- requirements.txt
|-- config.yaml
`-- ROADMAP.md
```

## Installation

```bash
cd final_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install the correct PyTorch build for your GPU separately if you plan to retrain neural or transformer models.

## Usage

The preprocessing, split generation, baseline, neural, and transformer modeling pipelines are fully implemented.

### Data Pipeline
```bash
python src/create_splits.py
python src/validate_pipeline.py
```

### Milestone 2: Baselines
```bash
python src/train_baseline.py --config config.yaml
python src/validate_baseline.py --config config.yaml
python src/analyze_baseline_errors.py --config config.yaml
python src/plot_baseline_errors.py --config config.yaml
python src/validate_error_analysis.py --config config.yaml
```

### Milestone 3: Neural Models
```bash
python src/train_neural.py --config config.yaml
python src/compare_models.py --config config.yaml
python src/plot_neural_results.py --config config.yaml
python src/validate_neural.py --config config.yaml
```

### Milestone 4: Transformer Models & Explanations
```bash
python src/train_transformer.py --config config.yaml
python src/compare_models.py --config config.yaml
python src/plot_transformer_results.py --config config.yaml
python src/validate_transformer.py --config config.yaml
```

### Streamlit Application
To run the interactive web application:
```bash
streamlit run app/streamlit_app.py
```

## Reproducibility

The final project should use:

- Fixed random seed
- Saved train/validation/test splits
- Versioned configuration
- Saved predictions
- Saved model checkpoints
- Saved metric tables
- Documented preprocessing pipeline

## Future Work

- Build a manually verified clean test subset
- Add emoji-removal ablation study
- Add model confidence and calibration analysis
- Add Hugging Face Spaces or Streamlit Cloud deployment
- Explore XLM-T or parameter-efficient LoRA fine-tuning

## Author

Abu Bakar and M. Raqib Hayat  
CSC-355 Natural Language Processing  
Namal University Mianwali
