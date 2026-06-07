# Leakage-Aware Urdu Tweet Sentiment and Emotion Classification

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
| Milestone 1: Problem definition + dataset exploration | Assignment 1 proposal, preprocessing notebook, Assignment 3 EDA notebook | Mostly complete |
| Milestone 2: Baseline statistical model | TF-IDF + Logistic Regression and Linear SVM in Assignment 3 | Implemented, needs final packaging |
| Milestone 3: Neural model | Text-CNN and BiLSTM-Attention in Assignment 3 | Implemented, needs checkpoint/artifact organization |
| Milestone 4: Transformer + Generative AI | mBERT, XLM-R, Urdu-RoBERTa in Assignment 3 | Transformer complete; GenAI extension pending |
| Milestone 5: Evaluation + deployment + final report | Evaluation notebook and technical report in Assignment 3 | Evaluation partly complete; deployment pending |

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

The final project will keep this model lineup and reorganize it into scripts and reusable modules. Model logic should not be changed until the repository structure is stable.

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

The most suitable extension is an error-analysis and explanation assistant. It should not replace the classifier. Instead, it should help interpret model predictions by producing short human-readable explanations, identifying likely error types, and summarizing failure patterns across misclassified examples.

This extension can be added after core inference and evaluation artifacts are stable.

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

The current app is only a placeholder and does not load trained models yet.

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

The current structure is prepared for final development. The preprocessing,
label-normalization, and split-generation pipeline is implemented. Model
training and final inference are not implemented in this folder yet.

Data pipeline commands:

```bash
python src/create_splits.py
python src/validate_pipeline.py
```

Planned model commands:

```bash
python src/train_baseline.py --config config.yaml
python src/train_neural.py --config config.yaml
python src/train_transformer.py --config config.yaml
python src/evaluate.py --config config.yaml
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
- Add GenAI explanation assistant
- Add Hugging Face Spaces or Streamlit Cloud deployment
- Explore XLM-T or parameter-efficient LoRA fine-tuning

## Author

Abu Bakar and M. Raqib Hayat  
CSC-355 Natural Language Processing  
Namal University Mianwali
