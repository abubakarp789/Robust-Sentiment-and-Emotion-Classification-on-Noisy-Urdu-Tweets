# Source Analysis for Assignment 4

## Inspection Boundary

Only `Assignment#01`, `Assignment#02`, and `Assignment#03` were analyzed. The `final_project` folder was not opened, read, modified, moved, renamed, or otherwise touched. The earlier assignment folders were not modified.

The inspection covered DOCX and PDF reports, Markdown and LaTeX sources, four notebooks and their stored outputs, Python modules, configuration files, the CSV/XLSX dataset files, 20 literature-review PDFs (including one duplicate file), saved prediction CSVs, leaderboards, JSON summaries, and result figures.

## Extracted Project Identity

**Title:** Robust Sentiment and Emotion Classification on Noisy Urdu Tweets: A Multi-Stage NLP Pipeline on SentiUrdu-1M

**Authors:** M. Raqib Hayat (NUM-BSCS-2022-40) and Abu Bakar (NUM-BSCS-2022-41)

**Institution/course:** Department of Computer Science, Namal University Mianwali; CSC-355 Natural Language Processing; instructor Dr. Muzamil Ahmed.

## What Assignment 1 Contributed

Assignment 1 established the research proposal, motivation, dataset choice, complexity argument, expected outputs, and the first executable preprocessing notebook.

Key contributions:

- Defined Urdu social-media sentiment and emotion classification as a complex computing problem involving weak labels, script variation, code-mixing, noisy tweets, imbalance, and large-scale processing.
- Selected SentiUrdu-1M and documented its public Mendeley source.
- Proposed comparison of classical, neural, and transformer model families.
- Designed an eight-stage preprocessing pipeline: NFC Unicode normalization; URL removal; mention removal; hashtag-symbol cleanup while retaining content; emoji removal; Western and Arabic-Indic digit removal; punctuation removal; whitespace normalization.
- Identified emoji removal as a leakage-control requirement because emojis contributed to weak label generation.
- Supplied sample preprocessing code and notebook outputs.

Proposal-only outcomes not found as completed artifacts later:

- A manually verified 500-1,000 tweet gold test set.
- A deployed Gradio/Hugging Face demo with token-level saliency.
- A separate external clean Urdu benchmark evaluation.

## What Assignment 2 Contributed

Assignment 2 supplied the research foundation and an IEEE-style list of 20 references. Its review grouped prior work into traditional machine learning, deep learning, and transformer/recent methods.

Key findings:

- Classical Urdu systems commonly used lexicons, bag-of-words, TF-IDF, Naive Bayes, SVM, decision trees, or Markov chains on small corpora.
- CNN, LSTM, BiLSTM, and hybrid systems improved contextual modeling but were generally evaluated on modest, cleaner datasets.
- mBERT/XLM-R and domain-adapted multilingual transformers generally improved results, but reported scores varied sharply with dataset cleanliness and domain.
- Emoji-fusion work on SentiUrdu-1M reported very high accuracy, but Assignment 2 correctly flagged that emoji-derived labels can make such input a leakage shortcut.
- The review identified three central gaps: lack of million-scale fair comparison, inconsistent preprocessing/splits/metrics, and weak disclosure or control of emoji leakage.

The final report uses the bibliographic details in Assignment 2. No DOI or publication detail was added unless it was already present in the assignment sources.

## What Assignment 3 Contributed

Assignment 3 turned the proposal into an implemented and evaluated pipeline.

Implemented components:

- Reusable preprocessing and label canonicalization in `preprocessing.py`.
- Central configuration and fixed seed (`42`) in `config.py`.
- Text-CNN and two-layer BiLSTM with additive attention in `models_dl.py`.
- fastText loading, weighted training, and early stopping in `train_dl.py`.
- Hugging Face weighted fine-tuning for mBERT, XLM-R, and Urdu-RoBERTa in `train_transformer.py`.
- EDA, model implementation, and evaluation notebooks.
- Saved predictions for seven models on both tasks, two leaderboards, confusion matrices, F1 charts, and training curves.

Model families:

1. Classical: TF-IDF + Logistic Regression; TF-IDF + Linear SVM.
2. Neural: Text-CNN; BiLSTM with additive attention, initialized with 300-dimensional Urdu fastText vectors.
3. Transformers: `bert-base-multilingual-cased`, `xlm-roberta-base`, and `urduhack/roberta-urdu-small`.

Training protocol:

- Same stratified 70/15/15 train/validation/test design for all models.
- Random seed 42.
- Inverse-frequency class weighting.
- Headline metric: macro-F1, supported by accuracy, macro precision/recall, weighted-F1, and confusion matrices.

## Dataset Details Extracted

Local CSV facts independently checked from `Assignment#01/Urdu Tweets Dataset.csv`:

- Rows: 1,048,000.
- Columns: `Id`, `Text`, `Emotions`, `Category`.
- Missing `Category`: 514,571.
- Rows with a non-null `Category`: 533,429.
- Raw category forms: 298 non-null surface forms (299 values if missing is counted).

After preprocessing, 1,172 tweets become empty and are removed. The executed modeling dataset therefore contains 532,661 labeled examples, split into 372,862 training, 79,899 validation, and 79,900 test rows for each task.

Canonical emotion distribution from Assignment 3:

| Emotion | Count | Approx. share |
|---|---:|---:|
| Joy | 459,728 | 86.2% |
| Sad | 50,417 | 9.5% |
| Disgust | 17,083 | 3.2% |
| Angry | 2,802 | 0.5% |
| Fear | 1,847 | 0.3% |
| Surprise | 1,552 | 0.3% |

Derived sentiment mapping:

- Joy -> Positive.
- Sad, Angry, Fear, Disgust -> Negative.
- Surprise -> Neutral.

Derived sentiment counts are Positive 459,728; Negative 72,149; Neutral 1,552. The maximum/minimum class ratio is approximately 296:1.

The source paper reports 1,140,821 collected tweets. The XLSX workbook contains a first sheet with 1,048,000 data rows plus a header and a second sheet with 92,823 data rows plus a header. The project experiments explicitly use the 1,048,000-row CSV, so the final report does not merge or reinterpret the second sheet.

## Extracted Results

### Three-class sentiment

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.6598 | 0.3852 | 0.7107 |
| Linear SVM | 0.8783 | 0.4004 | 0.8409 |
| Text-CNN | 0.7848 | 0.4533 | 0.8117 |
| BiLSTM-Attention | 0.7762 | 0.4500 | 0.8040 |
| mBERT | 0.8054 | 0.4526 | 0.8217 |
| XLM-R | 0.7897 | 0.4475 | 0.8118 |
| Urdu-RoBERTa | 0.7750 | **0.4573** | 0.8011 |

### Six-class emotion

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.3601 | 0.1632 | 0.4959 |
| Linear SVM | 0.8773 | 0.2087 | 0.8347 |
| Text-CNN | 0.6476 | 0.2499 | 0.7215 |
| BiLSTM-Attention | 0.6064 | 0.2428 | 0.6910 |
| mBERT | 0.6922 | **0.2703** | 0.7467 |
| XLM-R | 0.6907 | 0.2535 | 0.7462 |
| Urdu-RoBERTa | 0.6153 | 0.2539 | 0.6971 |

Interpretation supported by the saved outputs:

- Linear SVM has the highest accuracy but a much lower macro-F1, reflecting majority-class behavior.
- Urdu-RoBERTa has the best sentiment macro-F1.
- mBERT has the best emotion macro-F1.
- Rare emotion recall remains poor for every model.
- The Assignment 3 error analysis identifies negation, sarcasm, code-mixing, religious/poetic language, weak-label noise, and class-prior collapse as recurring failure modes.

## Missing Information and TODOs

1. **Gold test set:** No manually verified 500-1,000 tweet test set was found. All reported test labels inherit the weak-label process.
2. **Repeated trials:** Results are from a fixed seed; no multi-seed mean, standard deviation, confidence interval, or significance test was found.
3. **Deployment:** No final demo or saliency interface was found.
4. **Model artifacts:** Saved prediction files and figures exist, but `outputs/models`, cached splits, and embedding artifacts were not present in the inspected Assignment 3 tree. Re-training reproducibility therefore depends on the code and external model/vector downloads.
5. **Hardware verification:** RTX 5070 Ti, CUDA 12.8, driver 581.57, and reported runtimes are documented in Assignment 3 but were not independently benchmarked during report preparation.
6. **Author email:** Not present. The IEEE author block uses names, roll numbers, department, and institution only.
7. **Submission extras:** Confirm whether Assignment 4 also requires plagiarism/similarity and AI-use reports.
8. **Reference review:** Assignment 2 lists the stacked CNN-BiLSTM paper as 2025 in its final reference; the included PDF also identifies the journal issue as 2025. That year is used in the final bibliography.
