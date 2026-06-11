# Dataset Card: SentiUrdu-1M

## Dataset Name

SentiUrdu-1M

## Source

The dataset is a public Urdu tweet sentiment/emotion dataset. The final project keeps a self-contained copy at:

```text
final_project/data/raw/Urdu Tweets Dataset.csv
```

The Assignment 3 documentation links the dataset to Mendeley Data:

```text
https://data.mendeley.com/datasets/rz3xg97rm5/1
```

## Size

The CSV in the current folder contains:

- Total rows: 1,048,000
- Columns: 4
- Duplicate `Text` values observed during inspection: 31,304

## Columns

| Column | Description |
| --- | --- |
| `Id` | Tweet identifier |
| `Text` | Raw Urdu tweet text |
| `Emotions` | Emoji-derived emotion/confidence information |
| `Category` | Noisy raw emotion category field |

## Label Fields

Two label sources exist:

- `Category`: raw emotion label field with inconsistent formatting
- `Emotions`: emoji/confidence field that reflects weak supervision signals

The current project uses `Category` normalization for:

- 6-class emotion classification
- 3-class sentiment classification derived from canonical emotion labels

## Missing Labels

The `Category` column is missing for 514,571 rows. This leaves 533,429 rows with non-null `Category` values for the current supervised experiments.

## Label Normalization Issue

The raw `Category` field contains many inconsistent surface forms, including leading spaces, list-like strings, repeated labels, comma-separated labels, and misspellings such as `Surprice`.

The final project should normalize these into canonical labels:

- Joy
- Sad
- Angry
- Fear
- Disgust
- Surprise

## Weak Supervision Issue

The dataset labels are weakly supervised and influenced by emoji/lexicon heuristics. This creates two major risks:

- Labels may not always match the true meaning of the tweet text.
- If emojis are kept in the model input, models may learn the labeling shortcut instead of Urdu sentiment and emotion semantics.

Emoji removal is therefore treated as a leakage-prevention step.

## Class Imbalance Issue

The labelled subset is highly imbalanced. Assignment 3 analysis found Joy/Positive to dominate the dataset, while classes such as Fear, Surprise, and Angry are rare.

The final project should use:

- Stratified splitting
- Macro-F1 as the headline metric
- Per-class metrics
- Class weights or sampling strategies
- Minority-class error analysis

## Preprocessing Steps

The project uses an 8-step preprocessing pipeline:

1. Unicode normalization
2. URL removal
3. Mention removal
4. Hashtag cleanup
5. Emoji removal
6. Number removal
7. Punctuation removal
8. Whitespace normalization

The final project should preserve the same preprocessing behavior before changing or optimizing model logic.

The implemented final-project preprocessing module is:

```text
final_project/src/preprocessing.py
```

It is controlled by the `preprocessing` section of `config.yaml`. Emoji removal
is enabled by default to prevent models from learning the emoji-derived labeling
shortcut.

## Label Normalization

The implemented label-normalization module is:

```text
final_project/src/label_mapping.py
```

It normalizes raw `Category` values into canonical emotion labels and maps those
emotion labels to sentiment classes when `labels.task` is set to `sentiment`.

The pipeline writes a label summary to:

```text
final_project/outputs/results/label_mapping_summary.json
```

## Final Project Data Organization

- `data/raw/` contains the copied 1,048,000-row raw dataset used by `config.yaml`.
- `data/processed/` contains the 517,966-row cleaned labelled dataset before splitting.
- `data/splits/` contains the final stratified train, validation, and test splits used by every reported model.
- `data/annotation/` contains an optional 300-row balanced sample for future manual review.

The raw dataset is now copied into `data/raw/` so the `final_project` folder is self-contained for data loading. The annotation sample is not used in training or evaluation and does not affect reported results.

## Train/Validation/Test Split Plan

Recommended split:

- Train: 70 percent
- Validation: 15 percent
- Test: 15 percent
- Random seed: 42
- Stratification: enabled by label

The split files should be saved under:

```text
final_project/data/splits/
```

Generate and validate the splits with:

```powershell
cd final_project
python src\create_splits.py --config config.yaml
python src\create_annotation_sample.py --config config.yaml
python src\validate_pipeline.py --config config.yaml
python src\validate_final_project.py --config config.yaml
```

Expected split files:

```text
final_project/data/splits/train.csv
final_project/data/splits/validation.csv
final_project/data/splits/test.csv
```

Each split contains:

- `id`
- `raw_text`
- `clean_text`
- `raw_label`
- `normalized_label`
- `task_label`
- `text_length`

## Limitations

- Labels are weakly supervised, not fully human verified.
- The dataset is Twitter-domain specific and may not generalize to formal Urdu text.
- The class distribution is highly imbalanced.
- Tweets may contain offensive, sensitive, or personal content.
- Some rows have missing category labels.
- Emoji-derived labeling creates leakage risk.
- Without a clean manually verified test set, reported metrics reflect agreement with noisy labels rather than guaranteed true sentiment.
