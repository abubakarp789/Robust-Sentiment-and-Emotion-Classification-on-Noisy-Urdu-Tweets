# Methodology

## End-to-End Pipeline

1. Load the repository-local SentiUrdu-1M CSV.
2. Normalize noisy raw category values.
3. Clean tweet text deterministically.
4. Remove rows without a usable label or sufficient cleaned text.
5. Create fixed stratified train, validation, and test splits.
6. Fit each representation and model only on training data.
7. Select configurations using validation macro-F1.
8. Evaluate once on saved test rows.
9. Save predictions, aggregate/per-class metrics, confusion matrices, and error samples.

## Preprocessing

`src/preprocessing.py` performs NFC normalization and common Urdu/Arabic character canonicalization, URL removal, mention removal, hashtag-symbol removal while preserving hashtag text, emoji removal, Western and Arabic-Indic digit removal, ASCII and Urdu punctuation removal, and whitespace normalization.

Stemming, lemmatization, and blanket stop-word removal are not used. They were not supported by a verified Urdu tool in the repository and may remove function words needed by contextual models.

## Leakage Prevention

SentiUrdu-1M labels were produced partly through emoji and lexical heuristics. If emojis remain in input, a classifier can learn a direct label cue. The pipeline therefore removes emojis before TF-IDF, embedding lookup, or subword tokenization. This generally lowers apparent performance but makes the experiment more faithful to text-only language understanding.

## Label Canonicalization

`src/label_mapping.py` parses leading spaces, list-like cells, repeated labels, comma-separated labels, and known variants such as `Sadness`, `Anger`, and `Surprice`.

Canonical emotion classes are Joy, Sad, Angry, Fear, Disgust, and Surprise. Multi-label cells are reduced by majority count with a deterministic tie order.

## Sentiment Mapping

- Joy -> Positive
- Sad, Angry, Fear, Disgust -> Negative
- Surprise -> Neutral

This mapping makes Neutral exceptionally rare and is a project design decision, not a universal linguistic truth.

## Model Branches

### Classical

TF-IDF word unigrams/bigrams feed Logistic Regression, Linear SVM, and Multinomial Naive Bayes. The vectorizer is fitted on training text only. Linear models use balanced class weights where supported.

### Neural

Text-CNN captures local n-gram patterns with multiple convolution widths. BiLSTM-Attention models bidirectional sequence context and learns a weighted sentence representation. The scripts support local pretrained embeddings, but the packaged rerun metadata records random trainable embeddings because no fastText file was present.

### Transformers

The code supports mBERT, XLM-R, and Urdu-RoBERTa configurations with dynamic padding, class-weighted cross-entropy, validation macro-F1 selection, early stopping, and optional mixed precision. Pretrained models must already be local/cached; the scripts do not download resources automatically as part of this submission.

## Design Decisions

- Macro-F1 is primary because accuracy can be dominated by Joy/Positive.
- Weighted-F1 and accuracy remain visible to quantify majority-class performance.
- One fixed split supports fair within-snapshot comparison.
- Saved predictions allow evaluation without retraining expensive models.
- The baseline model is the live-demo default because its full artifact is included.

## Alternatives Considered

- Keeping emojis: rejected for the main experiment because of leakage.
- Character TF-IDF: reasonable for spelling variation but not part of verified final outputs.
- Focal loss/balanced sampling: proposed future improvements, not verified experiments.
- Gold-label evaluation: preferable but missing because the annotation sample is not completed.
- External clean benchmarks and XLM-T: discussed in the literature/future work, not run locally.

## Snapshot Warning

The Assignment 4 report and packaged rerun share the methodology but not identical cleaning thresholds or outputs. See `dataset_description.md` and `results_analysis.md` before quoting numbers.
