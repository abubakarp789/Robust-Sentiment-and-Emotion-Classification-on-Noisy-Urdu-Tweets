# Live Demo Script

## One-Minute Introduction

"Our project studies sentiment and emotion classification for noisy Urdu tweets using SentiUrdu-1M. The local CSV has 1.048 million tweets, but more than half have no category label and the remaining labels are weak and highly imbalanced. The central technical risk is label leakage: emojis helped produce the labels, so a model that sees those emojis may learn the labeling shortcut. We therefore remove emojis before training, normalize noisy labels into six emotions, derive three sentiment classes, and compare classical, neural, and Transformer approaches using macro-F1 as the main metric."

## Pipeline Explanation

"The pipeline removes URLs and mentions, keeps hashtag words, removes emojis, numbers, and punctuation, normalizes Urdu Unicode, maps variants such as `Surprice` to `Surprise`, and creates a fixed stratified 70/15/15 split with seed 42. TF-IDF features feed Logistic Regression and SVM; neural branches use Text-CNN and BiLSTM-Attention; Transformer branches use mBERT, XLM-R, and Urdu-RoBERTa in the Assignment 3 report evidence."

## Results

"In the Assignment 4 dual-task report, Urdu-RoBERTa has the best sentiment macro-F1 at 0.4573 and mBERT has the best emotion macro-F1 at 0.2703. Linear SVM has the highest accuracy but weaker balanced performance. In the later packaged sentiment rerun with a stricter two-token filter, Linear SVM is the best macro-F1 model at 0.5040, while Multinomial NB has higher accuracy but zero Neutral F1."

## Files and Commands

Show `src/preprocessing.py`, `src/label_mapping.py`, `outputs/results/split_summary.json`, the model leaderboard, and the Assignment 4 PDF. Then run:

```powershell
python scripts\05_evaluate_models.py --pattern "baseline_linear_svm_test_predictions.csv"
streamlit run app\streamlit_app.py
```

Enter an Urdu tweet containing an emoji and point out that the displayed cleaned text has no emoji.

## Limitations and Future Work

"The labels are weak, there is no completed human gold test set, results use one seed, and the deep model weights are not included in the repository. The next step is native-speaker annotation, repeated-seed evaluation, confidence intervals, and noise-robust learning."

## Short Q&A

- Emoji removal matters because it blocks a label-generation shortcut.
- Macro-F1 matters because it exposes minority-class failure.
- Classical models remain competitive because tweets are short and lexical signals are strong.
- Transformers help context but cannot repair noisy or scarce labels alone.
- The two result snapshots are documented separately and must not be combined.
