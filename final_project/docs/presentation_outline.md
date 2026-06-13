# Presentation Outline

This 12-slide outline follows the Assignment 4 research report. The final slide sequence separates the report results from the later packaged sentiment demo.

## Slide 1: Title

- Robust Sentiment and Emotion Classification on Noisy Urdu Tweets
- Leak-Free Multi-Stage Evaluation on SentiUrdu-1M
- M. Raqib Hayat and Abu Bakar
- CSC-355 Natural Language Processing, Namal University Mianwali

## Slide 2: Problem and CCP Characteristics

- Urdu tweets are noisy, code-mixed, ambiguous, and weakly labeled.
- Local CSV size: 1,048,000 rows.
- Multiple competing objectives: accuracy, macro-F1, minority recall, leakage prevention, and compute cost.
- Alternative approaches: classical, neural, and Transformer models.

## Slide 3: Literature Gap

- Traditional Urdu studies often use small clean datasets.
- Neural and Transformer studies use inconsistent corpora, preprocessing, and metrics.
- Emoji-fusion work can exploit the same cues used to create weak labels.
- Gap: one shared, leak-aware comparison using class-balanced evaluation.

## Slide 4: Dataset and Labels

- 533,429 rows have a non-null raw `Category` before text cleaning.
- Canonical emotions: Joy, Sad, Angry, Fear, Disgust, Surprise.
- Sentiment mapping: Joy -> Positive; Surprise -> Neutral; remaining emotions -> Negative.
- Approximately 296:1 imbalance between the largest and smallest emotion classes.

Use the Assignment 4 emotion-distribution figure.

## Slide 5: Leakage-Aware Pipeline

- Unicode normalization.
- URL and mention removal.
- Hashtag text preservation.
- Emoji removal before feature extraction.
- Number, punctuation, and whitespace cleanup.
- Fixed 70/15/15 stratified split with seed 42.

Explain that emoji removal blocks a label-generation shortcut.

## Slide 6: System Architecture

- Shared preprocessing and label-canonicalization stages.
- Classical branch: TF-IDF + Logistic Regression / Linear SVM.
- Neural branch: Text-CNN / BiLSTM-Attention with Urdu fastText in the report experiment.
- Transformer branch: mBERT / XLM-R / Urdu-RoBERTa.
- Shared evaluation: accuracy, macro precision/recall/F1, weighted-F1, and confusion matrices.

Use the system-architecture figure embedded in `outputs/reports/final_nlp_project_report.pdf`.

## Slide 7: Experimental Setup

- Assignment 3/4 report snapshot: 532,661 cleaned labeled rows.
- Train: 372,862; validation: 79,899; test: 79,900.
- Training-only feature/vocabulary fitting.
- Class weighting and validation macro-F1 model selection.
- GPU environment and runtime details are documented in the report.

## Slide 8: Sentiment Results

- Highest accuracy: Linear SVM, 0.8783.
- Best macro-F1: Urdu-RoBERTa, 0.4573.
- Text-CNN, BiLSTM, and mBERT are close to the best macro-F1.
- Main lesson: accuracy and balanced class performance produce different rankings.

Use `outputs/report_snapshot/leaderboard_sentiment.csv` and `sentiment_f1_comparison.png`.

## Slide 9: Emotion Results

- Highest accuracy: Linear SVM, 0.8773.
- Best macro-F1: mBERT, 0.2703.
- Rare Angry, Fear, Disgust, and Surprise classes remain difficult.
- Model complexity cannot compensate fully for weak and scarce labels.

Use `outputs/report_snapshot/leaderboard_emotion.csv` and `emotion_f1_comparison.png`.

## Slide 10: Error Analysis and Optimization

- Common errors: negation, sarcasm, code-mixing, short context, poetic/religious language, and weak-label disagreement.
- Optimization used: class weights, early stopping, gradient clipping, mixed precision support, and smoothed Transformer weights.
- Macro-F1 and confusion matrices reveal minority-class failure hidden by accuracy.

## Slide 11: Live Demonstration

- Launch the Streamlit application.
- Enter an Urdu tweet containing an emoji.
- Show that preprocessing removes the emoji.
- Demonstrate the packaged Linear SVM prediction and confidence output.
- Explain that the packaged demo uses the later 517,966-row sentiment rerun; do not mix its metrics with the Assignment 4 tables.

## Slide 12: Conclusion and Future Work

- Main contribution: controlled leak-aware evaluation, not a single inflated accuracy score.
- Assignment 4 best models: Urdu-RoBERTa for sentiment macro-F1 and mBERT for emotion macro-F1.
- Limitations: weak labels, extreme imbalance, one seed, and no human gold test set.
- Future work: native-speaker annotation, repeated seeds, confidence intervals, calibration, and noise-robust learning.

End with the responsible-use warning: the system is for research and education, not high-stakes decisions.
