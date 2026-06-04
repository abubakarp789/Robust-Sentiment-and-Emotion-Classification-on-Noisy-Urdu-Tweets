# Assignment 3 Review and Viva Prep

## Overall Verdict

The assignment is substantially complete and stronger than a minimum submission. It covers all five professor tasks, includes a full EDA notebook, architecture/math write-up, reusable implementation modules, trained models, saved predictions, evaluation plots, a technical report, an AI declaration, and a similarity-report template.

It is not "perfect" yet. The main marks risk is not missing code; it is documentation/reproducibility polish and explaining the weak results correctly in viva.

Estimated score if submitted as-is, assuming similarity report is accepted: 25-27 / 30.
Estimated score after the fixes below and good viva preparation: 27-30 / 30.

## Rubric Audit

| Rubric item | Marks | Current status | Risk |
|---|---:|---|---|
| Dataset statistical analysis and visualizations | 5 | Strong: missingness, label distribution, length histograms, token frequency, Zipf, word clouds, noise, emoji leakage, preprocessing impact | Low |
| Architecture explanation and components | 5 | Strong: pipeline modules, TF-IDF, CNN, BiLSTM, transformers, classification/evaluation modules | Low |
| Mathematical modeling | 5 | Strong: TF-IDF, softmax, SVM, CNN, LSTM gates, attention, transformer attention, loss, AdamW | Low |
| Implementation and experimentation | 7 | Strong model lineup across classical, DL, transformers; saved models and predictions exist | Medium |
| Evaluation and analysis | 4 | Metrics are reproducible and plots exist; rare-class failure is correctly surfaced with macro-F1 | Low-Medium |
| Technical report formatting | 2 | PDF/DOCX/LaTeX/Markdown exist and are coherent | Low |
| Similarity report, AI declaration, documentation | 2 | AI declaration is complete; similarity report still has placeholder values | High until filled |

## Findings to Fix Before Submission

1. Fill the actual similarity/plagiarism percentages in `report/Similarity_Report.md`.
   Do not submit it with `_to be filled_` values. Add the Turnitin screenshot or exported PDF if your LMS provides one.

2. Rerun CNN/BiLSTM if time allows after the trainer fix.
   `train_dl.py` now monitors validation macro-F1 instead of validation accuracy. This is the correct objective for the imbalanced classes. Rerunning may improve rare-class macro-F1 and will make the saved history JSON include `val_macro_f1`.

3. Regenerate evaluation figures after rerunning deep models.
   Open `02_model_implementation.ipynb`, rerun the CNN/BiLSTM cells, then rerun `03_evaluation.ipynb`.

4. Explain accuracy vs macro-F1 clearly.
   Linear SVM has the best accuracy but bad macro-F1 because it nearly collapses to Joy/Positive. The report already explains this; memorize it for viva.

5. Keep the dataset limitation honest.
   Labels are weakly supervised from emoji heuristics, so this is not a gold-standard human-labelled emotion dataset.

## Result Improvement Plan

Best low-risk improvements:

1. Macro-F1 checkpointing for CNN/BiLSTM.
   Already fixed in `train_dl.py`. Rerun only the deep-learning section if GPU time is limited.

2. Oversampling or weighted sampler for rare classes.
   Add `WeightedRandomSampler` for CNN/BiLSTM training. This may help Fear, Surprise, Angry recall, but can lower overall accuracy.

3. Focal loss experiment.
   Replace cross-entropy with focal loss for deep models:
   `FL = -alpha_t * (1 - p_t)^gamma * log(p_t)`.
   Use `gamma=2`. This targets easy-majority-class dominance.

4. Character n-gram baseline.
   Urdu morphology and spelling variation often benefit from character 3-5 grams. Add a `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))` baseline or combine word + char features.

5. Merge rare classes only as an ablation, not as the main task.
   You can show that Fear/Surprise are statistically too tiny, but do not change the official 6-class task unless the professor allows it.

6. Add a small manual error-analysis table.
   Pick 10 misclassified tweets from `03_evaluation.ipynb`, show true label, predicted label, and reason: negation, sarcasm, code-mixing, label noise, or rare-class ambiguity.

## Viva Answers

### What is your project?

It is Urdu tweet sentiment and emotion classification on SentiUrdu-1M. We train and compare seven models: Logistic Regression, Linear SVM, Text-CNN, BiLSTM with attention, mBERT, XLM-R, and Urdu-RoBERTa.

### Why did you remove emojis?

The labels in SentiUrdu-1M are derived from emoji co-occurrence. If emojis remain in the input, the model can learn the labelling shortcut instead of Urdu text meaning. Removing emojis prevents label leakage.

### Why do you use macro-F1 as the headline metric?

The dataset is extremely imbalanced. Joy/Positive is about 86% of labelled rows, while Fear and Surprise are below 1%. Accuracy hides rare-class failure; macro-F1 gives each class equal weight.

### Why does Linear SVM have high accuracy but low macro-F1?

It predicts the majority class too often. That gives high accuracy because most examples are Joy/Positive, but rare classes get poor recall, so macro-F1 drops.

### Which model is best?

For sentiment, Urdu-RoBERTa has the best macro-F1: 0.4573. For emotion, mBERT has the best macro-F1: 0.2703. Linear SVM has the best accuracy, but it is not the best model under macro-F1.

### Why are the macro-F1 scores low?

The labels are weakly supervised, the classes are highly imbalanced, and rare labels like Fear and Surprise have very few examples. After emoji removal, the easy shortcut disappears, so models must rely on noisy Urdu text.

### What preprocessing steps did you apply?

Unicode normalization, URL removal, mention removal, hashtag cleanup, emoji removal, digit removal, punctuation removal, and whitespace normalization.

### What is TF-IDF?

TF-IDF weights a token by how frequent it is in a document and how rare it is across the corpus. Common words get lower weight; class-informative words get higher weight.

### What is attention in your BiLSTM?

The BiLSTM creates a hidden state for each token. Additive attention learns weights over those hidden states and combines the important ones into one sentence vector for classification.

### What is self-attention in transformers?

Self-attention compares each token with every other token using query, key, and value vectors. The softmax of scaled dot products decides how much context each token should receive.

### Why compare classical, deep, and transformer models?

Classical models give a baseline, CNN/BiLSTM test neural sequence models with Urdu fastText embeddings, and transformers test contextual subword representations. The comparison is fair because all use the same preprocessing, splits, and metrics.

### What is the biggest limitation?

The labels are not human gold labels. They come from emoji-based weak supervision, so the dataset contains label noise and the results should be interpreted as performance on noisy weak labels.

## One-Minute Viva Summary

Our assignment implements a leak-free Urdu sentiment and emotion classifier on SentiUrdu-1M. The key decision is emoji removal because the dataset labels are emoji-derived, so keeping emojis would cause label leakage. We canonicalize the noisy `Category` column into six emotions and derive three sentiment classes. Then we compare seven models under the same 70/15/15 stratified split: LR, SVM, CNN, BiLSTM-attention, mBERT, XLM-R, and Urdu-RoBERTa. We report accuracy, precision, recall, macro-F1, weighted-F1, and confusion matrices. Because the dataset is extremely imbalanced, macro-F1 is the main metric. Linear SVM gets the highest accuracy but collapses toward the majority class, while Urdu-RoBERTa is best for sentiment macro-F1 and mBERT is best for emotion macro-F1. The main limitation is weak emoji-based labelling and poor rare-class support.

