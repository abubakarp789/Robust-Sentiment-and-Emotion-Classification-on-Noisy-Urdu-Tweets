# Group-Safe Urdu Tweet Sentiment and Emotion Classification

## Abstract

This project presents a reproducible comparison of classical machine-learning, neural-network, and Transformer approaches for sentiment and emotion classification on noisy Urdu tweets. SentiUrdu-1M labels are weakly supervised and partly derived from emoji signals, so emojis are removed from model input and duplicate-linked tweet IDs and normalized texts are assigned to only one split. Conflicting duplicate-label groups are excluded. The official benchmark evaluates eight models on separate three-class sentiment and six-class emotion pipelines. Classical and neural models are repeated with seeds 42, 52, and 62; resource-constrained Transformers use seed 42, 50,000 training rows, and one epoch. Models are ranked only by mean validation macro-F1. Linear SVM is selected for both tasks, reaching test macro-F1 **0.4590** for sentiment and **0.2854** for emotion. No human-gold evaluation is claimed because independent Urdu annotators were unavailable.

## Introduction

Urdu social-media classification is difficult because tweets mix Urdu script, Roman Urdu, English, informal spelling, hashtags, sarcasm, negation, and limited context. The corpus is also severely imbalanced and weakly labeled. These properties make accuracy alone unreliable and create a complex computational problem requiring data-quality controls, competing model families, resource-aware optimization, and class-balanced evaluation.

The project asks two research questions: (1) which model family provides the strongest class-balanced sentiment performance after duplicate-safe evaluation, and (2) how much harder is six-class emotion classification under the same preprocessing and split policy? Macro-F1 is the primary metric; accuracy, weighted-F1, per-class results, confusion matrices, runtime, and uncertainty are supporting evidence.

## Related Work

Prior Urdu sentiment research spans TF-IDF classifiers, lexicon and rule-based systems, CNN/LSTM architectures, multilingual encoders, Urdu-specific pretraining, and multimodal approaches. Three persistent gaps motivate this implementation: weak-label shortcuts are often not controlled, experiments frequently use incomparable preprocessing/splits, and accuracy obscures rare-class failure. This project addresses those gaps through emoji removal, connected duplicate grouping, shared task-specific splits, repeated seeds, validation-only selection, and full prediction artifacts. The bibliography includes the dataset paper, core statistical/deep-learning methods, and the Transformer models used in the experiments.

## Problem Definition and Objectives

The sentiment task maps Joy to Positive, Surprise to Neutral, and Sad/Angry/Fear/Disgust to Negative. The emotion task preserves Joy, Sad, Angry, Fear, Disgust, and Surprise. Objectives are to build separate reproducible pipelines, compare eight models, prevent ID/text duplicate overlap, quantify seed variation, retain all checkpoints and predictions, and deploy both tasks in one Streamlit demonstration.

## Proposed Methodology

### Preprocessing

The deterministic preprocessing pipeline normalizes Urdu/Arabic Unicode, removes URLs and mentions, preserves hashtag text, removes emojis, numbers, and punctuation, normalizes whitespace, and requires at least two cleaned tokens. Emoji removal controls direct target-signal leakage from the weak-label construction process.

### Group-Safe Splitting

Rows are connected when they share either a tweet ID or normalized text. Any connected group containing multiple task labels is excluded. Remaining exact normalized-text duplicates are reduced to one row. Complete groups are then assigned to deterministic stratified 70/15/15 train, validation, and test partitions. Automated checks confirm zero shared IDs and zero shared normalized texts across every split pair for both tasks.

### Models

The benchmark includes Logistic Regression, Linear SVM, Multinomial Naive Bayes, Text-CNN, BiLSTM with additive attention, mBERT, XLM-RoBERTa, and Urdu-RoBERTa. TF-IDF and neural vocabularies are fit on training data only. Neural models use inverse-frequency class weights, gradient clipping, validation macro-F1 checkpointing, and early stopping. Transformers use smoothed class weights and mixed precision.

### Selection and Uncertainty

Classical and neural models use three seeds. Transformer results are one-seed pilot experiments because of the available 24-36 hour compute window. The official ranking uses mean validation macro-F1; test metrics do not influence model choice. The selected canonical run is the seed with the highest validation macro-F1. Test uncertainty is reported with a 1,000-sample non-parametric bootstrap interval.

## Dataset and Experimental Setup

The raw CSV contains 1,048,000 rows, of which 514,571 lack the required Category label. After preprocessing, connected-group conflict removal, and deduplication, the sentiment benchmark retains **446,745** rows and the emotion benchmark retains **446,640** rows.

| Task | Train | Validation | Test | Conflict rows removed | Duplicate rows removed |
|---|---:|---:|---:|---:|---:|
| Sentiment | 312,710 | 67,016 | 67,019 | 14,069 | 57,152 |
| Emotion | 312,643 | 66,995 | 67,002 | 14,310 | 57,016 |

Training used an NVIDIA GeForce RTX 5070 Ti. The saved repository contains 36 official runs: 18 classical, 12 neural, and 6 Transformer runs. Every run has isolated models, metrics, predictions, and metadata.

## Results and Discussion

### Sentiment Results

| Rank | Family | Model | Seeds | Validation Macro-F1 | Test Macro-F1 |
|---:|---|---|---:|---:|---:|
| 1 | baseline | `linear_svm` | 3 | 0.4578 +/- 0.0000 | 0.4590 +/- 0.0000 |
| 2 | neural | `text_cnn` | 3 | 0.4329 +/- 0.0111 | 0.4320 +/- 0.0086 |
| 3 | baseline | `logistic_regression` | 3 | 0.4317 +/- 0.0000 | 0.4361 +/- 0.0000 |
| 4 | neural | `bilstm_attention` | 3 | 0.4231 +/- 0.0035 | 0.4214 +/- 0.0046 |
| 5 | transformer | `urdu_roberta` | 1 | 0.4172 +/- 0.0000 | 0.4137 +/- 0.0000 |
| 6 | transformer | `xlm_roberta` | 1 | 0.4091 +/- 0.0000 | 0.4122 +/- 0.0000 |
| 7 | transformer | `mbert` | 1 | 0.4074 +/- 0.0000 | 0.4076 +/- 0.0000 |
| 8 | baseline | `multinomial_nb` | 3 | 0.3867 +/- 0.0000 | 0.3820 +/- 0.0000 |

Linear SVM is selected by mean validation macro-F1 **0.4578** and obtains test macro-F1 **0.4590**. Its bootstrap 95% interval is **[0.4456, 0.4742]**. Neural seed variation is material, particularly for Text-CNN, which supports reporting repeated runs rather than one favorable seed.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.4083 | 0.3879 | 0.3978 | 8,683 |
| Neutral | 0.1111 | 0.0489 | 0.0679 | 184 |
| Positive | 0.9072 | 0.9156 | 0.9114 | 58,152 |

### Emotion Results

| Rank | Family | Model | Seeds | Validation Macro-F1 | Test Macro-F1 |
|---:|---|---|---:|---:|---:|
| 1 | baseline | `linear_svm` | 3 | 0.2856 +/- 0.0000 | 0.2854 +/- 0.0000 |
| 2 | baseline | `logistic_regression` | 3 | 0.2652 +/- 0.0000 | 0.2667 +/- 0.0000 |
| 3 | neural | `bilstm_attention` | 3 | 0.2179 +/- 0.0035 | 0.2190 +/- 0.0039 |
| 4 | transformer | `urdu_roberta` | 1 | 0.2167 +/- 0.0000 | 0.2170 +/- 0.0000 |
| 5 | transformer | `xlm_roberta` | 1 | 0.2126 +/- 0.0000 | 0.2136 +/- 0.0000 |
| 6 | transformer | `mbert` | 1 | 0.2106 +/- 0.0000 | 0.2102 +/- 0.0000 |
| 7 | baseline | `multinomial_nb` | 3 | 0.2002 +/- 0.0000 | 0.1997 +/- 0.0000 |
| 8 | neural | `text_cnn` | 3 | 0.1940 +/- 0.0020 | 0.1952 +/- 0.0015 |

Linear SVM is also selected for emotion with mean validation macro-F1 **0.2856** and test macro-F1 **0.2854**. Its bootstrap 95% interval is **[0.2749, 0.2962]**. Emotion remains substantially harder because the minority classes contain far fewer examples and emoji removal eliminates some of the most direct weak-label cues.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Angry | 0.1091 | 0.0809 | 0.0929 | 371 |
| Disgust | 0.1047 | 0.1287 | 0.1155 | 2,083 |
| Fear | 0.1022 | 0.0837 | 0.0920 | 227 |
| Joy | 0.9105 | 0.9054 | 0.9079 | 58,152 |
| Sad | 0.4187 | 0.4190 | 0.4189 | 5,985 |
| Surprise | 0.0888 | 0.0815 | 0.0850 | 184 |

### Interpretation

Sparse word n-grams remain strong on this corpus because repeated lexical patterns are informative even after duplicate groups are separated. The neural models show greater seed sensitivity and are trained with randomly initialized embeddings. Transformers are under-trained by design and should be interpreted as resource-constrained pilots rather than matched-budget evidence that pretraining is ineffective. High accuracy for majority-biased models confirms why macro-F1 must remain the headline metric.

## Error Analysis and Optimization

The selected models continue to struggle most on Neutral sentiment and the rare Fear/Surprise/Angry emotion classes. The pipeline preserves per-class reports, confusion matrices, and full prediction CSVs for inspection. Optimization includes class weighting, validation checkpointing, early stopping, gradient clipping, mixed precision, deterministic seeds, training-only feature fitting, and artifact isolation. Further improvements should prioritize human annotation, domain-adapted embeddings, longer matched-budget Transformer training, and calibrated decision probabilities.

## Ethical Considerations and Limitations

The source contains public social-media text and weak labels that may encode demographic, topical, and cultural bias. The models are unsuitable for surveillance, punitive moderation, diagnosis, or decisions about individuals. No human-gold evaluation is claimed. Confidence-like values from Linear SVM are explicitly labeled decision scores because they are normalized margins, not calibrated probabilities. The benchmark controls duplicate-instance leakage and direct emoji shortcuts, but weak-label noise and domain limitations remain.

## Conclusion

The project delivers separate, runnable sentiment and emotion pipelines with group-safe splits, eight model implementations, 36 isolated training runs, repeated-seed statistics, bootstrap uncertainty, complete saved artifacts, and dual-task inference. Linear SVM provides the strongest validation-ranked macro-F1 for both tasks under the official protocol. The main scientific conclusion is not that deep models are intrinsically weaker, but that a well-tuned sparse baseline remains difficult to beat under noisy weak supervision and a constrained training budget.

## References

1. M. T. Ali et al., "SentiUrdu-1M: A large-scale weakly-labelled Urdu Twitter dataset," Data in Brief, 2023.
2. A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
3. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.
4. A. Conneau et al., "Unsupervised Cross-Lingual Representation Learning at Scale," ACL, 2020.
5. Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv:1907.11692, 2019.
6. Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP, 2014.
7. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
8. D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," ICLR, 2015.
9. M. Schuster and K. K. Paliwal, "Bidirectional Recurrent Neural Networks," IEEE Transactions on Signal Processing, 1997.
10. T. Mikolov et al., "Advances in Pre-Training Distributed Word Representations," LREC, 2018.
11. T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space," ICLR Workshop, 2013.
12. J. Pennington, R. Socher, and C. D. Manning, "GloVe: Global Vectors for Word Representation," EMNLP, 2014.
13. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, 2011.
14. T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," EMNLP Demos, 2020.
15. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019.
16. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," ICLR, 2015.
17. P. Micikevicius et al., "Mixed Precision Training," ICLR, 2018.
18. C. Cortes and V. Vapnik, "Support-Vector Networks," Machine Learning, 1995.
19. J. Platt, "Probabilistic Outputs for Support Vector Machines," Advances in Large-Margin Classifiers, 1999.
20. G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," Information Processing and Management, 1988.
