# Results and Analysis

## Assignment 4 Results

These are the dual-task results in `outputs/reports/final_nlp_project_report.pdf`, preserved as structured files under `outputs/report_snapshot/`.

### Sentiment

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.6598 | 0.3852 | 0.7107 |
| Linear SVM | 0.8783 | 0.4004 | 0.8409 |
| Text-CNN | 0.7848 | 0.4533 | 0.8117 |
| BiLSTM-Attention | 0.7762 | 0.4500 | 0.8040 |
| mBERT | 0.8054 | 0.4526 | 0.8217 |
| XLM-R | 0.7897 | 0.4475 | 0.8118 |
| Urdu-RoBERTa | 0.7750 | **0.4573** | 0.8011 |

### Emotion

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.3601 | 0.1632 | 0.4959 |
| Linear SVM | 0.8773 | 0.2087 | 0.8347 |
| Text-CNN | 0.6476 | 0.2499 | 0.7215 |
| BiLSTM-Attention | 0.6064 | 0.2428 | 0.6910 |
| mBERT | 0.6922 | **0.2703** | 0.7467 |
| XLM-R | 0.6907 | 0.2535 | 0.7462 |
| Urdu-RoBERTa | 0.6153 | 0.2539 | 0.6971 |

Best Assignment 4 sentiment macro-F1: Urdu-RoBERTa.<br>
Best Assignment 4 emotion macro-F1: mBERT.<br>
Highest accuracy on both tasks: Linear SVM.

## Packaged Sentiment Rerun

The packaged rerun uses 517,966 rows and a minimum two-token filter.

| Model | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|
| Linear SVM | 0.8531 | **0.5040** | 0.8527 |
| Logistic Regression | 0.7740 | 0.4613 | 0.8013 |
| BiLSTM-Attention | 0.7408 | 0.4506 | 0.7763 |
| Text-CNN | 0.7582 | 0.4476 | 0.7882 |
| XLM-R | 0.8528 | 0.4346 | 0.8426 |
| mBERT | 0.8520 | 0.4240 | 0.8382 |
| Multinomial NB | **0.8787** | 0.4014 | 0.8417 |

The packaged best macro-F1 model is Linear SVM. Multinomial NB has higher accuracy but zero Neutral F1 and substantially lower macro-F1.

## Macro-F1 vs Weighted-F1

Weighted-F1 weights each class by its frequency, so strong Positive/Joy performance dominates. Macro-F1 gives every class equal importance. The large gap between the two metrics shows that high overall accuracy does not imply usable rare-class behavior.

For packaged Linear SVM, test weighted-F1 is 0.8527 but macro-F1 is 0.5040. Neutral F1 is only 0.1303. For packaged Multinomial NB, accuracy is 0.8787 while macro-F1 falls to 0.4014 and Neutral F1 is zero.

## Confusion Matrix Interpretation

- Majority Positive/Joy examples are recognized well.
- Neutral/Surprise is frequently predicted as Positive or Negative.
- Rare emotion classes are often absorbed into Joy.
- Class weighting increases minority predictions but can increase false positives.
- Transformer contextualization does not overcome weak labels and extreme imbalance by itself.

## Qualitative Errors

Assignment 3 and packaged error artifacts identify:

- Negation scope errors.
- Sarcasm with positive surface vocabulary.
- Urdu/English/Roman-Urdu code-mixing.
- Very short or context-dependent tweets.
- Religious or poetic expressions with ambiguous weak labels.
- High-confidence disagreements likely caused by label noise.

## Threats to Validity

- No human gold test set.
- Single fixed seed with no uncertainty intervals.
- Different preprocessing thresholds between the final report and packaged rerun.
- Packaged Transformers were resource-constrained subset runs.
- Packaged neural rerun lacks fastText initialization.
- Missing deep-model weight files prevent exact checkpoint inference verification.

The correct conclusion is not that one architecture universally wins. The strongest result is the controlled demonstration that leakage prevention, class imbalance, and label quality dominate this task.
