# Final Evaluation Summary

This document presents the final performance rankings and comparative findings for the Leakage-Aware Urdu Tweet Sentiment Classification project.

## Final Model Rankings

Models are ranked by their performance on the test split (77,695 examples), with **Test Macro-F1** serving as the primary metric for model comparison due to class imbalance.

| Rank | Model Family | Model Name | Test Macro-F1 | Test Accuracy | Neutral F1 | Negative F1 | Positive F1 | Beats Linear SVM |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | baseline | **linear_svm** | **0.5040** | 0.8531 | **0.1303** | 0.4662 | 0.9156 | - |
| 2 | baseline | **logistic_regression** | 0.4613 | 0.7740 | 0.0828 | 0.4409 | 0.8601 | No |
| 3 | neural | **bilstm_attention** | 0.4506 | 0.7408 | 0.0991 | 0.4181 | 0.8346 | No |
| 4 | neural | **text_cnn** | 0.4476 | 0.7582 | 0.0777 | 0.4164 | 0.8488 | No |
| 5 | transformer | **xlm_roberta** | 0.4346 | 0.8528 | 0.0000 | 0.3873 | 0.9166 | No |
| 6 | transformer | **mbert** | 0.4240 | 0.8520 | 0.0000 | 0.3553 | 0.9165 | No |
| 7 | baseline | **multinomial_nb** | 0.4014 | **0.8787** | 0.0000 | 0.2702 | **0.9339** | No |

---

## Best Models by Metric

- **Best Model by Macro-F1**: **Linear SVM** (`0.5040`)
- **Best Model by Accuracy**: **Multinomial Naive Bayes** (`0.8787`)
- **Best Model by Neutral F1**: **Linear SVM** (`0.1303`)

---

## Analysis and Comparative Discussion

### Accuracy vs. Macro-F1
Accuracy is a highly misleading indicator of performance on this dataset due to the severe class imbalance. The majority `Positive` class represents **86.2%** of the test split. Consequently, a naive baseline classifier that predicts `Positive` for every single tweet would achieve an accuracy of `0.8622`, but its Macro-F1 would be only `0.3087` (with `0.0` F1 for both `Neutral` and `Negative`). 

Multinomial Naive Bayes exemplifies this trap: it achieves the highest test accuracy (`0.8787`) because it is heavily biased toward predicting the majority `Positive` class (achieving a `Positive` F1 of `0.9339`), but it fails completely on `Neutral` (`0.0` F1) and performs poorly on `Negative` (`0.2702` F1), resulting in a low Macro-F1 of `0.4014`. Macro-F1 provides a balanced assessment by weighting each class equally.

### Class Imbalance Impact
The SentiUrdu-1M dataset splits exhibit a severe imbalance:
- **Positive**: 66,985 test rows (86.21%)
- **Negative**: 10,496 test rows (13.51%)
- **Neutral**: 214 test rows (0.28%)

The extremely low representation of the `Neutral` class (<0.3%) makes it difficult for any model to learn distinct neutral indicators. Neutral F1 is low across all families:
- The **Linear SVM** achieved the highest score (`0.1303`) by drawing high-dimensional sparse decision margins.
- The **BiLSTM-Attention** model reached `0.0991` Neutral F1 but misclassified a large proportion of positive examples due to the extreme class weights.
- Both **transformers** failed to predict any Neutral examples correctly (`0.0000` F1).

### Baseline vs. Neural vs. Transformer
1. **Classical Baselines Outperform Neural Networks**: The `Linear SVM` baseline outperformed the `BiLSTM-Attention` by a margin of `0.0534` Macro-F1. This is because sparse n-gram TF-IDF representations directly exploit a massive vocabulary (100,000 features). Neural models with randomly initialized embeddings had to learn Urdu lexical semantics from scratch using noisy weakly supervised labels, which led to overfitting on majority patterns.
2. **Transformers Underperform in this Run**: XLM-RoBERTa (`0.4346`) and mBERT (`0.4240`) failed to beat the baseline and neural models. This is due to resource constraints on their training: they were fine-tuned on a 50,000-sample training subset for only 1 epoch. This duration was insufficient for large pre-trained multilingual models to adapt their vocabularies to noisy, informal, code-mixed Roman/Arabic Urdu tweet semantics.

---

## Model Selection for Deployment
The **TF-IDF + Linear SVM** has been selected as the final deployed model in the Streamlit application because:
1. It achieves the highest Test Macro-F1 (`0.5040`).
2. It exhibits the best performance on the minority `Neutral` class (`0.1303` F1).
3. It has extremely low inference latency and memory requirements, making it ideal for CPU-based lightweight deployment without requiring deep learning or GPU instances.

---

## Limitations and Future Work

### Limitations of Current Work
- **Noisy Weak Labels**: SentiUrdu-1M labels are derived from emoji heuristics, introducing substantial label noise.
- **Limited Transformer Training**: Fine-tuning was restricted to a 50,000-sample subset and 1 epoch due to local GPU memory and training time constraints.
- **Rule-Based Explanations**: The explanation assistant is rule-based and lacks the capacity to capture subtle contextual, sarcastic, or cultural nuances of Urdu social media discourse.

### Recommended Future Improvements
1. **Longer Transformer Training**: Fine-tune XLM-RoBERTa and mBERT on the full 362,576-row training split for 5–10 epochs using early stopping and learning rate scheduling.
2. **Domain-Specific Encoders**: Use domain-adapted models such as `XLM-T` (Twitter-trained) or `urduhack/roberta-urdu-small` (Urdu-specific).
3. **Pre-trained Embeddings**: Incorporate pre-trained Urdu fastText embeddings in the BiLSTM-Attention model rather than random embeddings.
4. **Manual Annotation**: Annotate a high-quality, clean sample of test tweets (e.g., 2,000 rows) to evaluate the models on verified labels rather than weakly supervised emoji heuristics.
