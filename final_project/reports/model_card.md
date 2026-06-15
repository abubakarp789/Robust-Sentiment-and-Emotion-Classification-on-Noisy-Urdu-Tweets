# Model Card: Urdu Sentiment and Emotion Benchmark

## Models

`linear_svm`, `logistic_regression`, `multinomial_nb`, `text_cnn`, `bilstm_attention`, `mbert`, `xlm_roberta`, and `urdu_roberta` are trained separately for Sentiment and Emotion.

## Selected Models

| Task | Model | Validation Macro-F1 | Test Macro-F1 | Test 95% interval |
|---|---|---:|---:|---:|
| Sentiment | `linear_svm` | 0.4578 | 0.4590 | [0.4456, 0.4742] |
| Emotion | `linear_svm` | 0.2856 | 0.2854 | [0.2749, 0.2962] |

## Intended Use

Course demonstration, reproducible Urdu NLP research, and aggregate error analysis. Outputs are estimates and must not be used for decisions about individuals.

## Score Semantics

Neural and Transformer outputs are softmax probabilities. Linear SVM outputs are normalized decision margins and are displayed as decision scores, not calibrated probabilities.

## Limitations

Weak labels, extreme imbalance, social-media domain restriction, absence of human-gold evaluation, randomly initialized neural embeddings, and one-seed resource-constrained Transformer experiments.
