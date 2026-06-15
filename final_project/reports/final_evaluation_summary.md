# Final Evaluation Summary

## Protocol

- Tasks: Sentiment (3 classes) and Emotion (6 classes)
- Models: 8 per task
- Completed runs: 36
- Classical/neural seeds: 42, 52, 62
- Transformer seed: 42
- Selection metric: mean validation macro-F1
- Test uncertainty: 1,000-sample bootstrap interval
- Human-gold evaluation: unavailable and not claimed

## Sentiment

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

Selected model: `linear_svm`<br>
Validation macro-F1: 0.4578<br>
Test macro-F1: 0.4590<br>
Bootstrap 95% interval: [0.4456, 0.4742]

## Emotion

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

Selected model: `linear_svm`<br>
Validation macro-F1: 0.2856<br>
Test macro-F1: 0.2854<br>
Bootstrap 95% interval: [0.2749, 0.2962]
