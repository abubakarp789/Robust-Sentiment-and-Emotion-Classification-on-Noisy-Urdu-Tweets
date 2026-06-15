# Results Analysis

The official project is a dual-task benchmark with 36 official runs. Sentiment retains 446,745 rows and emotion retains 446,640 rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. Labels are weak references; no human-gold evaluation is claimed.

| Task | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|---:|
| Sentiment Linear SVM | 0.4578 | 0.8448 | 0.4590 | 0.8425 |
| Emotion Linear SVM | 0.2856 | 0.8282 | 0.2854 | 0.8301 |

The gap between accuracy and macro-F1 reflects majority-class dominance. Emotion is harder because six classes include several very rare labels. See `reports/final_evaluation_summary.md` and the task leaderboards for all eight models and seed variation.
