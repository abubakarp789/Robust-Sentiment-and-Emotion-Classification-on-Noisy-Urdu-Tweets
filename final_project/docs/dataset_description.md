# Dataset Description

The official project is a dual-task benchmark with 36 official runs. Sentiment retains 446,745 rows and emotion retains 446,640 rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. Labels are weak references; no human-gold evaluation is claimed.

The raw file `data/raw/Urdu Tweets Dataset.csv` has 1,048,000 rows and columns `Id`, `Text`, `Emotions`, and `Category`. Of these, 514,571 rows lack the required category label. The pipeline removes URLs, mentions, emojis, numbers, punctuation, empty/short text, conflicting connected groups, and exact normalized-text duplicates.

| Task | Train | Validation | Test | Classes |
|---|---:|---:|---:|---:|
| Sentiment | 312,710 | 67,016 | 67,019 | 3 |
| Emotion | 312,643 | 66,995 | 67,002 | 6 |

See `reports/dataset_card.md` and each task's `outputs/<task>/results/split_summary.json` for complete distributions and removal counts.
