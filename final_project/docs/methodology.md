# Methodology

The official project is a dual-task benchmark with 36 official runs. Sentiment retains 446,745 rows and emotion retains 446,640 rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. Labels are weak references; no human-gold evaluation is claimed.

## Protocol

1. Normalize Urdu/Arabic Unicode.
2. Remove URLs and mentions; preserve hashtag text.
3. Remove emojis before feature extraction.
4. Remove numbers/punctuation and require at least two cleaned tokens.
5. Build connected duplicate groups using shared ID or normalized text.
6. Exclude connected groups containing conflicting task labels.
7. Deduplicate normalized text and create deterministic 70/15/15 group-level splits.
8. Fit all text representations on training data only.
9. Run eight models with three classical/neural seeds and one resource-limited Transformer seed.
10. Rank by mean validation macro-F1 and bootstrap the selected test predictions.

Task configurations are `config_sentiment.yaml` and `config_emotion.yaml`. Run isolation is under `outputs/<task>/runs/<family>/<model>/seed_<seed>/`.
