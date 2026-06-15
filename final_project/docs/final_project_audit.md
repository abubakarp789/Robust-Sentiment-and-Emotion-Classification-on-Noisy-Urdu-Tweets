# Final Project Audit

Audit date: June 15, 2026.

The official project is a dual-task benchmark with 36 official runs. Sentiment retains 446,745 rows and emotion retains 446,640 rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. Labels are weak references; no human-gold evaluation is claimed.

## Closed Gaps

- Separate sentiment and emotion configs, splits, runs, results, inference, and demo paths.
- Connected ID/text duplicate grouping with zero measured cross-split overlap.
- Three seeds for classical/neural models and one declared resource-limited Transformer seed.
- Validation-only selection, canonical runs, bootstrap intervals, complete predictions/checkpoints, and generated reporting.

## Remaining Scientific Limitation

No native-speaker human-gold evaluation was possible. This is disclosed throughout the package and prevents a claim of gold-standard real-world performance.
