# Ethics and Limitations

The official project is a dual-task benchmark with 36 official runs. Sentiment retains 446,745 rows and emotion retains 446,640 rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. Labels are weak references; no human-gold evaluation is claimed.

- Weak labels can encode emoji heuristics and annotation noise; disagreement is not always a model error.
- Exact tweets are retained only for local academic analysis and should not be used for user profiling.
- Severe imbalance limits reliability for Neutral, Angry, Fear, Disgust, Surprise, and Sad.
- Transformer experiments are not compute-matched to classical/neural runs.
- Linear SVM output is a decision score, not a calibrated probability.
- No deployment should make consequential decisions without native-speaker review, consent, and domain-specific validation.
