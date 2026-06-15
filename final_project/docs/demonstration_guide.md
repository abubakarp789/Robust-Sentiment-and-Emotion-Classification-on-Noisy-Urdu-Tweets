# Demonstration Guide

The official project is a dual-task benchmark with 36 official runs. Sentiment retains 446,745 rows and emotion retains 446,640 rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. Labels are weak references; no human-gold evaluation is claimed.

1. Run `streamlit run app/streamlit_app.py`.
2. Select Sentiment or Emotion.
3. Select one of the eight saved models; Linear SVM is the dependable selected model for both tasks.
4. Enter an Urdu tweet and show the cleaned text after emoji/URL/mention removal.
5. Run prediction and explain that SVM shows a decision score rather than calibrated confidence.
6. Show the validation-ranked leaderboard and the weak-label/group-safe disclosures.
7. Switch tasks to demonstrate separate pipelines and artifacts.

Do not claim human-gold accuracy or that the one-epoch Transformer pilots represent their maximum capability.
