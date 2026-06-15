# Experiment Setup

The official project is a dual-task benchmark with 36 official runs. Sentiment retains 446,745 rows and emotion retains 446,640 rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. Labels are weak references; no human-gold evaluation is claimed.

Training ran on an NVIDIA GeForce RTX 5070 Ti within a 24-36 hour execution window. Classical and neural runs use seeds 42, 52, and 62. mBERT, XLM-RoBERTa, and Urdu-RoBERTa use seed 42, 50,000 training examples, one epoch, and mixed precision. Transformer results are resource-limited pilots.

The complete run count is 18 classical + 12 neural + 6 Transformer = 36. Every run includes validation and test predictions plus a saved model artifact.
