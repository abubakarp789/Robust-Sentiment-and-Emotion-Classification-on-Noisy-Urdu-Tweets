# Demo Script

"This project evaluates sentiment and emotion separately on noisy Urdu tweets. The official benchmark contains 36 official runs across eight models. We remove emojis because the source labels are weakly related to emoji signals, and we prevent shared IDs or normalized text from crossing data splits. Models are selected using validation macro-F1 only. Linear SVM is selected for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. The labels are weak references and no human-gold evaluation is claimed."

During the live demo, switch between tasks, show preprocessing, predict with Linear SVM, then compare another model using the validation-ranked table.
