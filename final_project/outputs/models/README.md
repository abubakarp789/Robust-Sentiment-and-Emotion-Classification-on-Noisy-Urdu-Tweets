# Models Folder

## Purpose

This folder contains the saved weights, vocabulary mapping files, and label serialization artifacts for all trained models. These files are used for inference and deployment in the Streamlit application.

## Contents

| File/Folder | Description |
|---|---|
| [baseline_linear_svm.joblib](baseline_linear_svm.joblib) | Deployed Linear SVM model checkpoint |
| [baseline_logistic_regression.joblib](baseline_logistic_regression.joblib) | Baseline Logistic Regression model checkpoint |
| [baseline_multinomial_nb.joblib](baseline_multinomial_nb.joblib) | Baseline Multinomial Naive Bayes model checkpoint |
| [neural_bilstm_attention.pt](neural_bilstm_attention.pt) | Saved weights for BiLSTM-Attention model |
| [neural_text_cnn.pt](neural_text_cnn.pt) | Saved weights for Text-CNN model |
| [neural_vocab.json](neural_vocab.json) | Vocabulary index mapping for neural networks |
| [neural_label_mapping.json](neural_label_mapping.json) | Label mapping index for neural networks |
| [transformer_label_mapping.json](transformer_label_mapping.json) | Label mapping index for transformer models |
| [transformer_mbert/](transformer_mbert/) | mBERT transformer model directory |
| [transformer_xlm_roberta/](transformer_xlm_roberta/) | XLM-RoBERTa transformer model directory |

## How It Is Used

The files in this directory are generated during training and loaded during evaluation or deployment:
- **Baseline Models**: Created by `src/train_baseline.py`. The best baseline model, [baseline_linear_svm.joblib](baseline_linear_svm.joblib), serves as the **final deployed model** in this project due to its superior Macro-F1 performance.
- **Neural Models**: Created by `src/train_neural.py`. They require [neural_vocab.json](neural_vocab.json) and [neural_label_mapping.json](neural_label_mapping.json) for mapping tokens and target classes.
- **Transformer Encoders**: Fine-tuned by `src/train_transformer.py` and saved under their respective directories.
- **Deployment**: The Streamlit application loads these checkpoints to compute predictions on user-input text.

## Related Files

- [../README.md](../README.md)
- [../../app/streamlit_app.py](../../app/streamlit_app.py)
- [../../src/inference.py](../../src/inference.py)

## Notes

- **Warning**: Model files are binary check-points and should not be modified manually. Retraining or evaluating models requires invoking the corresponding python scripts in `src/`.
- If deep learning or transformer weights are missing or deleted, the inference module will automatically fallback to the [baseline_linear_svm.joblib](baseline_linear_svm.joblib) classifier.
