# Model Artifacts

## Included and Runnable

- `baseline_linear_svm.joblib`
- `baseline_logistic_regression.joblib`
- `baseline_multinomial_nb.joblib`

These scikit-learn pipelines include TF-IDF preprocessing and can be loaded for local inference. Linear SVM is the default live-demo model.

## Included Support Files

- `neural_vocab.json`
- `neural_label_mapping.json`
- `transformer_label_mapping.json`
- mBERT tokenizer/config files under `transformer_mbert/best/`
- XLM-R tokenizer/config files under `transformer_xlm_roberta/best/`

## Not Included

- `neural_text_cnn.pt`
- `neural_bilstm_attention.pt`
- Transformer `.safetensors` or `.bin` weight files

The missing files are large and are excluded by `final_project/.gitignore`. Tokenizer/config directories are not complete inference checkpoints without model weights. Saved predictions, metrics, reports, and training histories remain available in the neighboring output folders.

The Streamlit app checks for real weight files and falls back to the included Linear SVM when a selected deep model is unavailable.
