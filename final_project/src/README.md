# Source Code Folder

## Purpose

This folder contains the source code for the Urdu Sentiment and Emotion classification project. It is structured into data preparation, model architectures, training loops, evaluation routines, visual plotting, inference assistants, and validation scripts.

## Contents

The scripts in this directory are grouped by their operational purpose:

### Data Pipeline
- [create_splits.py](create_splits.py): Entry-point script to load raw data, apply preprocessing, normalize labels, and output stratified CSV splits.
- [create_annotation_sample.py](create_annotation_sample.py): Generates an optional balanced annotation sample of 300 tweets.
- [preprocessing.py](preprocessing.py): Leakage-aware Urdu tweet text cleaning module (Unicode normalization, mention/URL/emoji stripping).
- [label_mapping.py](label_mapping.py): Canonical labeling module mapping inconsistent category formats to emotions and sentiment tasks.

### Modeling
- [train_baseline.py](train_baseline.py): Script training classical classifiers (Linear SVM, Logistic Regression, Multinomial Naive Bayes) using TF-IDF.
- [train_neural.py](train_neural.py): Script training deep neural networks (Text-CNN, BiLSTM-Attention) using PyTorch.
- [train_transformer.py](train_transformer.py): Script fine-tuning pre-trained models (mBERT, XLM-RoBERTa).
- [models_dl.py](models_dl.py): PyTorch network architecture definitions (CNN and Attention-LSTM).
- [neural_utils.py](neural_utils.py): Utility functions for PyTorch data handling, vocabulary indexing, and tokenization.

### Evaluation and Analysis
- [evaluate.py](evaluate.py): Computes precision, recall, F1, accuracy, and confusion matrices.
- [error_analysis.py](error_analysis.py): Extracts classification error logs (confusion pairs, high-confidence wrong cases).
- [analyze_baseline_errors.py](analyze_baseline_errors.py): Investigates failure modes specifically for classical ML models.
- [compare_models.py](compare_models.py): Compiles metrics and training logs into a master CSV leaderboard.
- [plot_baseline_errors.py](plot_baseline_errors.py): Generates baseline error heatmap figures.
- [plot_neural_results.py](plot_neural_results.py): Generates training curves and F1 charts for neural networks.
- [plot_transformer_results.py](plot_transformer_results.py): Generates metrics charts for transformer encoders.

### Inference and App Support
- [inference.py](inference.py): Wrapper class for loading trained models and scoring arbitrary user text.
- [explanation_assistant.py](explanation_assistant.py): Generative AI assistant providing rule-based explanation texts for user-submitted tweets.

### Validation
- [validate_pipeline.py](validate_pipeline.py): Checks split counts, columns, and emoji preprocessing logic.
- [validate_baseline.py](validate_baseline.py): Validates baseline models and their saved outputs.
- [validate_neural.py](validate_neural.py): Checks neural networks weights, vocabulary maps, and histories.
- [validate_transformer.py](validate_transformer.py): Checks transformer best-model directory structures.
- [validate_error_analysis.py](validate_error_analysis.py): Checks error analysis outputs and sample explanations.
- [validate_notebooks.py](validate_notebooks.py): Verifies notebook compilation and execution consistency.
- [validate_final_project.py](validate_final_project.py): Unified project-wide test checking all outputs and metadata.

## How It Is Used

You can invoke these scripts from the project root. Common commands include:

```powershell
# Prepare data
python src\create_splits.py --config config.yaml

# Train baselines
python src\train_baseline.py --config config.yaml

# Run project-wide validation
python src\validate_final_project.py --config config.yaml
```

## Related Files

- [../README.md](../README.md)
- [../config.yaml](../config.yaml)

## Notes

- All core logic is defined here. Python files should be kept modular and well-documented. Always run the validation suites after editing files.
