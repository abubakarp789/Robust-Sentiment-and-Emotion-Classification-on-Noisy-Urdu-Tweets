# Source Code Folder

## Purpose

This folder is the reusable implementation package for the Urdu Sentiment and Emotion classification project. It contains data preparation, model architectures, training loops, evaluation routines, visual plotting, inference assistants, and validation code. User-facing pipeline commands are kept separately in `scripts/`.

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
- [evaluation_workflow.py](evaluation_workflow.py): Recomputes complete evaluation artifacts from saved prediction files.
- [error_analysis.py](error_analysis.py): Extracts classification error logs (confusion pairs, high-confidence wrong cases).
- [analyze_baseline_errors.py](analyze_baseline_errors.py): Investigates failure modes specifically for classical ML models.
- [compare_models.py](compare_models.py): Compiles metrics and training logs into a master CSV leaderboard.
- [plot_baseline_errors.py](plot_baseline_errors.py): Generates baseline error heatmap figures.
- [plot_neural_results.py](plot_neural_results.py): Generates training curves and F1 charts for neural networks.
- [plot_transformer_results.py](plot_transformer_results.py): Generates metrics charts for transformer encoders.
- [visualization_workflow.py](visualization_workflow.py): Builds submission figures from saved results.

### Inference and App Support
- [inference.py](inference.py): Wrapper class for loading trained models and scoring arbitrary user text.
- [explanation_assistant.py](explanation_assistant.py): Deterministic, rule-based summaries for predictions and saved error examples.

### Validation
- [validate_pipeline.py](validate_pipeline.py): Checks split counts, columns, and emoji preprocessing logic.
- [validate_baseline.py](validate_baseline.py): Validates baseline models and their saved outputs.
- [validate_neural.py](validate_neural.py): Checks neural networks weights, vocabulary maps, and histories.
- [validate_transformer.py](validate_transformer.py): Checks transformer best-model directory structures.
- [validate_error_analysis.py](validate_error_analysis.py): Checks error analysis outputs and sample explanations.
- [validate_notebooks.py](validate_notebooks.py): Verifies notebook structure and saved-artifact references.
- [validate_professor_requirements.py](validate_professor_requirements.py): Checks evidence for every professor stage, deliverable, and report section.
- [validate_final_project.py](validate_final_project.py): Unified project-wide test checking all outputs and metadata.

## How It Is Used

For the normal end-to-end workflow, use the numbered commands in `scripts/`. Individual source modules can still be invoked for development or validation. Common commands include:

```powershell
# Prepare data and train baselines through stable entry points
python scripts\01_prepare_data.py --config config.yaml
python scripts\02_train_classical.py --config config.yaml

# Run project-wide validation
python src\validate_final_project.py --config config.yaml
```

## Related Files

- [../README.md](../README.md)
- [../scripts/README.md](../scripts/README.md)
- [../config.yaml](../config.yaml)

## Notes

- All core logic is defined here. Python files should be kept modular and well-documented. Always run the validation suites after editing files.
