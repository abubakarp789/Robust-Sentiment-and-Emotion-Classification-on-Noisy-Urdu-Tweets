# Final Project Submission Checklist

This checklist documents the completion state of the **Leakage-Aware Urdu Tweet Sentiment and Emotion Classification** semester project across ten core dimensions.

---

## 1. Code & Repository Structure
- [x] **Separation of Concerns**: Preprocessing, labeling, model training, evaluation, error analysis, and inference are organized into dedicated scripts under `src/`.
- [x] **No Inter-Assignment Mutation**: Asserts that `Assignment#01`, `Assignment#02`, and `Assignment#03` directories are left completely unmodified.
- [x] **Project Configuration**: All file paths, hyperparameters, and preprocessing options are centralized in `config.yaml`.
- [x] **Compilation and Linting**: All codebase scripts compile cleanly and pass py_compile validations.

## 2. Data Splits
- [x] **Self-Contained Raw Data**: Copied the 1,048,000-row source CSV to `data/raw/Urdu Tweets Dataset.csv` and updated `config.yaml`.
- [x] **Processed Dataset**: Saved the 517,966-row cleaned labelled pre-split dataset to `data/processed/processed_sentiment_dataset.csv`.
- [x] **Emoji Removal**: Stripped all emojis from tweet inputs to control label leakage from emoji heuristics.
- [x] **Normalizations**: Performed Unicode NFC normalization, mention/URL stripping, and canonical label mapping.
- [x] **Stratified Splitting**: Split 517,966 valid rows into reproducible 70/15/15 train (362,576), validation (77,695), and test (77,695) sets.
- [x] **Split Summary**: Saved dataset split distribution and normalizations to `outputs/results/split_summary.json`.
- [x] **Annotation Support**: Created a balanced optional annotation sample with 100 Positive, 100 Negative, and 100 Neutral examples.
- [x] **Evaluation Isolation**: The annotation sample is not used in training or evaluation and does not affect reported results.

## 3. Models
- [x] **Baseline Classifiers**: Trained and saved TF-IDF + Linear SVM, Logistic Regression, and Multinomial Naive Bayes models.
- [x] **Neural Classifiers**: Implemented and trained PyTorch Text-CNN and BiLSTM-Attention models using early stopping on Validation Macro-F1.
- [x] **Transformer Classifiers**: Fine-tuned mBERT and XLM-RoBERTa on a 50k stratified subset for 1 epoch using smoothed class weights.
- [x] **Checkpoint Verification**: Verified all `.joblib`, `.pt`, and Hugging Face tokenizer/weight files are stored in `outputs/models/`.

## 4. Evaluation
- [x] **Leaderboard Integration**: Compiled all 7 models into a master comparison leaderboard at `outputs/results/model_comparison_leaderboard.csv`.
- [x] **Headline Metric**: Selected **Macro-F1** as the headline metric to properly evaluate performance under SentiUrdu-1M's severe class imbalance.
- [x] **Per-Class Metrics**: Reported per-class Precision, Recall, and F1 to show class-specific patterns.
- [x] **Evaluation Summary**: Created `reports/final_evaluation_summary.md` and `outputs/results/final_evaluation_summary.json` storing structured ranking metrics.

## 5. Error Analysis
- [x] **Error Breakdown**: Quantitative categorization showing that polarity swaps (Positive/Negative confusion) account for **96.26%** of all SVM errors.
- [x] **Neutral Class Failure**: Documented that all models fail to classify the scarce Neutral class effectively, with F1 scores between 0% and 13.03%.
- [x] **High-Confidence Error Table**: Filtered and saved cases where predictions were wrong despite confidence scores >= 0.80.
- [x] **Explanation Samples**: Generated and saved `outputs/error_analysis/explanation_samples.json` containing human-readable explanations of correct and incorrect predictions.

## 6. Deployment
- [x] **Streamlit Web Application**: Completed the interactive web app at `app/streamlit_app.py`.
- [x] **Best Model Badge**: Added a prominent banner highlighting the selected **TF-IDF + Linear SVM** model.
- [x] **Robust Fallbacks**: Configured the application to fall back gracefully to the Linear SVM classifier if transformer checkpoints are missing.
- [x] **Inference Pipeline**: Preprocesses inputs, returns predictions, shows confidence score bar charts, and integrates the explanation assistant.

## 7. Documentation
- [x] **Submission README**: Updated `README.md` at the project root to outline the methodology, pipeline, best model results, and quick-start instructions.
- [x] **Dataset Card**: Completed `reports/dataset_card.md` detailing dataset origin, curation, attributes, and limitation notes.
- [x] **Final Report**: Prepared a comprehensive, publication-ready 14-section report at `reports/final_report.md`.

## 8. Presentation
- [x] **Presentation Slides**: Prepared a detailed 12-slide presentation outline with visual guidance and comprehensive speaker notes at `reports/slides_outline.md`.
- [x] **Demo Script**: Created a 2-3 minute spoken demo walkthrough at `reports/demo_script.md` for live evaluations or video recordings.

## 9. Ethical Considerations & Limitations
- [x] **Ethics Review**: Documented risks associated with weak labels, surveillance misuse, class bias, and text exposures in `reports/ethics_and_limitations.md`.
- [x] **Honest Constraints**: Addressed transformer subset training (50k sample, 1 epoch) and random embedding limitations without fabricating or smoothing results.

## 10. Reproducibility & Verification
- [x] **Reproducibility Rules**: Used fixed random seeds, saved immutable splits, documented training hyper-parameters, and saved all model prediction CSV files.
- [x] **Data Loading**: The raw dataset is now copied into `data/raw/` so the `final_project` folder is self-contained for data loading.
- [x] **Data Commands**: `python src\create_splits.py --config config.yaml`, `python src\create_annotation_sample.py --config config.yaml`, `python src\validate_pipeline.py --config config.yaml`, and `python src\validate_final_project.py --config config.yaml`.
- [x] **Project Unit Tests**: Verified that all 22 unit tests (`python -m pytest tests`) pass successfully.
- [x] **Project-Wide Validator**: Implemented `src/validate_final_project.py` and confirmed that all project directory, file, and report structure validations pass successfully.

## 11. Analysis Notebooks
- [x] **Artifact-Backed Review**: The notebooks are analysis notebooks that load already generated artifacts instead of retraining models. This ensures reproducibility and avoids expensive re-training during review.
- [x] **Notebook Validation**: Run `python src\validate_notebooks.py --config config.yaml` before submission.
- [x] **Notebook Usage**: Open the analysis collection with `jupyter notebook notebooks`.
