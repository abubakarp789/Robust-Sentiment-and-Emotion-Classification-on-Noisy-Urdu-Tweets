# Final Project Roadmap

## Current Status

The repository currently contains three assignment folders. Assignment 1 defines the project and preprocessing pipeline, Assignment 2 provides the literature review, and Assignment 3 implements a substantial model comparison with EDA, classical baselines, neural models, transformer models, evaluation outputs, and a technical report.

This `final_project/` folder is a clean final-submission structure. It does not overwrite or delete any assignment work.

The final-project preprocessing, label-normalization, and split-generation pipeline has now been migrated into:

- `src/preprocessing.py`
- `src/label_mapping.py`
- `src/create_splits.py`
- `src/validate_pipeline.py`

The pipeline is controlled by `config.yaml`.

## Completed Work from Assignment 1

- Defined the project title and problem statement.
- Selected SentiUrdu-1M as the main dataset.
- Motivated Urdu tweet sentiment and emotion analysis as a complex NLP problem.
- Identified weak supervision, noisy tweets, code-mixing, and emoji leakage as key challenges.
- Built an 8-step preprocessing pipeline:
  - Unicode normalization
  - URL removal
  - Mention removal
  - Hashtag cleanup
  - Emoji removal
  - Number removal
  - Punctuation removal
  - Whitespace normalization

## Completed Work from Assignment 2

- Completed a research-style literature review.
- Reviewed 20 related papers.
- Organized prior work into classical ML, deep learning, transformer-based, and multimodal/advanced approaches.
- Identified the gap around weak supervision, fair model comparison, label leakage, and clean evaluation.
- Built a comparative literature table with methods, datasets, findings, and limitations.

## Completed Work from Assignment 3

- Created reusable preprocessing and configuration modules.
- Performed detailed dataset analysis and visualizations.
- Implemented label normalization for noisy `Category` values.
- Trained/evaluated classical baselines:
  - TF-IDF + Logistic Regression
  - TF-IDF + Linear SVM
- Trained/evaluated neural models:
  - Text-CNN
  - BiLSTM-Attention
- Trained/evaluated transformer models:
  - mBERT
  - XLM-RoBERTa
  - Urdu-RoBERTa
- Generated leaderboard CSV files.
- Generated figures and confusion matrices.
- Wrote a technical report.

## Completed Final Project Migration

- [x] Migrate model-training code into `final_project/src`.
- [x] Save model checkpoints under `outputs/models`.
- [x] Save richer prediction files under `outputs/predictions` with raw text, cleaned text, labels, predictions, and confidence.
- [x] Build complete error-analysis outputs under `outputs/error_analysis`.
- [x] Add a deployment-ready Streamlit app.
- [x] Add final report content under `reports/final_report.md`.
- [x] Add final slides or presentation outline.
- [x] Add ethics and limitations into the final report.
- [x] Fix any mismatch between documentation and actual saved artifacts.

## Week-Wise Final Implementation Plan

### Week 1: Repository Consolidation and Data Pipeline

Tasks:

- Finalize `final_project/` structure. Completed.
- Move stable preprocessing and label-mapping logic into `src/`. Completed.
- Create dataset card. Completed.
- Create processed data and split-generation scripts. Completed.
- Save stratified train/validation/test splits. Run `python src/create_splits.py`.

Expected output:

- Clean final repository structure.
- Documented dataset card.
- Reproducible data splits.

Commands:

```bash
cd final_project
python src/create_splits.py
python src/validate_pipeline.py
```

### Week 2: Baseline and Neural Model Packaging

Tasks:

- Convert baseline notebook logic into `src/train_baseline.py`. Completed.
- Convert neural model training logic into `src/train_neural.py`. Completed.
- Save model artifacts and predictions. Completed.
- Add reusable evaluation script. Completed.

Expected output:

- Reproducible baseline and neural model runs. Completed.
- Saved metric tables and prediction files. Completed.

### Week 3: Transformer, Evaluation, and Error Analysis

Tasks:

- Convert transformer training logic into `src/train_transformer.py`. Completed.
- Save transformer predictions and model metadata. Completed.
- Build `src/error_analysis.py`. Completed.
- Generate misclassified example tables. Completed.
- Add confusion matrix and per-class report exports. Completed.

Expected output:

- Final model comparison table. Completed.
- Error-analysis report. Completed.
- Model selection justification. Completed.

### Week 4: Deployment, Report, and Presentation

Tasks:

- Implement Streamlit demo. Completed.
- Add inference pipeline. Completed.
- Add model explanation placeholder or GenAI assistant. Completed.
- Write final report. Completed.
- Prepare slides outline and demo script. Completed.

Expected output:

- Working demo app. Completed.
- Final report. Completed.
- Presentation-ready project. Completed.

## Must-Have Tasks

- [x] Preserve original assignment folders.
- [x] Finalize preprocessing and label mapping modules.
- [x] Save train/validation/test splits.
- [x] Report macro-F1 as the main metric.
- [x] Save model comparison table.
- [x] Save confusion matrices.
- [x] Add ethics and limitations.
- [x] Build basic Streamlit demo.
- [x] Write final README and final report.

## Should-Have Tasks

- [x] Save model checkpoints.
- [x] Save prediction files with text and confidence.
- [ ] Create clean manually verified test subset (Future Work).
- [ ] Add emoji-removal ablation (Future Work).
- [x] Add per-class error analysis.
- [x] Add reproducibility checklist.

## Nice-to-Have Tasks

- [x] Add GenAI explanation assistant.
- [ ] Add LoRA or adapter-based transformer fine-tuning (Future Work).
- [ ] Add XLM-T or another Twitter-domain transformer (Future Work).
- [ ] Add model calibration analysis (Future Work).
- [ ] Deploy on Hugging Face Spaces or Streamlit Cloud (Future Work).
- [ ] Record a short demo video (Future Work).

## Analysis Notebook Review

The notebooks are analysis notebooks that load already generated artifacts instead of retraining models. This ensures reproducibility and avoids expensive re-training during review.

```powershell
jupyter notebook notebooks
python src\validate_notebooks.py --config config.yaml
```
