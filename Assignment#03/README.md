# Assignment 3 — Robust Sentiment & Emotion Classification on Noisy Urdu Tweets

**Course:** CSC-355 Natural Language Processing
**University:** Namal University Mianwali
**Group:** Raqib Hayat (NUM-BSCS-2022-40) and Abu Bakar (NUM-BSCS-2022-41)
**Dataset:** [SentiUrdu-1M](https://data.mendeley.com/datasets/rz3xg97rm5/1) — ~1,048,000 weakly-labelled Urdu tweets

This milestone implements and compares **classical, deep-learning, and transformer-based** approaches to Urdu sentiment (3-class) and emotion (6-class) classification, fulfilling the five tasks defined in `Assignment 3.docx`.

---

## Repository Layout

```
Assignment#03/
├── data/                              # Cached preprocessed splits (created at runtime)
├── outputs/
│   ├── figures/                       # All publication-quality plots
│   ├── models/                        # Trained model checkpoints
│   ├── results/                       # Metrics CSVs, predictions, classification reports
│   ├── cache/                         # Cached preprocessed data, vocabularies
│   └── embeddings/                    # fastText Urdu vectors (cc.ur.300)
├── report/                            # Compiled Markdown / LaTeX / DOCX report sources
├── 01_dataset_analysis.ipynb          # Task 1 — EDA + 12+ visualizations
├── 02_model_implementation.ipynb      # Task 3 — full model pipeline
├── 03_evaluation.ipynb                # Task 4 — metrics, confusion matrices, error analysis
├── 04_architecture_math.md            # Task 2 — architectures + mathematical modelling
├── config.py                          # Centralized hyperparameters & paths
├── preprocessing.py                   # 8-step Urdu preprocessing pipeline (from Assignment 1)
├── pyproject.toml                     # Project + dependency manifest
├── requirements.txt                   # Pip-installable dependency list
└── README.md                          # This file
```

---

## Task → Deliverable Map

| Assignment Task | Marks | Deliverable |
|---|---|---|
| Task 1 — Dataset Analysis & Statistical Exploration | 5 | `01_dataset_analysis.ipynb` + figures in `outputs/figures/` |
| Task 2 — Architecture + Mathematical Modelling | 5 + 5 | `04_architecture_math.md` |
| Task 3 — Implementation | 7 | `02_model_implementation.ipynb` + `preprocessing.py` |
| Task 4 — Experimental Results & Evaluation | 4 | `03_evaluation.ipynb` + `outputs/results/` |
| Task 5 — Technical Report | 2 + 2 | `report/Assignment3_TechnicalReport.{md,tex,docx,pdf}` |

---

## Experimental Setup

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 5070 Ti, 16 GB VRAM, CUDA 13 / driver 581.57 |
| Framework | PyTorch ≥ 2.4 + HuggingFace Transformers ≥ 4.40 |
| Tasks | Sentiment (3-class) and Emotion (6-class) — both on the ~533 K Category-labelled rows |
| Emotion labels | `Joy`, `Sad`, `Angry`, `Fear`, `Disgust`, `Surprise` (majority-vote canonicalisation of the noisy raw `Category` column) |
| Sentiment labels | `Joy → Positive`, `{Sad, Angry, Fear, Disgust} → Negative`, `Surprise → Neutral` |
| Split | 70 / 15 / 15 train/val/test, stratified, `seed=42` |
| Class imbalance | Inverse-frequency class weights in cross-entropy |
| Mixed precision | fp16 (bf16 fallback) for transformer training |
| Reproducibility | Fixed seeds across NumPy, PyTorch, scikit-learn; cached splits |

### Model Lineup (7 models × 2 tasks)

| Family | Models |
|---|---|
| Classical | Logistic Regression (TF-IDF), Linear SVM (TF-IDF) |
| Deep Learning | Multi-kernel 1D CNN, BiLSTM + attention (both with pretrained fastText Urdu embeddings) |
| Transformer | `bert-base-multilingual-cased`, `xlm-roberta-base`, `urduhack/roberta-urdu-small` |

---

## Quickstart

### 1. Create environment and install dependencies

```bash
# Recommended: create a fresh venv with Python 3.10–3.12
python -m venv .venv
.venv\Scripts\activate          # PowerShell on Windows

# Install PyTorch with CUDA 12.4 for RTX 5070 Ti
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install everything else
pip install -r requirements.txt
```

### 2. Place dataset

The notebooks expect the SentiUrdu-1M CSV at:
```
c:/Users/USER/Desktop/NLP Project/Assignment#01/Urdu Tweets Dataset.csv
```
This path is configurable in `config.py`.

### 3. Run notebooks in order

```
01_dataset_analysis.ipynb   →   outputs/figures/*.png
02_model_implementation.ipynb   →   outputs/models/, outputs/results/
03_evaluation.ipynb   →   final tables, confusion matrices, error analysis
```

Each notebook is self-contained and idempotent; cached artefacts in `outputs/cache/` are reused on subsequent runs.

---

## Reproducibility Notes

- All random seeds fixed to `42` (numpy, torch, sklearn, dataloader workers).
- Stratified splits persisted as Parquet under `outputs/cache/`.
- Trained checkpoints saved with the exact training config and tokenizer for portable inference.
- All metrics computed identically (sklearn) across all models for fair comparison.

## License & Use of AI

See `report/AI_Declaration.md` (compiled at submission time). Any AI assistance used during development is disclosed in the AI Usage Declaration.
