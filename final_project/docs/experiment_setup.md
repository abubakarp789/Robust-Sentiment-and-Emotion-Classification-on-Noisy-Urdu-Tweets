# Experiment Setup

## Software

- Python project using pandas, NumPy, scikit-learn, SciPy, joblib, PyTorch, Hugging Face Transformers/Datasets, emoji, matplotlib, seaborn, PyYAML, Streamlit, and pytest.
- Project/report metadata targets Python 3.10-3.12; Python 3.11 or 3.12 is recommended for full dependency compatibility.

## Recorded Hardware

Packaged neural and Transformer metadata records:

- NVIDIA GeForce RTX 5070 Ti
- PyTorch `2.12.0.dev20260222+cu128`
- CUDA device used for the saved packaged runs

Assignment 4 also transcribes CUDA 12.8, driver 581.57, and 16 GB VRAM from Assignment 3. CPU and RAM were not verified from available project files.

## Reproducibility

- Global seed: 42
- Saved immutable CSV splits
- Training-only TF-IDF vocabulary and neural vocabulary
- Validation macro-F1 for model selection/early stopping
- Test predictions saved separately
- Config-driven relative paths in `config.yaml`

## Packaged Classical Setup

- Word TF-IDF, unigrams and bigrams
- Maximum 100,000 features
- `min_df=2`, `max_df=0.95`, sublinear TF
- Logistic Regression: balanced weights, liblinear, max 1,000 iterations
- Linear SVM: balanced weights, max 5,000 iterations
- Multinomial NB: alpha 1.0

## Packaged Neural Setup

- Vocabulary: 50,000
- Maximum sequence length: 80
- Embedding dimension: 300
- Batch size: 128
- Epochs: 5
- Learning rate: 0.001
- Class weighting and gradient clipping enabled
- Text-CNN filters: 128 with kernels 3, 4, 5
- BiLSTM hidden size: 128 with additive attention
- Packaged metadata: pretrained embeddings were not used

## Packaged Transformer Setup

- mBERT and XLM-R enabled; Urdu-RoBERTa disabled in current config
- Maximum sequence length: 96
- Batch size: 16
- Learning rate: 2e-5
- Epoch setting: 3 in config
- Weight decay: 0.01
- Warmup ratio: 0.1
- Smoothed class weights: 0.5
- Mixed precision enabled when CUDA is available

The saved packaged metadata records short mBERT/XLM-R runs, but the exact sample size/epoch override is not stored in that metadata. Existing documentation states 50,000 training examples and one epoch; treat this as repository documentation rather than independently reconstructed evidence.

## Resource Availability

- Raw data and classical models are included.
- Neural `.pt` checkpoints are not included.
- Transformer weight files are not included; tokenizer/config files are present.
- Urdu fastText vectors are not present.
- Install `requirements.txt` in a Python 3.11 or 3.12 environment before the live demonstration. The automated suite skips optional Streamlit checks when Streamlit is unavailable.
