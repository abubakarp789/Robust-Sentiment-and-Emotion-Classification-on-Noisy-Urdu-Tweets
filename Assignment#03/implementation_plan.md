# Assignment 3 — Implementation Plan

## Project Context (From Analysis of Assignments 1 & 2)

### What You've Done So Far

**Assignment 1 — Preprocessing (Milestone 1):**
- Dataset: **SentiUrdu-1M** (~1,048,000 Urdu tweets) with columns `Id`, `Text`, `Emotions`, `Category`
- Source: [Mendeley Data](https://data.mendeley.com/datasets/rz3xg97rm5/1)
- Key stats: ~514,571 rows have `Category = NaN` (only ~533K have emotion labels)
- Built an **8-step preprocessing pipeline**: Unicode normalization → URL removal → Mention removal → Hashtag cleanup → Emoji removal (critical for label-leakage prevention) → Number removal → Punctuation removal → Whitespace normalization
- Justified excluding lowercasing, stemming, lemmatization, and stopword removal for Urdu
- Team: Raqib Hayat (NUM-BSCS-2022-40) + Abu Bakar (NUM-BSCS-2022-41)

**Assignment 2 — Literature Review (Milestone 2):**
- Reviewed **20 peer-reviewed papers** organized into 4 generations: Classical ML → Deep Learning → Transformer-based → Multimodal
- Identified **research gap**: No Urdu study trains transformers on million-scale weakly-labeled data AND evaluates on a clean held-out set AND compares all three model families (classical, deep, transformer) fairly
- Proposed approach: Compare TF-IDF + LR/SVM, CNN/BiLSTM, and fine-tuned mBERT/XLM-R on SentiUrdu-1M with clean evaluation subset

### Dataset Characteristics to Keep in Mind
- Labels are **weakly supervised** (emoticon heuristics + SentiWordNet) — inherently noisy
- `Category` column has: Joy, Sadness, Anger, Fear, Surprise, Love (emotion labels) — but ~49% are NaN
- The `Emotions` column contains emoji-based emotion descriptions with confidence scores
- Emojis **must be removed** from input features (they ARE the labeling signal → leakage)

---

## User Review Required

> [!IMPORTANT]
> **Scope Decision — Sentiment vs. Emotion Classification:**
> The `Category` column only covers ~533K rows with labels like Joy, Sadness, Anger, etc. For the remaining ~515K tweets, Category is NaN. You have two options:
> 1. **Sentiment Classification (3-class)**: Derive Positive/Negative/Neutral from the Emotions column scores (as the SentiUrdu-1M paper does). Uses the full 1M+ dataset.
> 2. **Emotion Classification (6-class)**: Use only the ~533K rows with non-null Category labels (Joy, Sadness, Anger, Fear, Surprise, Love).
>
> **Recommendation**: Do **both** — sentiment as primary task (full dataset, 3-class) and emotion as a secondary/bonus task (533K subset, 6-class). This earns bonus marks for comparative study.

> [!IMPORTANT]
> **Compute Resources:**
> Fine-tuning mBERT and XLM-R on 1M tweets requires a GPU. Options:
> 1. **Google Colab** (free tier gives ~12 hours GPU per session) — most accessible
> 2. **Kaggle Notebooks** (30 hours/week of GPU)
> 3. **Local GPU** if you have one
>
> For models on the full 1M dataset, we may need to sample a **stratified subset** (e.g., 100K–200K) to keep training feasible. Please confirm your compute setup.

> [!WARNING]
> **Dataset Size**: The CSV is ~219MB and the XLSX is ~99MB. For all notebooks, we should use the **CSV** file (much faster to load). Confirm the CSV has the same columns and content as the XLSX you used in Assignment 1.

## Open Questions

1. **Do you want to run the code locally or on Google Colab/Kaggle?** This affects how we structure file paths and GPU access.
2. **Do you want the technical report auto-generated (LaTeX/Markdown → PDF) or will you write it manually in Word?**
3. **Subset size for transformer training**: Should we use 100K, 200K, or attempt the full 1M? (Depends on GPU availability)

---

## Proposed Changes

Assignment 3 requires 5 tasks. I propose organizing them into **4 Jupyter notebooks** + **1 report document**, all within `Assignment#03/`:

```
Assignment#03/
├── data/                           # Symlink or path reference to dataset
├── outputs/                        # All generated figures, models, results
│   ├── figures/                    # Visualizations for report
│   ├── models/                     # Saved model checkpoints
│   └── results/                    # CSV files with metrics
├── 01_dataset_analysis.ipynb       # Task 1: EDA + Visualizations
├── 02_model_implementation.ipynb   # Task 3: Full pipeline implementation
├── 03_evaluation.ipynb             # Task 4: Evaluation + error analysis
├── 04_architecture_math.md         # Task 2: Architecture + math docs (for report)
├── requirements.txt                # Dependencies
├── pyproject.toml                  # uv project config
├── preprocessing.py                # Reusable preprocessing module (from Assignment 1)
└── README.md                       # Project overview
```

---

### Task 1: Dataset Analysis and Statistical Exploration (5 marks)

#### [NEW] `01_dataset_analysis.ipynb`

This notebook goes **far beyond** Assignment 1's basic exploration. It will contain:

**Section 1 — Dataset Overview & Structure**
- Load CSV, display shape, dtypes, memory usage
- Show sample rows with raw text
- Missing value analysis (especially the 49% NaN in Category)
- Duplicate detection and analysis

**Section 2 — Label Distribution Analysis**
- `Category` column value counts → bar chart (Joy, Sadness, Anger, etc.)
- Class imbalance ratio calculation
- Derived 3-class sentiment distribution (Positive/Negative/Neutral)
- Stacked comparison: emotion labels vs. derived sentiment labels

**Section 3 — Text Statistics**
- Token count distribution (histogram of word counts per tweet)
- Character length distribution (histogram)
- Average, median, min, max token counts per class
- Box plots of text length by emotion category

**Section 4 — Token Frequency Analysis**
- Top-50 most frequent tokens (after preprocessing) — bar chart
- Top-50 tokens per emotion class — grouped bar chart or word clouds
- Token frequency distribution (Zipf's law plot)
- Unique vocabulary size before/after preprocessing

**Section 5 — Emoji and Noise Analysis**
- Emoji frequency distribution (before removal)
- Percentage of tweets containing URLs, mentions, hashtags
- Code-mixing analysis (English words in Urdu tweets)
- Label-leakage analysis: correlation between emoji sentiment and assigned Category

**Section 6 — Preprocessing Impact**
- Before vs. after preprocessing text length comparison
- Empty tweets after preprocessing (percentage)
- Vocabulary reduction statistics

**Section 7 — Dataset Suitability Discussion** (markdown cells)
- Why SentiUrdu-1M is suitable for Urdu sentiment/emotion research
- Challenges: noisy labels, class imbalance, code-mixing, NaN categories
- Preprocessing decisions and their impact (referencing Assignment 1)
- Expected limitations: weak supervision bias, emoji-derived labels, domain specificity

**Visualizations** (minimum 10 high-quality plots):
1. Class distribution bar chart (emotion labels)
2. 3-class sentiment distribution pie/bar chart
3. Histogram of sentence lengths (tokens)
4. Histogram of character lengths
5. Box plot: text length by emotion class
6. Top-50 token frequency bar chart
7. Word clouds per emotion class (at least 3)
8. Zipf's law log-log plot
9. Emoji frequency distribution
10. Heatmap: class co-occurrence or confusion between emotion and sentiment labels
11. Before/after preprocessing text length comparison
12. Missing data visualization

---

### Task 2: Architecture & Mathematical Modelling (5+5 = 10 marks)

#### [NEW] `04_architecture_math.md` (to be incorporated into final report)

**Section 1 — System Architecture Overview**
- Full pipeline diagram (text input → preprocessing → feature extraction → model → prediction → evaluation)
- Component-wise breakdown for each of the 3 model families

**Section 2 — Text Preprocessing Module**
- Restate the 8-step pipeline from Assignment 1
- Mathematical notation: text transformation function $T(x) = t_8 \circ t_7 \circ \ldots \circ t_1(x)$

**Section 3 — Feature Extraction / Embedding Layer**

*Model Family 1 — TF-IDF:*
- TF formula: $\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$
- IDF formula: $\text{idf}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$
- TF-IDF: $\text{tfidf}(t, d, D) = \text{tf}(t,d) \times \text{idf}(t, D)$
- N-gram TF-IDF (unigram + bigram)

*Model Family 2 — Word Embeddings (for CNN/BiLSTM):*
- Word embedding representation: $w_i \in \mathbb{R}^d$ where $d = 300$
- Embedding matrix: $E \in \mathbb{R}^{|V| \times d}$
- Sequence representation: $X = [w_1, w_2, \ldots, w_n]$

*Model Family 3 — Contextual Embeddings (mBERT/XLM-R):*
- WordPiece/SentencePiece tokenization
- Token + Segment + Position embeddings
- $h_i = \text{Transformer}(x_1, x_2, \ldots, x_n)_i$

**Section 4 — Model Architectures**

*Logistic Regression:*
- $P(y=k|x) = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^{K} e^{w_j^T x + b_j}}$ (Softmax)
- Cross-entropy loss: $\mathcal{L} = -\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$

*SVM:*
- Optimization: $\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i} \xi_i$
- Subject to: $y_i(w^T x_i + b) \geq 1 - \xi_i$
- Kernel trick for non-linear: $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$

*CNN for Text Classification:*
- 1D convolution: $c_i = f(W \cdot x_{i:i+h-1} + b)$
- Max-over-time pooling: $\hat{c} = \max(c_1, c_2, \ldots, c_{n-h+1})$
- Multiple filter sizes (3, 4, 5) for n-gram capture

*BiLSTM:*
- LSTM cell equations:
  - Forget gate: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
  - Input gate: $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
  - Cell state: $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$, $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
  - Output gate: $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$, $h_t = o_t \odot \tanh(C_t)$
- Bidirectional: $h_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}]$

*Transformer / BERT Architecture:*
- Self-attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- Multi-head attention: $\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, \ldots, head_h)W^O$
- Feed-forward: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
- Layer normalization and residual connections
- [CLS] token pooling for classification

**Section 5 — Classification Head**
- Dense layer + Softmax: $\hat{y} = \text{softmax}(W_c h_{[CLS]} + b_c)$

**Section 6 — Loss Functions & Optimization**
- Cross-entropy loss: $\mathcal{L}_{CE} = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)$
- Adam optimizer: $\theta_{t+1} = \theta_t - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
- Learning rate scheduling (warm-up + linear decay for transformers)
- L2 regularization / weight decay

**Section 7 — Evaluation Metrics**
- Accuracy: $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
- Precision: $P_k = \frac{TP_k}{TP_k + FP_k}$
- Recall: $R_k = \frac{TP_k}{TP_k + FN_k}$
- F1: $F1_k = \frac{2 P_k R_k}{P_k + R_k}$
- Macro/Weighted averaging

---

### Task 3: Implementation (7 marks)

#### [NEW] `preprocessing.py` — Reusable Preprocessing Module

Extract the 8 preprocessing functions from Assignment 1 into a clean Python module:
- `normalize_unicode()`, `remove_urls()`, `remove_mentions()`, `clean_hashtags()`, `remove_emojis()`, `remove_numbers()`, `remove_punctuation()`, `normalize_whitespace()`
- `preprocess_tweet()` — full pipeline
- `derive_sentiment_label()` — map emotion categories to 3-class sentiment

#### [NEW] `02_model_implementation.ipynb`

**Section 1 — Setup & Data Loading**
- Load CSV → Apply preprocessing pipeline → Split into train/val/test (70/15/15, stratified)
- Handle NaN Category rows (for sentiment task: derive labels; for emotion task: drop NaN)
- Save processed splits as CSV for reproducibility

**Section 2 — Baseline Model: TF-IDF + Logistic Regression / SVM**
- TF-IDF vectorizer (unigram + bigram, max_features=50000)
- Logistic Regression (multinomial, saga solver)
- LinearSVC with calibration
- Train, predict, save results

**Section 3 — Deep Learning: CNN Text Classifier**
- Build vocabulary from training set
- Word embedding layer (randomly initialized, dim=300)
- Multi-kernel 1D-CNN (filter sizes 3,4,5; 128 filters each)
- Global max pooling → Dense → Softmax
- Training with Adam, early stopping, batch_size=256
- PyTorch implementation

**Section 4 — Deep Learning: BiLSTM**
- Same embedding layer
- 2-layer BiLSTM (hidden_size=256)
- Attention or mean pooling → Dense → Softmax
- Training with Adam, early stopping
- PyTorch implementation

**Section 5 — Transformer: Fine-tuned mBERT**
- Load `bert-base-multilingual-cased` from HuggingFace
- Custom classification head
- Training: lr=2e-5, epochs=3-5, warmup, gradient accumulation
- HuggingFace Trainer API

**Section 6 — Transformer: Fine-tuned XLM-RoBERTa**
- Load `xlm-roberta-base` from HuggingFace
- Same training setup as mBERT
- HuggingFace Trainer API

**Section 7 — Inference Pipeline**
- Load best model → Predict on test set → Save predictions
- Example predictions on sample tweets

**Frameworks/Libraries Used:**
- `scikit-learn` — TF-IDF, LR, SVM, metrics
- `PyTorch` — CNN, BiLSTM
- `transformers` (HuggingFace) — mBERT, XLM-R
- `datasets` (HuggingFace) — efficient data loading

---

### Task 4: Experimental Results & Evaluation (4 marks)

#### [NEW] `03_evaluation.ipynb`

**Section 1 — Results Summary Table**
- Comparison table: Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted)
- For both sentiment (3-class) and emotion (6-class) tasks

**Section 2 — Detailed Per-Class Metrics**
- Classification reports for each model
- Per-class precision, recall, F1

**Section 3 — Confusion Matrices**
- Heatmap confusion matrix for each model
- Normalized confusion matrices

**Section 4 — Performance Analysis & Interpretation**
- Which classes are hardest to classify and why?
- Where do models disagree?
- Impact of noisy labels on performance
- Classical vs. Deep vs. Transformer comparison discussion

**Section 5 — Error Analysis (Bonus)**
- Random sample of misclassified tweets with analysis
- Common error patterns (negation, sarcasm, code-mixing)
- Confidence distribution of correct vs. incorrect predictions

**Section 6 — Training Curves (Bonus)**
- Loss curves for deep learning models
- Validation accuracy over epochs

---

### Task 5: Technical Report (2 + 2 = 4 marks)

#### [NEW] Technical Report (PDF)

Structure:
1. Title Page
2. Abstract
3. Introduction (problem statement, objectives)
4. Dataset Description (from Task 1)
5. Related Work Summary (from Assignment 2)
6. Proposed Architecture (from Task 2)
7. Mathematical Modelling (from Task 2)
8. Implementation Details (from Task 3)
9. Experimental Results (from Task 4)
10. Discussion and Analysis
11. Conclusion and Future Work
12. References (IEEE format)

#### Additional Submission Items
- AI Usage Declaration
- Similarity/Plagiarism Report
- Contribution details (Raqib + Abu Bakar)

---

## Dependencies

```
# Core
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
wordcloud>=1.9

# NLP Preprocessing
emoji>=2.0
regex>=2023.0

# Machine Learning
scikit-learn>=1.3

# Deep Learning
torch>=2.0
torchtext>=0.15

# Transformers
transformers>=4.35
datasets>=2.16
accelerate>=0.25

# Utilities
tqdm>=4.65
openpyxl>=3.1
```

---

## Verification Plan

### Automated Tests
- Run `01_dataset_analysis.ipynb` end-to-end → all visualizations saved to `outputs/figures/`
- Run `02_model_implementation.ipynb` → models trained, predictions saved
- Run `03_evaluation.ipynb` → metrics tables and confusion matrices generated
- Verify all figures are publication quality (labeled axes, legends, titles)

### Manual Verification
- Review all 10+ visualizations for correctness and aesthetics
- Verify mathematical formulations match the implemented code
- Cross-check evaluation metrics across notebooks for consistency
- Review sample predictions for sanity

### Quality Checks
- Ensure no emoji leakage in input features
- Verify stratified splits maintain class distribution
- Confirm reproducibility with set random seeds
- Check that preprocessing module matches Assignment 1's pipeline exactly

---

## Implementation Order (Estimated Timeline)

| Step | Task | Estimated Effort |
|------|------|-----------------|
| 1 | Create project structure + `pyproject.toml` + `preprocessing.py` | 15 min |
| 2 | `01_dataset_analysis.ipynb` — full EDA with 12+ visualizations | 2-3 hours |
| 3 | `04_architecture_math.md` — architecture diagrams + math | 1-2 hours |
| 4 | `02_model_implementation.ipynb` — TF-IDF + LR/SVM | 1 hour |
| 5 | `02_model_implementation.ipynb` — CNN + BiLSTM | 2-3 hours |
| 6 | `02_model_implementation.ipynb` — mBERT + XLM-R | 2-3 hours |
| 7 | `03_evaluation.ipynb` — all metrics + confusion matrices + error analysis | 1-2 hours |
| 8 | Technical report compilation | 2-3 hours |

**Total: ~12-18 hours of implementation work**

---

## Bonus Points Strategy

The assignment lists several bonus criteria. Here's how we address each:

| Bonus Criterion | Our Approach |
|---|---|
| Advanced Transformer architectures | ✅ mBERT + XLM-R fine-tuning |
| Comparative experimentation (multiple models) | ✅ 5 models: LR, SVM, CNN, BiLSTM, mBERT, XLM-R |
| High-quality visualizations + error analysis | ✅ 12+ plots, word clouds, confusion matrices, error sample analysis |
| Multiple datasets / comparative study | ✅ Two tasks (sentiment 3-class + emotion 6-class) on same data |
| Well-optimized + reproducible | ✅ Fixed seeds, saved checkpoints, requirements.txt |
| Strong technical discussion | ✅ Noisy label analysis, cross-model comparison, per-class error analysis |
