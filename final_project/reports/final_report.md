# Leakage-Aware Urdu Tweet Sentiment and Emotion Classification

## Abstract
This project presents a comparative study of statistical baselines, deep neural network architectures, and pre-trained multilingual transformer models for the classification of sentiment and emotions in noisy, informal Urdu tweets. Using the SentiUrdu-1M dataset, we address critical challenges including class imbalance, code-mixing, and data leakage. By implementing emoji removal, we prevent models from exploiting emoji heuristics used during dataset collection. Our experiments show that a classical TF-IDF representation combined with a Linear Support Vector Machine (SVM) remains the most robust model, achieving a Test Macro-F1 of **0.5040** and a Neutral-class F1 of **0.1303**, outperforming randomly initialized deep neural networks (BiLSTM-Attention, Text-CNN) and short-epoch multilingual transformers (XLM-RoBERTa, mBERT). We also implement a rule-based explanation assistant to interpret predictions and error categories.

---

## 1. Introduction
Sentiment and emotion classification is a crucial area of Natural Language Processing (NLP) with applications in social media monitoring, public opinion analysis, and customer feedback interpretation. While sentiment classifiers are highly mature for high-resource languages like English, low-resource and morphologically rich languages like Urdu pose major difficulties. These challenges are amplified on social media platforms like Twitter, where text is highly informal, heavily code-mixed (mixing Urdu script, Roman Urdu, and English), and grammatically unstructured. This project aims to systematically evaluate multiple model families under controlled evaluation standards to find the most viable approach for classifying noisy Urdu tweets.

---

## 2. Problem Statement
Urdu social media text is challenging due to spelling variations, right-to-left formatting problems, lack of lexical resources, code-mixing, and sarcasm. The SentiUrdu-1M dataset provides a large corpus, but its labels are derived via emoji-based weak supervision. If emojis are kept in the input, classifiers learn to associate emojis with labels directly (label leakage) instead of learning actual Urdu text semantics. Additionally, the dataset is highly imbalanced, with the `Neutral` class representing less than 0.3% of the corpus. The problem is thus to construct a leakage-aware pipeline that normalizes noisy labels, removes emojis, and evaluates models on a balanced basis using Macro-F1.

---

## 3. Motivation
Constructing robust Urdu NLP classifiers reduces the digital language barrier and provides public opinion insights in Urdu-speaking regions. Furthermore, comparing classical, deep learning, and transformer approaches under the same leakage-aware preprocessing parameters establishes clear empirical bounds for Urdu text classification. Finally, introducing a lightweight explanation assistant makes predictions interpretable for downstream end-users and course demonstrations.

---

## 4. Dataset Description
- **Dataset**: SentiUrdu-1M
- **Domain**: Twitter-style Urdu social media posts
- **Total size**: Approximately 1,048,000 raw rows
- **Usable category-labelled rows**: 533,429 rows (after filtering empty or unlabelled entries)
- **Columns**: `Id`, `Text`, `Emotions`, `Category`
- **Imbalance**: Severe class imbalance. In the final splits, the class distribution is:
  - **Positive**: 86.21%
  - **Negative**: 13.51%
  - **Neutral**: 0.28%
- **Weak Supervision**: Labels were generated via emoji heuristics, introducing substantial label noise.

---

## 5. Literature Review Summary
We reviewed 20 research papers on Urdu sentiment analysis. Prior work is broadly categorized into:
1. **Classical Machine Learning**: Traditional TF-IDF bag-of-words models combined with Naive Bayes, Support Vector Machines, and Logistic Regression. These are fast and interpretable but fail on contextual polarity or out-of-vocabulary words.
2. **Deep Learning**: Sequential networks (LSTM, GRU, BiLSTM) and convolutional models (CNN). These capture sequential context but are limited by small training datasets or randomly initialized word embeddings.
3. **Transformer-Based Models**: Fine-tuning pre-trained multilingual encoders (mBERT, XLM-R). These leverage massive multilingual pre-training but require high compute resources.
4. **Key Gap**: Most prior papers do not address label leakage caused by emojis in weakly supervised datasets. They also often evaluate models on accuracy alone, hiding poor performance on minority classes.

---

## 6. Methodology

### 6.1 Data Preprocessing
Implemented in `src/preprocessing.py`. To control label leakage and clean noisy text, the pipeline performs:
- Unicode normalization (NFC) and variant mapping.
- URL and `@mention` removal.
- Hashtag text extraction.
- **Emoji removal** (crucial to prevent the model from learning shortcuts from emoji-based labels).
- Western and Eastern Arabic-Indic number removal.
- Punctuation removal and whitespace normalization.

### 6.2 Label Normalization
Implemented in `src/label_mapping.py`. Raw surface forms under the `Category` column are normalized to canonical emotions (Joy, Sad, Angry, Fear, Disgust, Surprise). These are mapped to three sentiment classes:
- `Joy` -> **Positive**
- `Sad`, `Angry`, `Fear`, `Disgust` -> **Negative**
- `Surprise` -> **Neutral**

### 6.3 Train/Validation/Test Split
Implemented in `src/create_splits.py`. After filtering out tweets that are empty or missing labels, we split the 517,966 valid rows into stratified train (70%), validation (15%), and test (15%) splits:
- **Train split**: 362,576 rows
- **Validation split**: 77,695 rows
- **Test split**: 77,695 rows

### 6.4 Baseline Statistical Models
Implemented in `src/train_baseline.py`. A TF-IDF vectorizer (word unigrams/bigrams, 100,000 max features) is fitted on training text only. We evaluate:
- **Logistic Regression**: Linear classifier with balanced class weights and explicit One-vs-Rest wrapping.
- **Linear SVM**: Max-margin classifier suitable for high-dimensional sparse text.
- **Multinomial Naive Bayes**: A simple probabilistic baseline.

### 6.5 Neural Models
Implemented in `src/train_neural.py`. Tensors are prepared from a training-fit vocabulary of 50,000 words. We train:
- **Text-CNN**: Uses parallel convolutional kernels of sizes 3, 4, 5.
- **BiLSTM-Attention**: Learns sequential context and uses additive attention to focus on important words.
- Both use random trainable 300D embeddings, balanced class weights, and validation macro-F1 early stopping.

### 6.6 Transformer-Based Models
Implemented in `src/train_transformer.py`. We fine-tune multilingual pre-trained sequence classifiers:
- **XLM-RoBERTa** (`xlm-roberta-base`)
- **mBERT** (`bert-base-multilingual-cased`)
- Trained on a stratified training subset of 50,000 samples for 1 epoch.
- Class weights are calculated with a smoothing factor of `0.5` to prevent gradient instability from the rare Neutral class.

### 6.7 Explanation Assistant
Implemented in `src/explanation_assistant.py`. A lightweight template-based interpreter that provides plain-English descriptions of model predictions, attributes errors to specific linguistic factors (negation, text length, minority-class confusion), and generates model summaries.

---

## 7. Experimental Setup
All models are evaluated on the exact same validation (77,695) and test (77,695) splits. Training was conducted locally on an NVIDIA GeForce RTX 5070 Ti with CUDA mixed precision (`fp16`). The primary evaluation metric is **Macro-F1** to prevent the majority Positive class from distorting model comparison. Accuracy and Neutral F1 are also reported.

---

## 8. Results

### 8.1 Baseline Results
| Model | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | 0.8538 | **0.5066** | 0.8531 | **0.5040** | 0.8527 |
| **Logistic Regression** | 0.7754 | 0.4618 | 0.7740 | 0.4613 | 0.8013 |
| **Multinomial NB** | **0.8797** | 0.4046 | **0.8787** | 0.4014 | 0.8417 |

### 8.2 Neural Results
| Model | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM-Attention** | 0.7435 | **0.4557** | 0.7408 | **0.4506** | 0.7763 |
| **Text-CNN** | **0.7576** | 0.4520 | **0.7582** | 0.4476 | 0.7882 |

### 8.3 Transformer Results
| Model | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XLM-RoBERTa** | **0.8546** | **0.4392** | **0.8528** | **0.4346** | **0.8426** |
| **mBERT** | 0.8545 | 0.4285 | 0.8520 | 0.4240 | 0.8382 |

### 8.4 Final Model Comparison
| Rank | Model Name | Model Family | Test Macro-F1 | Test Accuracy | Neutral F1 | Negative F1 | Positive F1 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **linear_svm** | baseline | **0.5040** | 0.8531 | **0.1303** | **0.4662** | 0.9156 |
| 2 | **logistic_regression** | baseline | 0.4613 | 0.7740 | 0.0828 | 0.4409 | 0.8601 |
| 3 | **bilstm_attention** | neural | 0.4506 | 0.7408 | 0.0991 | 0.4181 | 0.8346 |
| 4 | **text_cnn** | neural | 0.4476 | 0.7582 | 0.0777 | 0.4164 | 0.8488 |
| 5 | **xlm_roberta** | transformer | 0.4346 | 0.8528 | 0.0000 | 0.3873 | 0.9166 |
| 6 | **mbert** | transformer | 0.4240 | 0.8520 | 0.0000 | 0.3553 | 0.9165 |
| 7 | **multinomial_nb** | baseline | 0.4014 | **0.8787** | 0.0000 | 0.2702 | **0.9339** |

---

## 9. Error Analysis
Linear SVM test errors were analyzed quantitatively:
- Total errors: 11,414 test errors (14.69% error rate).
- Class imbalance: The `Neutral` class error rate was **85.05%** (182 wrong out of 214).
- Polarity swaps: `Negative -> Positive` (5,599 cases) and `Positive -> Negative` (5,388 cases) account for **96.26%** of all errors. This indicates models confuse positive and negative expressions due to structural negation or sarcasm.
- High-confidence errors: 1,569 errors have decision-margin confidence >= 0.80, pointing to weak-label noise.

---

## 10. Ethical Considerations
- **Weak supervision bias**: Relying on emoji heuristics can introduce systemic noise. Removing emojis controls leakage but leaves labels imperfect.
- **Minority class representation**: Under-representing Neutral limits the model's fairness.
- **Dual-use concerns**: Sentiment tracking can be utilized for public surveillance or targeted advertising.

---

## 11. Deployment
The selected **Linear SVM** model is deployed in a Streamlit application (`app/streamlit_app.py`). It accepts Urdu tweet inputs, applies pre-processing, performs real-time classification and confidence score generation, provides explanations via the explanation assistant, and displays the project-wide model comparison leaderboard. The application includes robust fallbacks if transformer models are not loaded.

### 11.1 Artifact-Backed Analysis Notebooks

The notebooks are analysis notebooks that load already generated artifacts instead of retraining models. This ensures reproducibility and avoids expensive re-training during review.

```powershell
jupyter notebook notebooks
python src\validate_notebooks.py --config config.yaml
```

---

## 12. Limitations
- **Under-trained Transformers**: Transformers were trained for only 1 epoch on 50,000 samples due to resource constraints, resulting in low performance on minority classes.
- **Random Word Embeddings**: Deep learning models were trained on randomly initialized embeddings, which struggled to learn Urdu lexical semantics.
- **Rigid Explanations**: The explanation assistant is rule-based and cannot interpret semantic context or sarcasm.

---

## 13. Future Work
1. Fine-tune transformer models on the full dataset for 5-10 epochs.
2. Incorporate pre-trained Urdu word embeddings (fastText) for neural models.
3. Integrate large language models (LLMs) to generate contextual explanations.
4. Manually annotate a clean test subset to verify classification quality.

---

## 14. Conclusion
We evaluated statistical, neural, and transformer approaches for classifying noisy Urdu tweets. The classical TF-IDF Linear SVM baseline remains the strongest and most robust model (Test Macro-F1: **0.5040**), outperforming deep learning architectures and under-trained multilingual transformers. This demonstrates the difficulty of deep learning in low-resource settings with weak label noise and extreme class imbalance.

---

## References
1. SentiUrdu-1M: Dataset for Urdu Sentiment Analysis.
2. Devlin et al., 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
3. Conneau et al., 2019. Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa).
4. Kim, Y., 2014. Convolutional Neural Networks for Sentence Classification (Text-CNN).
