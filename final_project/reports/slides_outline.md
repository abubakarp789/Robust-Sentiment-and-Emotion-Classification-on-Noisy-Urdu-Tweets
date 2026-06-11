# Presentation Slides Outline

This document details a 12-slide outline for presenting the **Leakage-Aware Urdu Tweet Sentiment and Emotion Classification** project.

---

## Slide 1: Title
- **Slide Title**: Leakage-Aware Urdu Tweet Sentiment and Emotion Classification
- **Subtitle**: Re-evaluating Classifiers and Label Leakage on SentiUrdu-1M
- **Bullet Points**:
  - Abu Bakar & M. Raqib Hayat
  - Course: CSC-355 Natural Language Processing
  - Namal University Mianwali
- **Suggested Visual**: Premium clean typography with a background gradient (e.g. HSL tailored color palette or modern dark mode style).
- **Speaker Notes**:
  > Welcome everyone. Today we are presenting our final NLP semester project. Our work centers on classifying sentiment and emotions in noisy Urdu social media text. We focus on evaluating classical, neural, and transformer architectures under a leakage-aware data pipeline.

---

## Slide 2: Problem Background
- **Slide Title**: Challenges of Urdu Social Media NLP
- **Bullet Points**:
  - **Script Complexity**: Right-to-left formatting and lexical variation in Urdu.
  - **Noisy & Informal Writing**: Code-mixing (mixing Roman Urdu, Urdu script, and English terms).
  - **Label Leakage**: Weakly supervised labels created via emoji heuristics can lead to models learning shortcuts instead of Urdu text semantics.
  - **Imbalance**: Minority classes (e.g., Neutral) are extremely scarce, distorting traditional metric reporting.
- **Suggested Visual**: A diagram illustrating the flow of a noisy code-mixed tweet containing emojis and Roman Urdu words, pointing to the risk of label leakage.
- **Speaker Notes**:
  > Urdu social media text is highly unstructured and code-mixed. A major challenge in existing datasets like SentiUrdu-1M is that labels were weakly supervised using emojis. If we leave emojis in the input text, the models learn the emoji shortcut rather than actual Urdu sentiment, creating a leakage issue.

---

## Slide 3: Dataset Summary
- **Slide Title**: The SentiUrdu-1M Corpus
- **Bullet Points**:
  - **Domain**: Twitter-style Urdu social media posts.
  - **Total Raw Size**: 1,048,000 rows.
  - **Usable Filtered Rows**: 533,429 category-labeled rows.
  - **Severe Imbalance**:
    - Positive: 86.21%
    - Negative: 13.51%
    - Neutral: 0.28% (highly scarce)
- **Suggested Visual**: A pie chart or bar chart showing the severe class distribution, emphasizing the minority Neutral class size of 0.28%.
- **Speaker Notes**:
  > We utilize the SentiUrdu-1M corpus. After cleaning empty or unlabelled tweets, we have 533,429 rows. The data exhibits severe class imbalance. Positive tweets represent over 86% of the dataset, whereas Neutral represents a mere 0.28%. This imbalance requires us to focus on Macro-F1 rather than accuracy.

---

## Slide 4: Preprocessing & Label Normalization
- **Slide Title**: Leakage-Aware Pipeline
- **Bullet Points**:
  - **Emoji Removal**: Essential to strip the emoji heuristic and prevent label leakage.
  - **Standardization**: Unicode NFC normalization, variant mapping, punctuation, URL, and mention removal.
  - **Label Mapping**: Noise-reduction mapping from messy raw labels into Positive, Negative, and Neutral.
  - **Stratified Split**: Reproducible 70/15/15 train (362,576), validation (77,695), and test (77,695) splits.
- **Suggested Visual**: A flowchart mapping: Raw Tweet -> Unicode Normalization -> Emoji Removal -> Label Normalization -> Split Generation.
- **Speaker Notes**:
  > To address leakage, we built a preprocessing pipeline that enforces emoji removal. Emojis must be removed so the models learn linguistic features. We also normalize raw labels to canonical sentiment classes and split the corpus into stratified train, validation, and test subsets.

---

## Slide 5: Project Milestones
- **Slide Title**: Course Milestone Progress
- **Bullet Points**:
  - **Milestone 1**: Problem definition, dataset card, and clean split generation.
  - **Milestone 2**: Classical baselines (TF-IDF + Logistic Regression, SVM, Naive Bayes).
  - **Milestone 3**: Deep Learning (Text-CNN and BiLSTM-Attention with random embeddings).
  - **Milestone 4**: Multilingual Transformers (mBERT and XLM-RoBERTa fine-tuning).
  - **Milestone 5**: Combined evaluation, error analysis, Streamlit app, and final packaging.
- **Suggested Visual**: A timeline or progress bar showing the five milestones, highlighting the comprehensive structure of the final package.
- **Speaker Notes**:
  > Our project is organized into five course milestones. We started with data split generation, implemented classical baselines, built PyTorch neural networks, fine-tuned multilingual transformer models, and finally packaged everything with evaluations, explanations, and a Streamlit app.

---

## Slide 6: Milestone 2: Baseline Models
- **Slide Title**: Classical Machine Learning Baselines
- **Bullet Points**:
  - **Feature Representation**: Word-level TF-IDF (unigrams & bigrams, 100,000 max features) fit only on training split.
  - **Models**:
    - **Logistic Regression**: Linear classifier with balanced weights.
    - **Linear SVM**: Max-margin classifier. Best baseline model (Test Macro-F1: 0.5040).
    - **Multinomial Naive Bayes**: High accuracy (0.8787) but fails on Neutral (0.0 F1).
  - **Fit Audit**: Validation metadata verifies zero leakage of evaluation splits during fitting.
- **Suggested Visual**: A table showing baseline validation and test results, highlighting the gap between Naive Bayes accuracy and its low Macro-F1.
- **Speaker Notes**:
  > For our Milestone 2 baselines, we fit a word TF-IDF vectorizer on training text only. We trained Logistic Regression, Linear SVM, and Multinomial Naive Bayes. The Linear SVM emerged as the best baseline with a Test Macro-F1 of 0.5040, proving to be a highly competitive model.

---

## Slide 7: Milestone 3: Neural Models
- **Slide Title**: Deep Learning Implementations
- **Bullet Points**:
  - **Embeddings**: Randomly initialized trainable 300D word embeddings.
  - **Text-CNN**: Convolutional kernels of sizes 3, 4, 5 to capture local phrase patterns.
  - **BiLSTM-Attention**: Learns sequential dependencies and uses additive attention to focus on important words.
  - **Results**: Best neural model is **BiLSTM-Attention** (Test Macro-F1: 0.4506).
  - **Limitations**: Over-corrects toward Negative, and random embeddings fail to capture deep Urdu semantics.
- **Suggested Visual**: Schematic architecture diagram of the BiLSTM-Attention model, showing the Embedding layer, BiLSTM layer, Attention weights, and Softmax output.
- **Speaker Notes**:
  > In Milestone 3, we implemented Text-CNN and BiLSTM-Attention in PyTorch. The BiLSTM-Attention model achieved 0.4506 Test Macro-F1. However, since the embeddings were trained from scratch on noisy labels, the neural models did not beat the TF-IDF Linear SVM.

---

## Slide 8: Milestone 4: Transformer Models
- **Slide Title**: Multilingual Transformer Fine-Tuning
- **Bullet Points**:
  - **Models**: mBERT and XLM-RoBERTa (pretrained multilingual encoders).
  - **Training Setup**: Fine-tuned on a 50,000-sample training subset for 1 epoch.
  - **Class Imbalance**: Cross-entropy loss weighted with smoothed class weights (`class_weight_smoothing = 0.5`).
  - **Results**: Best transformer model is **XLM-RoBERTa** (Test Macro-F1: 0.4346, Test Accuracy: 0.8528).
  - **Limitations**: Short training epoch and subset size prevent full vocabulary adaptation to noisy Roman/Arabic Urdu script.
- **Suggested Visual**: A comparison chart of training epochs vs. Validation Macro-F1 for mBERT and XLM-RoBERTa.
- **Speaker Notes**:
  > In Milestone 4, we fine-tuned mBERT and XLM-RoBERTa. Due to resource constraints, we fine-tuned on a 50,000 sample subset for 1 epoch. XLM-RoBERTa achieved a Test Macro-F1 of 0.4346. The short fine-tuning duration limited their capacity to capture Urdu tweet structure.

---

## Slide 9: Final Results Comparison
- **Slide Title**: Combined Leaderboard
- **Bullet Points**:
  - **Top Performer**: **Linear SVM** remains the best overall classifier (Test Macro-F1: **0.5040**, Test Accuracy: **0.8531**).
  - **Class Imbalance**: Neutral remains the hardest class to classify due to extreme data scarcity.
  - **Key Lesson**: Model complexity does not automatically translate to better performance under severe class imbalance and label noise.
- **Suggested Visual**: A horizontal bar chart comparing Test Macro-F1 scores across all seven models, color-coded by model family (baseline, neural, transformer).
- **Speaker Notes**:
  > Comparing all models, the classical TF-IDF Linear SVM remains the top classifier with 0.5040 Macro-F1. Accuracy is highly misleading; for instance, Naive Bayes has 87.87% accuracy but only 0.4014 Macro-F1. This highlights the importance of using balanced metrics like Macro-F1.

---

## Slide 10: Error Analysis & Ethics
- **Slide Title**: Failure Analysis and Ethical Risks
- **Bullet Points**:
  - **Polarity Swaps**: Polarity swaps (Positive/Negative confusion) account for **96.26%** of all SVM errors.
  - **Neutral Failure**: All models struggle with Neutral. SVM achieved 0.1303 F1, while transformers achieved 0.0 F1.
  - **Ethical Risks**: Model bias against minority classes, propagation of weak emoji labels, and surveillance concerns.
- **Suggested Visual**: A confusion matrix heatmap for the best model (Linear SVM) highlighting the major confusions between Positive and Negative.
- **Speaker Notes**:
  > Our error analysis showed that polarity swaps represent over 96% of all errors. Furthermore, all models struggle with the Neutral class due to class scarcity. Ethically, we must be careful when using classifiers trained on weakly supervised emoji-based labels, as predictions may propagate dataset biases.

---

## Slide 11: Deployment Demo
- **Slide Title**: Streamlit Web Application
- **Bullet Points**:
  - **Inference Pipeline**: Preprocesses text (stripping emojis) and serves predictions.
  - **Model Fallback**: Attempts to load the selected model; falls back to the robust Linear SVM if model files are missing.
  - **Explanation Assistant**: A rule-based interpretability layer that explains predictions and details likely error categories.
  - **Leaderboard**: Displays the live model comparison leaderboard.
- **Suggested Visual**: Screenshot of the Streamlit application interface, showing the input area, prediction result, and explanation assistant note.
- **Speaker Notes**:
  > We deployed our pipeline in a Streamlit application. Users can enter an Urdu tweet, select a model, and view predictions. The app includes a rule-based explanation assistant to explain predictions and errors. It also includes robust fallbacks to Linear SVM if transformer models are missing.

---

## Slide 12: Conclusion & Future Work
- **Slide Title**: Summary and Future Directions
- **Bullet Points**:
  - **Main Finding**: Sparse n-gram representations with Linear SVM provide the most effective and stable classifier for noisy SentiUrdu-1M tweets.
  - **Neutral Class Scarcity**: Minority class scarcity remains a major challenge.
  - **Future Work**:
    - Fine-tune transformers on the full dataset for more epochs.
    - Incorporate pre-trained Urdu word embeddings (fastText).
    - Annotate a high-quality human-verified test split to validate labels.
- **Suggested Visual**: Visual summary slide with a concluding quote or a screenshot of the codebase structure.
- **Speaker Notes**:
  > In conclusion, the classical TF-IDF Linear SVM remains the most effective model for this task. For future work, we recommend fine-tuning transformers on the full dataset for longer epochs, utilizing pre-trained Urdu fastText embeddings, and creating a manually verified test subset. Thank you.

---
