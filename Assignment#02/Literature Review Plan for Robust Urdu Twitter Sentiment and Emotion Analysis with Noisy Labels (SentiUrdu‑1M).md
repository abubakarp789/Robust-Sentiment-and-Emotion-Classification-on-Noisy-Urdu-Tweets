# Literature Review for Robust Urdu Twitter Sentiment and Emotion Analysis with Noisy Labels (SentiUrdu‑1M)

## Overview
This report compiles 20 high-quality, peer-reviewed papers to support a semester project on robust Urdu Twitter sentiment and emotion analysis with noisy labels, centered on the SentiUrdu‑1M dataset and related resources. The selected works cover Urdu (and Roman Urdu) sentiment analysis, tweet-based emotion classification, weak/weakly supervised learning, label quality versus quantity, transformer-based methods, and large multilingual Twitter models relevant for low-resource languages.[^1][^2]

## Core SentiUrdu‑1M and Directly Related Papers

These papers are directly about SentiUrdu‑1M or use it as a primary resource for Urdu Twitter sentiment.

| ID | Reference (short) | Main focus | Data/domain | Key methods | Notes on labels & noise |
|----|-------------------|-----------|-------------|------------|-------------------------|
| P1 | SentiUrdu‑1M (PLOS ONE 2023) | Large-scale Urdu tweet dataset and weakly supervised labeling | 1.14M Urdu tweets with sentiment and emotion labels | Weakly supervised labeling using emoticons and SentiWordNet; baselines with VADER, TextBlob, and deep models | Explicitly discusses trade-off between scale and label noise; compares three automatic labeling strategies.[^1][^2] |
| P2 | Emoji-based Urdu sentiment (IEEE) | Fusion of emojis and text for Urdu tweet sentiment classification | SentiUrdu‑1M tweets plus 1,194 emojis | mBERT and XLM‑R fine-tuning; emoji2vec and FastText emoji embeddings; fusion (concatenation, NN fusion, attention) | Uses SentiUrdu‑1M labels and shows performance gains from leveraging emoji signals; demonstrates ~64–65% accuracy for text-only mBERT/XLM‑R and up to ~95% with emoji fusion for neural models.[^3] |

### Key insights for your project
- SentiUrdu‑1M provides a weakly supervised, noisy but massive training signal for Urdu sentiment and emotion analysis, making it ideal for experimentation with noise-robust learning, label cleaning, and weak supervision strategies.[^2][^1]
- Emoji-enhanced models show that auxiliary weak signals (emoji embeddings) can substantially improve classification on noisy-labeled Urdu tweets, suggesting similar strategies for robust modeling in your project.[^3]

## Urdu Tweet Emotion and Aspect-Level Work

These papers target Urdu tweets for emotions or aspect-level sentiment and provide complementary datasets, labels, and modeling techniques.

| ID | Reference (short) | Task | Data/domain | Methods | Relevance |
|----|-------------------|------|-------------|---------|-----------|
| P3 | Multi-label emotion classification of Urdu tweets (PeerJ CS 2022) | Multi-label emotion classification | Urdu tweets with multiple emotion labels | Stylometric features, pre-trained embeddings, n-grams, character n-grams, CNN features, and transformer-based BERT baselines | Provides emotion-focused annotation guidelines, multi-label evaluation (micro/macro F1, Hamming loss, exact match), and baselines relevant for extending SentiUrdu‑1M emotions.[^4][^5] |
| P4 | Weakly supervised ABSA for Urdu tweets (RANLP Stud 2023) | Aspect-based sentiment analysis (ABSA) with weak supervision | Unlabeled Urdu tweets | Joint aspect–sentiment topic embeddings with seed words; CNN/BiLSTM classifier; self-training on unlabeled data | Demonstrates weakly supervised ABSA with minimal manual labels, directly relevant to handling unlabeled or weakly labeled Urdu tweets.[^6] |
| P5 | Sentiment Analysis on Urdu Tweets Using Markov Chains (Springer 2020) | Document-level tweet sentiment | Urdu tweets | Markov-chain-based modeling of sentiment transitions | Provides a non-neural baseline and a different modeling angle; useful for understanding classical sequence-based approaches on Urdu tweets.[^7] |
| P6 | Detection of sarcasm in Urdu tweets (IEEE) | Sarcasm detection in Urdu tweets using deep and transformer-based hybrids | 12,910 manually re-annotated sarcastic/non-sarcastic tweets from an Urdu tweet corpus | Baselines with CNN, LSTM, GRU, BiLSTM, CNN-LSTM; proposed hybrid mBERT-BiLSTM with multi-head attention | Highlights challenges of sarcasm and annotation quality; shows how transformer embeddings can be combined with RNN and attention for nuanced tweet understanding.[^8] |

### Key insights for your project
- Multi-label emotion modeling on Urdu tweets shows how to design evaluation metrics (micro/macro F1, Hamming loss) for multi-label setups that could be applied to SentiUrdu‑1M emotions.[^4][^5]
- Weakly supervised ABSA techniques (seed-word-based joint embeddings + self-training) illustrate strategies for mining aspects and sentiments from large unlabeled or weakly labeled Urdu tweet corpora.[^6]
- Sarcasm detection work underscores the limitations of surface-level sentiment labels and motivates robust models that consider context and figurative language.[^8]

## Urdu Sentiment Analysis Beyond Twitter (Text, Blogs, Mixed Sources)

These works focus on Urdu (and Roman Urdu) sentiment analysis outside or beyond Twitter but offer datasets, features, and modeling techniques that can be transferred to your project.

| ID | Reference (short) | Language/script | Data source | Methods | Notes |
|----|-------------------|-----------------|------------|---------|------|
| P7 | Multi-class sentiment analysis of Urdu text using mBERT (PeerJ 2022) | Urdu | Mixed Urdu text (two datasets) | Rule-based, traditional ML (SVM, NB, etc.), deep models (CNN, LSTM, GRU, BiLSTM, Bi-GRU) and fine-tuned mBERT | Shows that fine-tuned mBERT outperforms classical and deep baselines on Urdu text, an important reference for transformer-based baselines on SentiUrdu‑1M.[^9] |
| P8 | Hybrid dependency-based approach for Urdu sentiment analysis (2023) | Urdu | Social media Urdu text | Concept-level framework combining language rules and deep neural networks | Demonstrates a hybrid rule + DNN approach that surpasses LSTM, CNN, SVM, LR, and MLP, suggesting that combining linguistic knowledge with deep models can increase robustness.[^10] |
| P9 | Innovations in Urdu sentiment analysis using ML and DL (MDPI Symmetry 2023) | Urdu and Roman Urdu | Social media (Instagram, Twitter, Facebook, etc.) | Preprocessing innovations, feature extraction, ML and DL models, and hybrid methods | Reviews and proposes several innovations in sentiment classification, offering a broad view of methods suitable for Urdu data.[^11] |
| P10 | In-depth Urdu sentiment analysis via mBERT and supervised methods (2024) | Urdu | Manually annotated Urdu blog data | Traditional classifiers (SVM, KNN, NB, LR) vs fine-tuned mBERT | Reports that mBERT reaches ~96.5% accuracy, reinforcing the value of transformer-based transfer learning for Urdu sentiment.[^12] |
| P11 | Empowering Urdu sentiment analysis with attention-based CNN-BiLSTM + mBERT (2024) | Urdu | UCSA-21 and UCSA Urdu SA datasets | Attention-based stacked two-layer CNN-BiLSTM with mBERT embeddings | Achieves ~79–83% accuracy on two Urdu datasets, showing benefit of attention and stacked BiLSTM over plain DNNs.[^13] |

### Key insights for your project
- Multiple studies confirm that multilingual BERT (mBERT) and related transformer architectures significantly outperform classical ML and vanilla deep models for Urdu sentiment classification.[^13][^12][^9]
- Hybrid approaches that combine rule-based or dependency-based linguistic features with deep learning can improve robustness, which is particularly relevant for handling noisy labels and complex phenomena in tweets.[^10]

## Roman Urdu Sentiment and Behavior Datasets

Roman Urdu (Urdu written with Latin script) is highly prevalent on Pakistani social media and offers additional perspectives on sentiment and label noise.

| ID | Reference (short) | Focus | Data | Methods | Relevance |
|----|-------------------|-------|------|---------|-----------|
| P12 | Sentiment classification of Roman-Urdu opinions using NB, DT, KNN (Elsevier 2016) | Binary sentiment classification | Blog opinions in Roman Urdu and English (150 positive, 150 negative) | Naïve Bayes, Decision Tree, KNN in WEKA | Early baseline work showing Naïve Bayes outperforming DT and KNN; useful for contrast with modern transformer-based methods.[^14] |
| P13 | Deep sentiment analysis for Roman Urdu dataset using Faster RCNN model (2022) | Sentiment analysis on Roman Urdu social media text | RUSA‑19 corpus | Faster recurrent CNN, RCNN, rule-based, n-gram models | Finds faster RCNN outperforming alternatives for binary and tertiary sentiment classification, highlighting deep CNN-based architectures for Roman Urdu.[^15] |
| P14 | Dataset construction for emotions, sentiments, mood in Roman Urdu (Data in Brief 2023) | Dataset for human behavior via emotions, sentiments, mood | Roman Urdu data from social media and other sources | Data collection and labeling methodology | Offers a structured Roman Urdu dataset with sentiment and emotion labels, useful for cross-script experiments and label quality analysis.[^16] |
| P15 | User interest identification from social data (IEEE 2022) | Topic modeling + sentiment for Pakistani fashion-related Roman Urdu tweets and reviews | 15,000 Roman Urdu tweets and 6,000 Google reviews | LDA, LSA, BERT for topics; VADER + TextBlob + DistilBERT for sentiment; K-means clustering | Combines classical and transformer-based methods on Roman Urdu data; shows practical use of DistilBERT and rule-based sentiment systems.[^17] |

### Key insights for your project
- Roman Urdu resources broaden coverage of Urdu social media language and illustrate challenges such as transliteration variation and informal spelling, similar to noisy labels in SentiUrdu‑1M.[^14][^16]
- These works provide additional datasets for potential cross-lingual or script-robust experiments if your project later extends beyond pure Urdu script.[^15][^17]

## Tweet Dataset Construction, Annotation Quality, and Label Noise

These studies emphasize dataset creation, annotation strategies, and the trade-off between label quantity and quality, which is central to your "noisy labels" focus.

| ID | Reference (short) | Language(s) | Main contribution | Relevance to label noise |
|----|-------------------|------------|-------------------|--------------------------|
| P16 | Gold standard dataset for sentiment analysis of tweets (2024) | English and Roman Urdu (multi-language) | Develops and evaluates a gold-standard sentiment tweet dataset; analyzes annotation reliability and trade-off between corpus size and label quality | Explicitly discusses annotation reliability, label suitability, and generalization from partial manual annotation, offering a conceptual basis for comparing SentiUrdu‑1M's weak labels with small gold subsets.[^18] |
| P17 | SentiUrdu‑1M dataset paper (already P1) | Urdu | Large weakly supervised tweet dataset with sentiment and emotion labels | Highlights challenges of manual labeling at scale; compares weakly supervised labeling with off-the-shelf tools like VADER and TextBlob that overproduce neutral labels.[^1][^2] |
| P18 | Dataset construction for Roman Urdu behavior (already P14) | Roman Urdu | Curates multi-emotion and sentiment dataset | Shows detailed labeling protocol and serves as an example of building emotion-rich datasets for low-resource languages.[^16] |

### Key insights for your project
- There is a clear trade-off between creating large-scale weakly supervised datasets (like SentiUrdu‑1M) and smaller, carefully annotated gold-standard corpora, and both are needed for robust modeling and evaluation.[^18][^1]
- Weak labeling strategies based on emoticons and lexicons can introduce systematic bias (e.g., over-tagging neutral), which your project can specifically target with noise-robust loss functions or label refinement.[^1][^2]

## General Twitter Sentiment and Multilingual Transformer Baselines

These works are not Urdu-specific but provide strong baselines and methodological ideas for tweet sentiment in multilingual or noisy settings.

| ID | Reference (short) | Scope | Contribution | Relevance |
|----|-------------------|-------|--------------|-----------|
| P19 | XLM‑T: Multilingual language models in Twitter for sentiment analysis (2022) | 30+ languages on Twitter | Pre-trains XLM‑R on millions of multilingual tweets; provides unified sentiment datasets and strong baselines | Offers a powerful multilingual tweet encoder that can be adapted to Urdu or used for cross-lingual transfer, especially when Urdu resources are limited.[^19] |
| P20 | M2SA: Multimodal and multilingual model for sentiment analysis of tweets (2024) | Multilingual, multimodal tweets | Converts textual Twitter sentiment dataset into multimodal format; evaluates multimodal models | Demonstrates how to extend textual Twitter sentiment analysis to images + text, relevant if you later incorporate emojis, images, or other modalities beyond plain text.[^20] |
| P21 | ASTD: Arabic Sentiment Tweets Dataset (2015) | Arabic tweets | Introduces ASTD with objective, subjective-positive, subjective-negative, and subjective-mixed labels | A mature Twitter sentiment dataset and benchmark in another low-resource Semitic language; useful as a reference for dataset design and label schema.[^21] |
| P22 | Sentiment analysis in tweets: assessment from classical to modern text representations (2021) | Multiple languages (general tweets) | Systematic comparison from BoW and classical models to BERTweet-like transformers on multiple tweet datasets | Provides a broad view of how different text representations behave on tweet sentiment, helping justify transformer choices for your own experiments.[^22] |

### Key insights for your project
- Multilingual Twitter-specific transformers like XLM‑T and strong baselines across classical and transformer models give you a reference for designing your experimental baselines and interpreting your SentiUrdu‑1M results.[^19][^22]
- Non-Urdu Twitter datasets (ASTD, others) can serve as auxiliary training sets or benchmarks in cross-lingual experiments or for pre-training Urdu-specific heads.[^21]

## Suggested Subset of 15–20 Papers for Assignment 2

For the assignment requirement of 15–20 highly relevant papers, a strong core set is:

- **Direct SentiUrdu‑1M and Urdu Twitter sentiment/emotion**: P1, P2, P3, P4, P5, P6.[^7][^5][^3][^8][^2][^4][^6][^1]
- **Urdu sentiment with transformers and hybrids (non-Twitter but transferable)**: P7, P8, P9, P10, P11.[^11][^12][^9][^13][^10]
- **Roman Urdu and behavior datasets**: P12, P13, P14, P15.[^16][^17][^15][^14]
- **Annotation quality / label noise and multilingual Twitter baselines**: P16, P19, P22 (and, optionally, P21 or P20 as bonus beyond 20).[^20][^22][^18][^19][^21]

Using this core, you can comfortably reach 18–21 papers while staying closely aligned with your project focus on Urdu Twitter sentiment, emotions, and noisy/weak labels.

## How to Structure Your Literature Review for Assignment 2

For Assignment 2, the literature review can be structured into thematic sections aligned with the course milestones and project focus:

1. **Datasets for Urdu and Roman Urdu Sentiment/Emotion on Social Media**
   - Describe SentiUrdu‑1M and its weakly supervised labeling process.[^2][^1]
   - Summarize other Urdu tweet emotion and aspect-level datasets (P3, P4, P5, P6).[^5][^8][^7][^4][^6]
   - Compare Roman Urdu datasets and behavior corpora (P12–P15, P14, P16).[^17][^15][^14][^18][^16]

2. **Methods for Urdu Sentiment and Emotion Analysis**
   - Classical ML and early approaches (P5, P12).[^14][^7]
   - Deep learning models (CNNs, RNNs, hybrids) for Urdu and Roman Urdu (P7, P8, P9, P10, P11, P13).[^12][^9][^15][^11][^13][^10]
   - Transformer-based and multilingual models (mBERT, XLM‑R, XLM‑T) (P1, P2, P3, P6, P7, P10, P11, P19).[^9][^3][^8][^4][^19][^5][^13][^12][^1][^2]

3. **Label Noise, Weak Supervision, and Annotation Strategies**
   - Weakly supervised labeling via emojis and lexicons in SentiUrdu‑1M.[^1][^2]
   - Weakly supervised ABSA with seed words and self-training (P4).[^6]
   - Trade-offs between gold-standard annotations and large noisy datasets (P16, P17, P18).[^18][^16][^2][^1]

4. **Evaluation Metrics, Experimental Protocols, and Limitations**
   - Metrics for multi-class and multi-label tasks (accuracy, micro/macro F1, Hamming loss, exact match) from P3 and related works.[^4][^5]
   - Typical train/test splits, baselines, and hyperparameters from transformer-based Urdu SA papers (P7, P10, P11, P31).**[^13][^12][^9]
   - Reported limitations: domain shift (blogs vs tweets), handling code-mixing, sarcasm, imbalanced labels, and lack of high-quality gold labels (P1, P3, P6, P16).[^8][^5][^18][^2][^4][^1]

5. **Identified Research Gaps for Your Project**
   - Limited work directly tackling **robustness to noisy labels** in Urdu sentiment, despite substantial weakly labeled resources.[^18][^2][^1]
   - Few studies combine weak supervision (e.g., seed words, emojis) with modern robust loss functions or label-noise-aware training for Urdu tweets.[^3][^6]
   - Under-explored joint modeling of **sentiment + emotion + sarcasm** in a single framework for Urdu Twitter.[^5][^8][^4]
   - Limited exploration of transfer from multilingual Twitter models (like XLM‑T) and multimodal signals (text + emoji + images) to Urdu Twitter datasets.[^19][^20]

By organizing the literature along these themes, you can clearly compare datasets, methods, evaluation strategies, and limitations, and then motivate your own contribution: robust sentiment and emotion analysis for Urdu tweets under noisy labels using SentiUrdu‑1M.

---

## References

1. [SentiUrdu-1M: A large-scale tweet dataset for Urdu text sentiment analysis using weakly supervised learning](https://dx.plos.org/10.1371/journal.pone.0290779) - Low-resource languages are gaining much-needed attention with the advent of deep learning models and...

2. [SentiUrdu-1M: A large-scale tweet dataset for Urdu text sentiment analysis using weakly supervised learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468080/) - ...attracting a lot of attention and support from the research community. One challenge faced by suc...

3. [Enhancing Emoji-Based Sentiment Classification in Urdu Tweets: Fusion Strategies With Multilingual BERT and Emoji Embeddings](https://ieeexplore.ieee.org/document/10643088/) - X (formerly known as Twitter) is a popular social network with hundreds of millions of users. We emp...

4. [Multi-label emotion classification of Urdu tweets](https://peerj.com/articles/cs-896) - ...CNN features) and transformer-based baseline (BERT). We used a combination of text representation...

5. [Multi-label emotion classification of Urdu tweets](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044368/) - ...CNN features) and transformer-based baseline (BERT). We used a combination of text representation...

6. [Weakly supervised learning for aspect based sentiment analysis of Urdu Tweets](https://acl-bg.org/proceedings/2023/RANLPStud%202023/pdf/2023.ranlpstud-1.9.pdf) - Aspect-based sentiment analysis (ABSA) is vital for text comprehension which benefits applications a...

7. [Sentiment Analysis on Urdu Tweets Using Markov Chains](https://link.springer.com/10.1007/s42979-020-00279-9)

8. [Detection of Sarcasm in Urdu Tweets Using Deep Learning and Transformer Based Hybrid Approaches](https://ieeexplore.ieee.org/document/10508575/) - Sarcasm has a significant role in human communication especially on social media platforms where use...

9. [Multi-class sentiment analysis of urdu text using multilingual BERT](https://pmc.ncbi.nlm.nih.gov/articles/PMC8971433/) - ...results using rule-based, machine learning (SVM, NB, Adabbost, MLP, LR and RF) and deep learning ...

10. [A hybrid dependency-based approach for Urdu sentiment analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10716113/) - ... demonstrate that the proposed framework surpasses state-of-the-art approaches, including LSTM, C...

11. [Innovations in Urdu Sentiment Analysis Using Machine and Deep Learning Techniques for Two-Class Classification of Symmetric Datasets](https://www.mdpi.com/2073-8994/15/5/1027) - Many investigations have performed sentiment analysis to gauge public opinions in various languages,...

12. [In-depth Urdu Sentiment Analysis Through Multilingual BERT and Supervised Learning Approaches](https://www.icck.org/article/abs/tis.2024.585616) - Sentiment analysis is the process of identifying and categorizing opinions expressed in a piece of t...

13. [Empowering Urdu sentiment analysis: an attention-based stacked CNN-Bi-LSTM DNN with multilingual BERT](https://link.springer.com/10.1007/s40747-024-01631-9) - Sentiment analysis (SA) as a research field has gained popularity among the researcher throughout th...

14. [Sentiment classification of Roman-Urdu opinions using Naïve Bayesian, Decision Tree and KNN classification techniques](https://linkinghub.elsevier.com/retrieve/pii/S1319157815001330) - Sentiment mining is a field of text mining to determine the attitude of people about a particular pr...

15. [Deep Sentiments Analysis for Roman Urdu Dataset Using Faster Recurrent Convolutional Neural Network Model](https://www.tandfonline.com/doi/full/10.1080/08839514.2022.2123094) - ABSTRACT Urdu language is being spoken by over 64 million people and its Roman script is very popula...

16. [Dataset construction to detect human behavior with the help of emotions, sentiments and mood for Roman Urdu](https://linkinghub.elsevier.com/retrieve/pii/S2352340923009496)

17. [A Machine Learning based Approach to Identify User Interests from Social Data](https://ieeexplore.ieee.org/document/9972956/) - Social media platforms like Twitter, Facebook, Instagram, etc., are considered a common source of ex...

18. [Development and Evaluation of Gold Standard Dataset for Sentiment Analysis of Tweets](https://jucmd.pk/pakjet/article/view/2563) - Pre-labeled data is typically required for supervised machine learning. A limited number of object c...

19. [XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis
  and Beyond](https://arxiv.org/pdf/2104.12250.pdf) - ...In this paper we provide: (1)
a new strong multilingual baseline consisting of an XLM-R (Conneau ...

20. [M2SA: Multimodal and Multilingual Model for Sentiment Analysis of Tweets](https://arxiv.org/pdf/2404.01753.pdf) - ...needs to be more clarity when it comes to analysing multimodal tasks in
multi-lingual contexts. W...

21. [ASTD: Arabic Sentiment Tweets Dataset](https://www.aclweb.org/anthology/D15-1299.pdf) - This paper introduces ASTD, an Arabic social sentiment analysis dataset gathered from Twitter. It co...

22. [Sentiment analysis in tweets: an assessment study from classical to
  modern text representation models](https://arxiv.org/pdf/2105.14373.pdf) - ...come from simple count-based methods, such as bag-of-words, to
more sophisticated ones, such as B...

