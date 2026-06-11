# Ethics and Limitations

## Dataset Bias

The dataset comes from Twitter-style Urdu social media. It may overrepresent specific demographics, regions, political discussions, religious expressions, humor styles, and platform-specific writing habits. A model trained on this dataset may not generalize to news, literature, customer reviews, formal Urdu, or spoken Urdu transcripts.

## Weak Labels

The labels are weakly supervised and influenced by emoji and lexicon heuristics. This means the labels are not guaranteed to represent the true sentiment or emotion of each tweet. Some labels may reflect the presence of an emoji rather than the semantic meaning of the text.

## Privacy Concerns

Tweets can contain personal information, usernames, mentions, locations, or sensitive opinions. Even if the dataset is public, responsible use requires avoiding re-identification, unnecessary display of private text, and harmful profiling.

## Offensive Language Exposure

Social-media datasets may include offensive language, harassment, hate speech, political hostility, or distressing content. Developers, annotators, and demo users should be warned that such content may appear during exploration or error analysis.

## Misclassification Risks

Sentiment and emotion models can misread sarcasm, negation, religious or poetic language, code-mixed text, and ambiguous statements. Incorrect predictions could be harmful if used for moderation, surveillance, student assessment, employee monitoring, or public-opinion decisions.

## Minority Class Harm

Rare classes such as Fear, Surprise, and Angry may receive poor recall or precision because of class imbalance. If only accuracy is reported, these harms can be hidden by strong performance on the majority Joy/Positive class.

## Transparency

The final system should clearly state:

- It is trained on weak labels.
- Predictions are probabilistic estimates, not ground truth.
- The model can fail on rare classes.
- Emoji leakage has been controlled through preprocessing.
- The system is intended for research and educational use.

## Responsible Use

This project should be used for academic NLP research, model comparison, and educational demonstration. It should not be used as the sole basis for high-stakes decisions about people, communities, political groups, or public sentiment.

## Limitations

- No fully verified clean test set is currently included.
- Model results may reflect noisy-label agreement rather than true human sentiment.
- The dataset is highly imbalanced.
- Urdu code-mixing and Roman Urdu are not fully normalized.
- Sarcasm and implicit emotion are difficult for all current models.
- The final-project Streamlit app is fully integrated with baseline and transformer inference pipelines, offering template-based explanations for predictions and error analysis.
