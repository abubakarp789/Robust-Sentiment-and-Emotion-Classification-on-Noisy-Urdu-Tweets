# Demonstration Guide

## Reliable Demo Path

Use the included TF-IDF + Linear SVM artifact. Neural and Transformer weights are not included, so those selections should be presented through saved predictions/figures unless the weights are restored locally.

From `final_project`:

```powershell
python -m pytest tests
python scripts\05_evaluate_models.py --pattern "baseline_linear_svm_test_predictions.csv"
python scripts\06_generate_visualizations.py
streamlit run app\streamlit_app.py
```

If Streamlit is not installed, install `requirements.txt` in Python 3.11/3.12 before the presentation. Do not install or download resources during the live grading session.

## What to Show

1. `README.md`: title, problem, two evidence snapshots, results, limitations.
2. `src/preprocessing.py`: emoji, URL, mention, hashtag, number, punctuation, and whitespace handling.
3. `src/label_mapping.py`: `Surprice` correction, six emotions, three sentiments.
4. `outputs/results/split_summary.json`: packaged counts and 70/15/15 split.
5. `outputs/results/model_comparison_leaderboard.csv`: packaged model comparison.
6. `outputs/reports/final_nlp_project_report.pdf`: dual-task final report tables and figures.
7. `outputs/figures/macro_f1_vs_weighted_f1.png`: imbalance explanation.
8. Streamlit: enter a tweet with an emoji and show cleaned text plus prediction.

## Suggested Live Flow

1. State the problem and why Urdu tweets are noisy.
2. Explain that emoji-derived labels create leakage risk.
3. Run or show preprocessing on an emoji-containing input.
4. Explain canonical emotion labels and sentiment mapping.
5. Show the split summary and class imbalance.
6. Compare classical, neural, and Transformer model families.
7. Contrast accuracy with macro-F1.
8. Show one confusion matrix and one error sample.
9. End with limitations: weak labels, no gold set, single seed, missing deep weights.

## Likely Questions

**Why remove emojis if they improve accuracy?**<br>
Because emojis helped generate the weak labels. Keeping them lets the model reproduce the annotation heuristic rather than infer sentiment from Urdu text.

**Why is accuracy misleading?**<br>
Positive/Joy dominates. A model can predict the majority class often and achieve high accuracy while failing Neutral and rare emotions. Macro-F1 weights each class equally.

**Why compare classical, neural, and Transformer models?**<br>
The comparison exposes trade-offs. Classical models are fast and deployable; neural models learn sequence/local patterns; Transformers provide contextual subword representations but cost more and still depend on label quality.

**Why did Linear SVM beat packaged Transformers?**<br>
The packaged Transformer runs were resource-constrained, the labels are noisy, and short tweets often reward strong lexical margins. It does not prove SVM universally beats Transformers.

**Why do Assignment 4 and final_project numbers differ?**<br>
Assignment 4 uses the earlier Assignment 3 dual-task snapshot with 532,661 rows. The packaged rerun applies a stricter two-token filter, uses 517,966 rows, and reruns sentiment. The project documents both rather than mixing them.

**Where are neural/Transformer checkpoints?**<br>
Large `.pt`, `.safetensors`, and `.bin` files are gitignored and absent. Predictions, metrics, histories, tokenizers/configs, and training code remain available. The included Linear SVM is the runnable demo model.

**What would improve the research most?**<br>
A native-speaker gold test set, repeated seeds, uncertainty estimates, and noise-robust training.
