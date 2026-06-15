# Dual-Task Benchmark Design

## Objective

Produce one defensible final-project benchmark for Urdu sentiment and emotion classification, with separate artifacts for each task and no cross-split duplicate leakage. Retrain every implemented model and generate the report and demo from the resulting machine-readable artifacts.

## Scope

The official benchmark includes eight models:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Text-CNN
- BiLSTM with additive attention
- mBERT
- XLM-RoBERTa
- Urdu-RoBERTa

Classical and neural models run with seeds 42, 52, and 62. Transformers run with seed 42 because of the 24-36 hour wall-clock constraint. Transformer runs are explicitly described as resource-constrained rather than budget-matched comparisons.

## Data Design

Preprocess and normalize the source once per task. Build connected duplicate groups so rows sharing either a tweet ID or normalized text cannot cross split boundaries. Remove groups that contain conflicting task labels. Deduplicate identical normalized text within the remaining groups, then assign groups to deterministic stratified 70/15/15 train, validation, and test partitions.

The pipeline must assert zero overlap for tweet IDs and normalized text across all split pairs. Sentiment and emotion data are stored independently under `data/processed/<task>/` and `data/splits/<task>/`.

## Artifact Layout

Each task owns its complete artifact tree:

```text
outputs/<task>/
|-- figures/
|-- results/
|   |-- runs/<family>/<model>/seed_<seed>/
|   |-- aggregate_metrics.json
|   `-- model_comparison_leaderboard.csv
|-- predictions/
|-- models/
`-- error_analysis/
```

No command may overwrite the other task's outputs. Seed-specific model and result directories make reruns auditable.

## Evaluation Protocol

- Fit feature extractors and vocabularies on training data only.
- Use validation macro-F1 for checkpointing and model ranking.
- Aggregate classical and neural results as mean and sample standard deviation across three seeds.
- Include bootstrap 95% confidence intervals for the final selected model on test predictions.
- Evaluate the final selected model on test only after validation ranking is fixed.
- Report per-class precision, recall, F1, support, confusion matrices, runtime, hardware, sample sizes, config hash, data hashes, git commit, and package versions.
- Display SVM outputs as decision scores unless a calibrated classifier is trained.
- State clearly that labels are weakly supervised and no human-gold evaluation is claimed.

## Training Budgets

Classical and neural models use the complete group-safe training split. Neural models use validation macro-F1 early stopping. Transformers use a deterministic stratified training subset sized to fit the available 24-36 hour budget, one seed, mixed precision, and validation checkpointing. The exact sample size and epochs are recorded in metadata and reports.

## Inference and Demo

The inference API accepts a task and model name. It supports baseline, neural, and Transformer checkpoints. The Streamlit app lets the user switch between sentiment and emotion, displays only models with valid task artifacts, shows preprocessing, predicted class, calibrated probability or clearly labeled decision score, task-specific leaderboards, and limitations.

## Documentation

Generate a current technical report, evaluation summary, dataset card, model card, experiment manifest, and presentation content from official aggregate artifacts. The report includes the full related-work bibliography, group-safe split method, repeated-seed statistics, resource constraints, weak-label limitation, and exact commands. Export the final report to PDF and presentation to PPTX/PDF if the local document runtime supports it.

## Testing and Acceptance

The implementation is accepted only when:

- split tests prove zero ID and normalized-text overlap;
- task outputs cannot overwrite each other;
- validation ranking is independent of test metrics;
- aggregate statistics match seed runs;
- inference works for every packaged model family on both tasks;
- all existing and new tests pass;
- all validators pass;
- all reported numbers reproduce from saved predictions;
- the Streamlit health endpoint and representative predictions work.

## Constraints

The project cannot claim a guaranteed instructor score. It will provide complete, internally consistent, reproducible evidence targeting every rubric item. No human-gold benchmark will be fabricated or implied.
