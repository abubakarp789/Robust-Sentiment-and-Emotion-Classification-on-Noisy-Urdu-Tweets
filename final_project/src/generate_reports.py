"""Generate final reports and figures from official aggregate experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_DISPLAY = {
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression",
    "multinomial_nb": "Multinomial NB",
    "text_cnn": "Text-CNN",
    "bilstm_attention": "BiLSTM-Attention",
    "mbert": "mBERT",
    "xlm_roberta": "XLM-RoBERTa",
    "urdu_roberta": "Urdu-RoBERTa",
}
FAMILY_METRICS = {
    "baseline": "baseline_metrics.json",
    "neural": "neural_metrics.json",
    "transformer": "transformer_metrics.json",
}
REFERENCES = [
    'M. T. Ali et al., "SentiUrdu-1M: A large-scale weakly-labelled Urdu Twitter dataset," Data in Brief, 2023.',
    'A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.',
    'J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.',
    'A. Conneau et al., "Unsupervised Cross-Lingual Representation Learning at Scale," ACL, 2020.',
    'Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv:1907.11692, 2019.',
    'Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP, 2014.',
    'S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.',
    'D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," ICLR, 2015.',
    'M. Schuster and K. K. Paliwal, "Bidirectional Recurrent Neural Networks," IEEE Transactions on Signal Processing, 1997.',
    'T. Mikolov et al., "Advances in Pre-Training Distributed Word Representations," LREC, 2018.',
    'T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space," ICLR Workshop, 2013.',
    'J. Pennington, R. Socher, and C. D. Manning, "GloVe: Global Vectors for Word Representation," EMNLP, 2014.',
    'F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, 2011.',
    'T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," EMNLP Demos, 2020.',
    'I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019.',
    'D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," ICLR, 2015.',
    'P. Micikevicius et al., "Mixed Precision Training," ICLR, 2018.',
    'C. Cortes and V. Vapnik, "Support-Vector Networks," Machine Learning, 1995.',
    'J. Platt, "Probabilistic Outputs for Support Vector Machines," Advances in Large-Margin Classifiers, 1999.',
    'G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," Information Processing and Management, 1988.',
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _selected_metrics(project_root: Path, task: str, aggregate: dict[str, Any]) -> dict[str, Any]:
    selected = aggregate["selected_model"]
    family = selected["model_family"]
    model = selected["model_name"]
    seed = selected["canonical_seed"]
    path = (
        project_root
        / "outputs"
        / task
        / "runs"
        / family
        / model
        / f"seed_{seed}"
        / "results"
        / FAMILY_METRICS[family]
    )
    return _load_json(path)["models"][model]


def _leaderboard_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Rank | Family | Model | Seeds | Validation Macro-F1 | Test Macro-F1 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.model_family} | `{row.model_name}` | {row.seed_count} | "
            f"{row.validation_macro_f1_mean:.4f} +/- {row.validation_macro_f1_std:.4f} | "
            f"{row.test_macro_f1_mean:.4f} +/- {row.test_macro_f1_std:.4f} |"
        )
    return "\n".join(lines)


def _class_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in metrics["test"]["per_class"].items():
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {int(values['support']):,} |"
        )
    return "\n".join(lines)


def _collect_runs(project_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for task in ("sentiment", "emotion"):
        root = project_root / "outputs" / task / "runs"
        for family, filename in FAMILY_METRICS.items():
            for path in sorted(root.glob(f"{family}/*/seed_*/results/{filename}")):
                model = path.parents[2].name
                seed = int(path.parents[1].name.removeprefix("seed_"))
                payload = _load_json(path)
                metrics = payload["models"][model]
                model_dir = path.parents[1] / "models"
                artifacts = [
                    {
                        "path": str(item.relative_to(project_root)).replace("\\", "/"),
                        "bytes": item.stat().st_size,
                    }
                    for item in sorted(model_dir.rglob("*"))
                    if item.is_file()
                ]
                runs.append(
                    {
                        "task": task,
                        "family": family,
                        "model": model,
                        "seed": seed,
                        "validation": metrics["validation"],
                        "test": metrics["test"],
                        "metrics_path": str(path.relative_to(project_root)).replace("\\", "/"),
                        "artifacts": artifacts,
                    }
                )
    return runs


@lru_cache(maxsize=2)
def build_report_bundle(project_root_value: str | Path) -> dict[str, str]:
    """Build all final report text from official machine-readable artifacts."""
    project_root = Path(project_root_value).resolve()
    task_data: dict[str, dict[str, Any]] = {}
    for task in ("sentiment", "emotion"):
        results = project_root / "outputs" / task / "results"
        aggregate = _load_json(results / "aggregate_metrics.json")
        split = _load_json(results / "split_summary.json")
        leaderboard = pd.read_csv(results / "model_comparison_leaderboard.csv")
        task_data[task] = {
            "aggregate": aggregate,
            "split": split,
            "leaderboard": leaderboard,
            "selected_metrics": _selected_metrics(project_root, task, aggregate),
        }

    sentiment = task_data["sentiment"]
    emotion = task_data["emotion"]
    sent_selected = sentiment["aggregate"]["selected_model"]
    emo_selected = emotion["aggregate"]["selected_model"]
    sent_split = sentiment["split"]
    emo_split = emotion["split"]

    references = "\n".join(f"{index}. {reference}" for index, reference in enumerate(REFERENCES, 1))
    report = f"""# Group-Safe Urdu Tweet Sentiment and Emotion Classification

## Abstract

This project presents a reproducible comparison of classical machine-learning, neural-network, and Transformer approaches for sentiment and emotion classification on noisy Urdu tweets. SentiUrdu-1M labels are weakly supervised and partly derived from emoji signals, so emojis are removed from model input and duplicate-linked tweet IDs and normalized texts are assigned to only one split. Conflicting duplicate-label groups are excluded. The official benchmark evaluates eight models on separate three-class sentiment and six-class emotion pipelines. Classical and neural models are repeated with seeds 42, 52, and 62; resource-constrained Transformers use seed 42, 50,000 training rows, and one epoch. Models are ranked only by mean validation macro-F1. Linear SVM is selected for both tasks, reaching test macro-F1 **{sent_selected['test_macro_f1_mean']:.4f}** for sentiment and **{emo_selected['test_macro_f1_mean']:.4f}** for emotion. No human-gold evaluation is claimed because independent Urdu annotators were unavailable.

## Introduction

Urdu social-media classification is difficult because tweets mix Urdu script, Roman Urdu, English, informal spelling, hashtags, sarcasm, negation, and limited context. The corpus is also severely imbalanced and weakly labeled. These properties make accuracy alone unreliable and create a complex computational problem requiring data-quality controls, competing model families, resource-aware optimization, and class-balanced evaluation.

The project asks two research questions: (1) which model family provides the strongest class-balanced sentiment performance after duplicate-safe evaluation, and (2) how much harder is six-class emotion classification under the same preprocessing and split policy? Macro-F1 is the primary metric; accuracy, weighted-F1, per-class results, confusion matrices, runtime, and uncertainty are supporting evidence.

## Related Work

Prior Urdu sentiment research spans TF-IDF classifiers, lexicon and rule-based systems, CNN/LSTM architectures, multilingual encoders, Urdu-specific pretraining, and multimodal approaches. Three persistent gaps motivate this implementation: weak-label shortcuts are often not controlled, experiments frequently use incomparable preprocessing/splits, and accuracy obscures rare-class failure. This project addresses those gaps through emoji removal, connected duplicate grouping, shared task-specific splits, repeated seeds, validation-only selection, and full prediction artifacts. The bibliography includes the dataset paper, core statistical/deep-learning methods, and the Transformer models used in the experiments.

## Problem Definition and Objectives

The sentiment task maps Joy to Positive, Surprise to Neutral, and Sad/Angry/Fear/Disgust to Negative. The emotion task preserves Joy, Sad, Angry, Fear, Disgust, and Surprise. Objectives are to build separate reproducible pipelines, compare eight models, prevent ID/text duplicate overlap, quantify seed variation, retain all checkpoints and predictions, and deploy both tasks in one Streamlit demonstration.

## Proposed Methodology

### Preprocessing

The deterministic preprocessing pipeline normalizes Urdu/Arabic Unicode, removes URLs and mentions, preserves hashtag text, removes emojis, numbers, and punctuation, normalizes whitespace, and requires at least two cleaned tokens. Emoji removal controls direct target-signal leakage from the weak-label construction process.

### Group-Safe Splitting

Rows are connected when they share either a tweet ID or normalized text. Any connected group containing multiple task labels is excluded. Remaining exact normalized-text duplicates are reduced to one row. Complete groups are then assigned to deterministic stratified 70/15/15 train, validation, and test partitions. Automated checks confirm zero shared IDs and zero shared normalized texts across every split pair for both tasks.

### Models

The benchmark includes Logistic Regression, Linear SVM, Multinomial Naive Bayes, Text-CNN, BiLSTM with additive attention, mBERT, XLM-RoBERTa, and Urdu-RoBERTa. TF-IDF and neural vocabularies are fit on training data only. Neural models use inverse-frequency class weights, gradient clipping, validation macro-F1 checkpointing, and early stopping. Transformers use smoothed class weights and mixed precision.

### Selection and Uncertainty

Classical and neural models use three seeds. Transformer results are one-seed pilot experiments because of the available 24-36 hour compute window. The official ranking uses mean validation macro-F1; test metrics do not influence model choice. The selected canonical run is the seed with the highest validation macro-F1. Test uncertainty is reported with a 1,000-sample non-parametric bootstrap interval.

## Dataset and Experimental Setup

The raw CSV contains 1,048,000 rows, of which 514,571 lack the required Category label. After preprocessing, connected-group conflict removal, and deduplication, the sentiment benchmark retains **{sent_split['rows_after_filtering']:,}** rows and the emotion benchmark retains **{emo_split['rows_after_filtering']:,}** rows.

| Task | Train | Validation | Test | Conflict rows removed | Duplicate rows removed |
|---|---:|---:|---:|---:|---:|
| Sentiment | {sent_split['train_size']:,} | {sent_split['validation_size']:,} | {sent_split['test_size']:,} | {sent_split['group_safety']['rows_removed_in_conflicting_groups']:,} | {sent_split['group_safety']['rows_removed_as_duplicate']:,} |
| Emotion | {emo_split['train_size']:,} | {emo_split['validation_size']:,} | {emo_split['test_size']:,} | {emo_split['group_safety']['rows_removed_in_conflicting_groups']:,} | {emo_split['group_safety']['rows_removed_as_duplicate']:,} |

Training used an NVIDIA GeForce RTX 5070 Ti. The saved repository contains 36 official runs: 18 classical, 12 neural, and 6 Transformer runs. Every run has isolated models, metrics, predictions, and metadata.

## Results and Discussion

### Sentiment Results

{_leaderboard_table(sentiment['leaderboard'])}

Linear SVM is selected by mean validation macro-F1 **{sent_selected['validation_macro_f1_mean']:.4f}** and obtains test macro-F1 **{sent_selected['test_macro_f1_mean']:.4f}**. Its bootstrap 95% interval is **[{sent_selected['bootstrap_95']['lower_95']:.4f}, {sent_selected['bootstrap_95']['upper_95']:.4f}]**. Neural seed variation is material, particularly for Text-CNN, which supports reporting repeated runs rather than one favorable seed.

{_class_table(sentiment['selected_metrics'])}

### Emotion Results

{_leaderboard_table(emotion['leaderboard'])}

Linear SVM is also selected for emotion with mean validation macro-F1 **{emo_selected['validation_macro_f1_mean']:.4f}** and test macro-F1 **{emo_selected['test_macro_f1_mean']:.4f}**. Its bootstrap 95% interval is **[{emo_selected['bootstrap_95']['lower_95']:.4f}, {emo_selected['bootstrap_95']['upper_95']:.4f}]**. Emotion remains substantially harder because the minority classes contain far fewer examples and emoji removal eliminates some of the most direct weak-label cues.

{_class_table(emotion['selected_metrics'])}

### Interpretation

Sparse word n-grams remain strong on this corpus because repeated lexical patterns are informative even after duplicate groups are separated. The neural models show greater seed sensitivity and are trained with randomly initialized embeddings. Transformers are under-trained by design and should be interpreted as resource-constrained pilots rather than matched-budget evidence that pretraining is ineffective. High accuracy for majority-biased models confirms why macro-F1 must remain the headline metric.

## Error Analysis and Optimization

The selected models continue to struggle most on Neutral sentiment and the rare Fear/Surprise/Angry emotion classes. The pipeline preserves per-class reports, confusion matrices, and full prediction CSVs for inspection. Optimization includes class weighting, validation checkpointing, early stopping, gradient clipping, mixed precision, deterministic seeds, training-only feature fitting, and artifact isolation. Further improvements should prioritize human annotation, domain-adapted embeddings, longer matched-budget Transformer training, and calibrated decision probabilities.

## Ethical Considerations and Limitations

The source contains public social-media text and weak labels that may encode demographic, topical, and cultural bias. The models are unsuitable for surveillance, punitive moderation, diagnosis, or decisions about individuals. No human-gold evaluation is claimed. Confidence-like values from Linear SVM are explicitly labeled decision scores because they are normalized margins, not calibrated probabilities. The benchmark controls duplicate-instance leakage and direct emoji shortcuts, but weak-label noise and domain limitations remain.

## Conclusion

The project delivers separate, runnable sentiment and emotion pipelines with group-safe splits, eight model implementations, 36 isolated training runs, repeated-seed statistics, bootstrap uncertainty, complete saved artifacts, and dual-task inference. Linear SVM provides the strongest validation-ranked macro-F1 for both tasks under the official protocol. The main scientific conclusion is not that deep models are intrinsically weaker, but that a well-tuned sparse baseline remains difficult to beat under noisy weak supervision and a constrained training budget.

## References

{references}
"""

    summary = f"""# Final Evaluation Summary

## Protocol

- Tasks: Sentiment (3 classes) and Emotion (6 classes)
- Models: 8 per task
- Completed runs: 36
- Classical/neural seeds: 42, 52, 62
- Transformer seed: 42
- Selection metric: mean validation macro-F1
- Test uncertainty: 1,000-sample bootstrap interval
- Human-gold evaluation: unavailable and not claimed

## Sentiment

{_leaderboard_table(sentiment['leaderboard'])}

Selected model: `linear_svm`<br>
Validation macro-F1: {sent_selected['validation_macro_f1_mean']:.4f}<br>
Test macro-F1: {sent_selected['test_macro_f1_mean']:.4f}<br>
Bootstrap 95% interval: [{sent_selected['bootstrap_95']['lower_95']:.4f}, {sent_selected['bootstrap_95']['upper_95']:.4f}]

## Emotion

{_leaderboard_table(emotion['leaderboard'])}

Selected model: `linear_svm`<br>
Validation macro-F1: {emo_selected['validation_macro_f1_mean']:.4f}<br>
Test macro-F1: {emo_selected['test_macro_f1_mean']:.4f}<br>
Bootstrap 95% interval: [{emo_selected['bootstrap_95']['lower_95']:.4f}, {emo_selected['bootstrap_95']['upper_95']:.4f}]
"""

    dataset_card = f"""# Dataset Card: Group-Safe SentiUrdu-1M Benchmark

## Source

SentiUrdu-1M contains 1,048,000 noisy Urdu tweets. Category labels are weakly supervised; 514,571 rows do not contain the required Category label.

## Official Task Datasets

| Task | Retained rows | Train | Validation | Test | Classes |
|---|---:|---:|---:|---:|---:|
| Sentiment | {sent_split['rows_after_filtering']:,} | {sent_split['train_size']:,} | {sent_split['validation_size']:,} | {sent_split['test_size']:,} | 3 |
| Emotion | {emo_split['rows_after_filtering']:,} | {emo_split['train_size']:,} | {emo_split['validation_size']:,} | {emo_split['test_size']:,} | 6 |

## Data Quality Controls

- Emoji-derived shortcut cues are removed from text.
- Tweet IDs and normalized texts define connected duplicate groups.
- Conflicting-label groups are excluded.
- Exact normalized-text duplicates are reduced to one row.
- Split validators require zero ID and text overlap.

## Limitations

Labels remain weakly supervised and may be wrong. No human-gold evaluation is claimed. The data represent Pakistani Twitter discourse and should not be assumed to generalize to formal Urdu, private messages, or other regions.
"""

    model_card = f"""# Model Card: Urdu Sentiment and Emotion Benchmark

## Models

`linear_svm`, `logistic_regression`, `multinomial_nb`, `text_cnn`, `bilstm_attention`, `mbert`, `xlm_roberta`, and `urdu_roberta` are trained separately for Sentiment and Emotion.

## Selected Models

| Task | Model | Validation Macro-F1 | Test Macro-F1 | Test 95% interval |
|---|---|---:|---:|---:|
| Sentiment | `linear_svm` | {sent_selected['validation_macro_f1_mean']:.4f} | {sent_selected['test_macro_f1_mean']:.4f} | [{sent_selected['bootstrap_95']['lower_95']:.4f}, {sent_selected['bootstrap_95']['upper_95']:.4f}] |
| Emotion | `linear_svm` | {emo_selected['validation_macro_f1_mean']:.4f} | {emo_selected['test_macro_f1_mean']:.4f} | [{emo_selected['bootstrap_95']['lower_95']:.4f}, {emo_selected['bootstrap_95']['upper_95']:.4f}] |

## Intended Use

Course demonstration, reproducible Urdu NLP research, and aggregate error analysis. Outputs are estimates and must not be used for decisions about individuals.

## Score Semantics

Neural and Transformer outputs are softmax probabilities. Linear SVM outputs are normalized decision margins and are displayed as decision scores, not calibrated probabilities.

## Limitations

Weak labels, extreme imbalance, social-media domain restriction, absence of human-gold evaluation, randomly initialized neural embeddings, and one-seed resource-constrained Transformer experiments.
"""

    runs = _collect_runs(project_root)
    manifest = {
        "generated_on": "2026-06-15",
        "git_commit": _git_commit(project_root),
        "protocol": {
            "tasks": ["sentiment", "emotion"],
            "selection_metric": "validation_macro_f1_mean",
            "classical_neural_seeds": [42, 52, 62],
            "transformer_seeds": [42],
            "transformer_sample_size": 50000,
            "transformer_epochs": 1,
            "human_gold_evaluation": False,
            "bootstrap_samples": 1000,
        },
        "configs": {
            name: {
                "path": name,
                "sha256": _sha256(project_root / name),
            }
            for name in ("config_sentiment.yaml", "config_emotion.yaml")
        },
        "tasks": {
            task: {
                "split_summary": task_data[task]["split"],
                "selected_model": task_data[task]["aggregate"]["selected_model"],
            }
            for task in ("sentiment", "emotion")
        },
        "runs": runs,
    }

    return {
        "final_report.md": report,
        "final_evaluation_summary.md": summary,
        "dataset_card.md": dataset_card,
        "model_card.md": model_card,
        "experiment_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2),
    }


def generate_figures(project_root: Path) -> None:
    """Generate official aggregate charts and selected-model confusion matrices."""
    sns.set_theme(style="whitegrid")
    reports_figures = project_root / "reports" / "figures"
    reports_figures.mkdir(parents=True, exist_ok=True)
    for task in ("sentiment", "emotion"):
        results_dir = project_root / "outputs" / task / "results"
        task_figures = project_root / "outputs" / task / "figures"
        task_figures.mkdir(parents=True, exist_ok=True)
        leaderboard = pd.read_csv(results_dir / "model_comparison_leaderboard.csv")
        split = _load_json(results_dir / "split_summary.json")
        aggregate = _load_json(results_dir / "aggregate_metrics.json")

        ordered = leaderboard.sort_values("validation_macro_f1_mean")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        colors = ["#0f766e" if name == aggregate["selected_model"]["model_name"] else "#64748b" for name in ordered["model_name"]]
        ax.barh(
            [MODEL_DISPLAY.get(name, name) for name in ordered["model_name"]],
            ordered["validation_macro_f1_mean"],
            xerr=ordered["validation_macro_f1_std"],
            color=colors,
            capsize=3,
        )
        ax.set_xlabel("Mean validation Macro-F1")
        ax.set_title(f"{task.title()} Model Comparison (Validation-Ranked)")
        ax.set_xlim(left=0)
        fig.tight_layout()
        for target in (task_figures / "validation_macro_f1_comparison.png", reports_figures / f"{task}_validation_macro_f1.png"):
            fig.savefig(target, dpi=180, bbox_inches="tight")
        plt.close(fig)

        distribution = split["class_distribution_before_split"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(distribution.keys(), distribution.values(), color="#2563eb")
        ax.set_yscale("log")
        ax.set_ylabel("Rows (log scale)")
        ax.set_title(f"{task.title()} Class Distribution After Data Quality Controls")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        for target in (task_figures / "class_distribution.png", reports_figures / f"{task}_class_distribution.png"):
            fig.savefig(target, dpi=180, bbox_inches="tight")
        plt.close(fig)

        selected = aggregate["selected_model"]
        family = selected["model_family"]
        model = selected["model_name"]
        seed = selected["canonical_seed"]
        matrix_path = (
            project_root
            / "outputs"
            / task
            / "runs"
            / family
            / model
            / f"seed_{seed}"
            / "results"
            / f"confusion_matrix_{family}_{model}_test.csv"
        )
        matrix = pd.read_csv(matrix_path, index_col=0)
        normalized = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title(f"{task.title()} Selected Model: Row-Normalized Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        for target in (task_figures / "selected_model_confusion_matrix.png", reports_figures / f"{task}_confusion_matrix.png"):
            fig.savefig(target, dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_report_bundle(project_root: Path) -> dict[str, Path]:
    bundle = build_report_bundle(project_root)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, content in bundle.items():
        path = reports_dir / name
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        outputs[name] = path
    generate_figures(project_root)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    outputs = write_report_bundle(root)
    print("\n".join(str(path) for path in outputs.values()))


if __name__ == "__main__":
    main()
