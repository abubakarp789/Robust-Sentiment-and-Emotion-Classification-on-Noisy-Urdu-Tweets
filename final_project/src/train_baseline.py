"""Train reproducible TF-IDF statistical baselines on saved data splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from .evaluate import (
        build_leaderboard,
        compute_classification_metrics,
        save_classification_report,
        save_confusion_matrix,
    )
    from .utils import load_config, set_seed
except ImportError:  # Support direct execution: python src/train_baseline.py
    from evaluate import (
        build_leaderboard,
        compute_classification_metrics,
        save_classification_report,
        save_confusion_matrix,
    )
    from utils import load_config, set_seed


SPLITS = ("train", "validation", "test")
EVALUATION_SPLITS = ("validation", "test")
REQUIRED_DATA_COLUMNS = ("id", "raw_text", "clean_text", "task_label", "text_length")
REQUIRED_PREDICTION_COLUMNS = (
    "id",
    "raw_text",
    "clean_text",
    "true_label",
    "predicted_label",
    "confidence",
    "split",
    "model_name",
    "is_correct",
    "text_length",
)
MODEL_FILE_NAMES = {
    "logistic_regression": "baseline_logistic_regression.joblib",
    "linear_svm": "baseline_linear_svm.joblib",
    "multinomial_nb": "baseline_multinomial_nb.joblib",
}


def resolve_project_path(project_root: Path, configured_path: str | Path) -> Path:
    """Resolve a configured path relative to the final-project directory."""
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def load_splits(project_root: Path, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Load saved split CSVs without reading or reprocessing the raw dataset."""
    split_dir = resolve_project_path(project_root, config["data"]["output_dir"])
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        frame = pd.read_csv(path, encoding="utf-8", dtype={"id": "string"})
        missing = sorted(set(REQUIRED_DATA_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{path.name} is empty")
        if frame[[config["baseline_models"]["text_column"], config["baseline_models"]["label_column"]]].isna().any().any():
            raise ValueError(f"{path.name} contains missing baseline text or labels")
        frames[split] = frame
    return frames


def create_vectorizer(config: Mapping[str, Any]) -> TfidfVectorizer:
    """Build the configured word-level TF-IDF vectorizer."""
    tfidf_config = dict(config["baseline_models"]["tfidf"])
    tfidf_config["ngram_range"] = tuple(tfidf_config["ngram_range"])
    dtype_name = tfidf_config.pop("dtype", "float32")
    tfidf_config["dtype"] = np.dtype(dtype_name).type
    return TfidfVectorizer(**tfidf_config)


def create_classifiers(config: Mapping[str, Any], seed: int) -> dict[str, ClassifierMixin]:
    """Build enabled statistical classifiers from configuration."""
    model_config = config["baseline_models"]["models"]
    classifiers: dict[str, ClassifierMixin] = {}

    lr_config = dict(model_config["logistic_regression"])
    if lr_config.pop("enabled", False):
        lr_config["random_state"] = seed
        logistic_regression = LogisticRegression(**lr_config)
        classifiers["logistic_regression"] = (
            OneVsRestClassifier(logistic_regression)
            if lr_config.get("solver") == "liblinear"
            else logistic_regression
        )

    svm_config = dict(model_config["linear_svm"])
    if svm_config.pop("enabled", False):
        svm_config["random_state"] = seed
        classifiers["linear_svm"] = LinearSVC(**svm_config)

    nb_config = dict(model_config["multinomial_nb"])
    if nb_config.pop("enabled", False):
        classifiers["multinomial_nb"] = MultinomialNB(**nb_config)

    if not classifiers:
        raise ValueError("At least one baseline model must be enabled")
    return classifiers


def mark_training_only_fit(pipeline: Pipeline, train_rows: int, text_column: str) -> None:
    """Attach auditable evidence that feature fitting used the train split only."""
    pipeline.fit_audit_ = {
        "tfidf_fit_split": "train",
        "text_column": text_column,
        "train_rows": int(train_rows),
        "validation_rows_used_for_fit": 0,
        "test_rows_used_for_fit": 0,
    }


def confidence_scores(pipeline: Pipeline, texts: Iterable[str]) -> np.ndarray:
    """Return predicted-class confidence from probabilities or normalized margins."""
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(texts)
        return np.asarray(probabilities).max(axis=1)

    decisions = np.asarray(pipeline.decision_function(texts))
    if decisions.ndim == 1:
        return expit(np.abs(decisions))
    return softmax(decisions, axis=1).max(axis=1)


def build_prediction_frame(
    source: pd.DataFrame,
    predicted_labels: Sequence[str],
    confidence: Sequence[float],
    split: str,
    model_name: str,
) -> pd.DataFrame:
    """Build the standard UTF-8 prediction artifact."""
    frame = pd.DataFrame(
        {
            "id": source["id"].astype("string"),
            "raw_text": source["raw_text"],
            "clean_text": source["clean_text"],
            "true_label": source["task_label"],
            "predicted_label": predicted_labels,
            "confidence": confidence,
            "split": split,
            "model_name": model_name,
            "is_correct": source["task_label"].to_numpy() == np.asarray(predicted_labels),
            "text_length": source["text_length"],
        }
    )
    return frame.loc[:, REQUIRED_PREDICTION_COLUMNS]


def train_baselines(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Train enabled baselines and save models, metrics, and predictions."""
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    seed = int(config["project"]["random_seed"])
    set_seed(seed)

    frames = load_splits(project_root, config)
    baseline_config = config["baseline_models"]
    text_column = baseline_config["text_column"]
    label_column = baseline_config["label_column"]
    labels = sorted(frames["train"][label_column].unique().tolist())
    unknown_eval_labels = (
        set(frames["validation"][label_column]) | set(frames["test"][label_column])
    ) - set(labels)
    if unknown_eval_labels:
        raise ValueError(f"Evaluation splits contain labels absent from train: {unknown_eval_labels}")

    output_config = config["outputs"]
    models_dir = resolve_project_path(project_root, output_config["models_dir"])
    results_dir = resolve_project_path(project_root, output_config["results_dir"])
    predictions_dir = resolve_project_path(project_root, output_config["predictions_dir"])
    for directory in (models_dir, results_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    vectorizer = create_vectorizer(config)
    print(f"Fitting TF-IDF on {len(frames['train']):,} training rows only...")
    x_train = vectorizer.fit_transform(frames["train"][text_column])
    transformed = {
        split: vectorizer.transform(frames[split][text_column]) for split in EVALUATION_SPLITS
    }
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")

    metrics: dict[str, Any] = {
        "metadata": {
            "random_seed": seed,
            "text_column": text_column,
            "label_column": label_column,
            "labels": labels,
            "split_sizes": {name: int(len(frame)) for name, frame in frames.items()},
            "tfidf_fit_split": "train",
            "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "models": {},
    }
    training_metadata: dict[str, Any] = {
        "tfidf_fit_split": "train",
        "train_rows": int(len(frames["train"])),
        "validation_rows_used_for_fit": 0,
        "test_rows_used_for_fit": 0,
        "models": {},
    }

    for model_name, classifier in create_classifiers(config, seed).items():
        model_started = time.perf_counter()
        print(f"Training {model_name}...")
        classifier.fit(x_train, frames["train"][label_column])
        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
        mark_training_only_fit(pipeline, len(frames["train"]), text_column)
        model_path = models_dir / MODEL_FILE_NAMES[model_name]
        joblib.dump(pipeline, model_path, compress=3)

        metrics["models"][model_name] = {}
        for split in EVALUATION_SPLITS:
            frame = frames[split]
            predictions = classifier.predict(transformed[split])
            if hasattr(classifier, "predict_proba"):
                confidence = np.asarray(classifier.predict_proba(transformed[split])).max(axis=1)
            else:
                decisions = np.asarray(classifier.decision_function(transformed[split]))
                confidence = (
                    expit(np.abs(decisions))
                    if decisions.ndim == 1
                    else softmax(decisions, axis=1).max(axis=1)
                )
            split_metrics = compute_classification_metrics(
                frame[label_column], predictions, labels=labels
            )
            metrics["models"][model_name][split] = split_metrics

            prediction_frame = build_prediction_frame(
                frame, predictions, confidence, split, model_name
            )
            prediction_frame.to_csv(
                predictions_dir / f"baseline_{model_name}_{split}_predictions.csv",
                index=False,
                encoding="utf-8",
            )
            save_classification_report(
                frame[label_column],
                predictions,
                results_dir / f"classification_report_baseline_{model_name}_{split}.json",
            )
            save_confusion_matrix(
                frame[label_column],
                predictions,
                labels,
                results_dir / f"confusion_matrix_baseline_{model_name}_{split}.csv",
            )
            print(
                f"  {split}: accuracy={split_metrics['accuracy']:.4f}, "
                f"macro-F1={split_metrics['macro_f1']:.4f}, "
                f"weighted-F1={split_metrics['weighted_f1']:.4f}"
            )

        training_metadata["models"][model_name] = {
            "model_path": str(model_path.relative_to(project_root)),
            "fit_seconds": round(time.perf_counter() - model_started, 3),
        }

    metrics["metadata"]["total_seconds"] = round(time.perf_counter() - started, 3)
    metrics_path = results_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "baseline_training_metadata.json").write_text(
        json.dumps(training_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_leaderboard(metrics, results_dir / "baseline_leaderboard.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_baselines(args.config)


if __name__ == "__main__":
    main()
