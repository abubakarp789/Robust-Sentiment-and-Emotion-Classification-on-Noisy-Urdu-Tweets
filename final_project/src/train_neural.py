"""Train Text-CNN and BiLSTM-attention models on saved split CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from .evaluate import (
        build_leaderboard,
        compute_classification_metrics,
        save_classification_report,
        save_confusion_matrix,
    )
    from .models_dl import BiLSTMAttention, TextCNN
    from .neural_utils import (
        PAD_TOKEN,
        build_vocab,
        compute_class_weights,
        encode_labels,
        encode_text,
        get_device,
        load_checkpoint,
        load_pretrained_embeddings,
        save_checkpoint,
        save_label_mapping,
        save_vocab,
        set_seed,
    )
    from .train_baseline import (
        REQUIRED_DATA_COLUMNS,
        build_prediction_frame,
        resolve_project_path,
    )
    from .utils import load_config
except ImportError:
    from evaluate import (
        build_leaderboard,
        compute_classification_metrics,
        save_classification_report,
        save_confusion_matrix,
    )
    from models_dl import BiLSTMAttention, TextCNN
    from neural_utils import (
        PAD_TOKEN,
        build_vocab,
        compute_class_weights,
        encode_labels,
        encode_text,
        get_device,
        load_checkpoint,
        load_pretrained_embeddings,
        save_checkpoint,
        save_label_mapping,
        save_vocab,
        set_seed,
    )
    from train_baseline import REQUIRED_DATA_COLUMNS, build_prediction_frame, resolve_project_path
    from utils import load_config


MODEL_CLASSES = {"text_cnn": TextCNN, "bilstm_attention": BiLSTMAttention}
MODEL_FILES = {
    "text_cnn": "neural_text_cnn.pt",
    "bilstm_attention": "neural_bilstm_attention.pt",
}
EVALUATION_SPLITS = ("validation", "test")


class EncodedTextDataset(Dataset):
    """In-memory int32 sequences for efficient repeated neural epochs."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=True))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def load_neural_splits(
    project_root: Path, config: Mapping[str, Any], sample_size: int | None = None
) -> dict[str, pd.DataFrame]:
    """Load only saved splits and optionally stratify-sample the training split."""
    split_dir = resolve_project_path(project_root, config["data"]["output_dir"])
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, encoding="utf-8", dtype={"id": "string"})
        missing = sorted(set(REQUIRED_DATA_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{path.name} is empty")
        frames[split] = frame

    if sample_size is not None and sample_size < len(frames["train"]):
        _, sample = train_test_split(
            frames["train"],
            test_size=sample_size,
            stratify=frames["train"][config["neural_models"]["label_column"]],
            random_state=int(config["neural_models"]["training"]["random_seed"]),
        )
        frames["train"] = sample.sort_index().reset_index(drop=True)
    return frames


def encode_frame(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    vocab: Mapping[str, int],
    label_to_id: Mapping[str, int],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one dataframe into fixed-length sequences and integer labels."""
    sequences = np.empty((len(frame), max_length), dtype=np.int32)
    for index, text in enumerate(frame[text_column].astype(str)):
        sequences[index] = encode_text(text, vocab, max_length)
    labels = frame[label_column].map(label_to_id)
    if labels.isna().any():
        unknown = sorted(frame.loc[labels.isna(), label_column].unique())
        raise ValueError(f"Unknown labels outside training mapping: {unknown}")
    return sequences, labels.to_numpy(dtype=np.int64)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_model(
    model_name: str,
    config: Mapping[str, Any],
    vocab_size: int,
    num_classes: int,
    padding_idx: int,
    pretrained_embeddings: torch.Tensor | None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct one configured neural model and serializable kwargs."""
    neural = config["neural_models"]
    embeddings = neural["embeddings"]
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": int(embeddings["embedding_dim"]),
        "num_classes": num_classes,
        "padding_idx": padding_idx,
        "pretrained_embeddings": pretrained_embeddings,
        "freeze_embeddings": bool(embeddings["freeze_embeddings"]),
    }
    model_config = dict(neural["models"][model_name])
    model_config.pop("enabled", None)
    if model_name == "text_cnn":
        model_config["kernel_sizes"] = list(model_config["kernel_sizes"])
    kwargs = {**common, **model_config}
    model = MODEL_CLASSES[model_name](**kwargs)
    serializable_kwargs = {key: value for key, value in kwargs.items() if key != "pretrained_embeddings"}
    return model, serializable_kwargs


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_confidence: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    criterion = nn.CrossEntropyLoss()
    for sequences, labels in loader:
        sequences = sequences.to(device, non_blocking=True).long()
        labels = labels.to(device, non_blocking=True)
        mask = sequences.ne(0)
        logits = model(sequences, mask)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        total_rows += labels.size(0)
        all_true.append(labels.cpu().numpy())
        all_pred.append(predictions.cpu().numpy())
        all_confidence.append(probabilities.max(dim=1).values.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    confidence = np.concatenate(all_confidence)
    from sklearn.metrics import f1_score

    macro_f1 = f1_score(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    return total_loss / max(total_rows, 1), float(macro_f1), y_true, y_pred, confidence


def fit_model(
    model: nn.Module,
    model_name: str,
    model_kwargs: dict[str, Any],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    training_config: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Train with macro-F1 early stopping and save the best checkpoint."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    clip_norm = float(training_config["gradient_clip_norm"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_macro_f1 = -1.0
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0
        for sequences, labels in train_loader:
            sequences = sequences.to(device, non_blocking=True).long()
            labels = labels.to(device, non_blocking=True)
            mask = sequences.ne(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(sequences, mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_loss, val_macro_f1, y_true, y_pred, _ = evaluate_model(
            model, validation_loader, device, len(class_weights)
        )
        val_accuracy = float((y_true == y_pred).mean())
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"  epoch {epoch}: train_loss={row['train_loss']:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, "
            f"val_macro_f1={val_macro_f1:.4f}, seconds={row['epoch_seconds']:.1f}"
        )
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            bad_epochs = 0
            save_checkpoint(
                model,
                checkpoint_path,
                {
                    **checkpoint_metadata,
                    "model_name": model_name,
                    "model_kwargs": model_kwargs,
                    "best_epoch": epoch,
                    "best_validation_macro_f1": val_macro_f1,
                },
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping after {bad_epochs} non-improving epoch(s)")
                break
    return history


def train_neural_models(
    config_path: str | Path = "config.yaml",
    sample_size: int | None = None,
    epochs_override: int | None = None,
    selected_model: str | None = None,
) -> dict[str, Any]:
    """Train configured neural models and save complete evaluation artifacts."""
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    neural = config["neural_models"]
    if not neural.get("enabled", True):
        raise ValueError("neural_models.enabled is false")
    training_config = dict(neural["training"])
    if epochs_override is not None:
        training_config["epochs"] = epochs_override
    seed = int(training_config["random_seed"])
    set_seed(seed)
    device = get_device(str(training_config["device"]))
    print(f"Using device: {device}")

    frames = load_neural_splits(project_root, config, sample_size)
    text_column = neural["text_column"]
    label_column = neural["label_column"]
    data_config = neural["data"]
    max_length = int(data_config["max_sequence_length"])
    vocab = build_vocab(
        frames["train"][text_column].astype(str),
        int(data_config["max_vocab_size"]),
        int(data_config["min_frequency"]),
    )
    label_to_id, id_to_label = encode_labels(frames["train"][label_column])
    labels = [id_to_label[index] for index in range(len(id_to_label))]
    encoded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, frame in frames.items():
        print(f"Encoding {split}: {len(frame):,} rows")
        encoded[split] = encode_frame(
            frame, text_column, label_column, vocab, label_to_id, max_length
        )

    output_config = config["outputs"]
    models_dir = resolve_project_path(project_root, output_config["models_dir"])
    results_dir = resolve_project_path(project_root, output_config["results_dir"])
    predictions_dir = resolve_project_path(project_root, output_config["predictions_dir"])
    for directory in (models_dir, results_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)
    vocab_path = models_dir / "neural_vocab.json"
    label_mapping_path = models_dir / "neural_label_mapping.json"
    save_vocab(vocab, vocab_path)
    save_label_mapping(label_to_id, id_to_label, label_mapping_path)

    pretrained_embeddings = None
    matched_embeddings = 0
    embedding_config = neural["embeddings"]
    if embedding_config.get("use_pretrained", False):
        pretrained_path = resolve_project_path(project_root, embedding_config["pretrained_path"])
        pretrained_embeddings, matched_embeddings = load_pretrained_embeddings(
            pretrained_path, vocab, int(embedding_config["embedding_dim"]), seed
        )

    datasets = {
        split: EncodedTextDataset(sequences, targets)
        for split, (sequences, targets) in encoded.items()
    }
    batch_size = int(data_config["batch_size"])
    workers = int(data_config["num_workers"])
    pin_memory = device.type == "cuda"
    loaders = {
        split: create_dataloader(
            dataset,
            batch_size,
            split == "train",
            workers,
            seed,
            pin_memory,
        )
        for split, dataset in datasets.items()
    }
    class_weights = (
        compute_class_weights(encoded["train"][1], len(labels))
        if training_config.get("use_class_weights", True)
        else torch.ones(len(labels), dtype=torch.float32)
    )
    enabled_models = [
        name
        for name, model_config in neural["models"].items()
        if model_config.get("enabled", False)
    ]
    if selected_model:
        if selected_model not in enabled_models:
            raise ValueError(f"Model is not enabled or unknown: {selected_model}")
        enabled_models = [selected_model]

    run_started = time.perf_counter()
    metrics: dict[str, Any] = {
        "metadata": {
            "random_seed": seed,
            "device": str(device),
            "sample_size": sample_size,
            "split_sizes": {split: int(len(frame)) for split, frame in frames.items()},
            "vocabulary_size": len(vocab),
            "max_sequence_length": max_length,
            "class_weights": class_weights.tolist(),
            "labels": labels,
        },
        "models": {},
    }
    training_metadata: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "vocabulary_fit_split": "train",
        "validation_rows_used_for_vocabulary": 0,
        "test_rows_used_for_vocabulary": 0,
        "pretrained_embeddings_used": bool(embedding_config.get("use_pretrained", False)),
        "pretrained_tokens_matched": matched_embeddings,
        "models": {},
    }
    vocab_hash = hashlib.sha256(vocab_path.read_bytes()).hexdigest()

    for model_name in enabled_models:
        print(f"Training {model_name}...")
        model, model_kwargs = build_model(
            model_name,
            config,
            len(vocab),
            len(labels),
            vocab[PAD_TOKEN],
            pretrained_embeddings,
        )
        checkpoint_path = models_dir / MODEL_FILES[model_name]
        history = fit_model(
            model,
            model_name,
            model_kwargs,
            loaders["train"],
            loaders["validation"],
            class_weights,
            device,
            training_config,
            checkpoint_path,
            {
                "labels": labels,
                "label_to_id": label_to_id,
                "vocab_path": str(vocab_path.relative_to(project_root)),
                "vocab_sha256": vocab_hash,
                "vocabulary_fit_split": "train",
                "sample_size": sample_size,
            },
        )
        history_path = results_dir / f"neural_{model_name}_training_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        best_model, checkpoint_metadata = load_checkpoint(
            checkpoint_path, MODEL_CLASSES[model_name], model_kwargs, device
        )
        metrics["models"][model_name] = {}
        for split in EVALUATION_SPLITS:
            _, _, y_true_ids, y_pred_ids, confidence = evaluate_model(
                best_model, loaders[split], device, len(labels)
            )
            y_true = np.asarray([id_to_label[int(value)] for value in y_true_ids])
            y_pred = np.asarray([id_to_label[int(value)] for value in y_pred_ids])
            split_metrics = compute_classification_metrics(y_true, y_pred, labels=labels)
            metrics["models"][model_name][split] = split_metrics
            predictions = build_prediction_frame(
                frames[split], y_pred, confidence, split, model_name
            )
            predictions.to_csv(
                predictions_dir / f"neural_{model_name}_{split}_predictions.csv",
                index=False,
                encoding="utf-8",
            )
            save_classification_report(
                y_true,
                y_pred,
                results_dir / f"classification_report_neural_{model_name}_{split}.json",
            )
            save_confusion_matrix(
                y_true,
                y_pred,
                labels,
                results_dir / f"confusion_matrix_neural_{model_name}_{split}.csv",
            )
            print(
                f"  {split}: accuracy={split_metrics['accuracy']:.4f}, "
                f"macro-F1={split_metrics['macro_f1']:.4f}, "
                f"Neutral-F1={split_metrics['per_class'].get('Neutral', {}).get('f1', 0):.4f}"
            )
        training_metadata["models"][model_name] = {
            "checkpoint": str(checkpoint_path.relative_to(project_root)),
            "history": str(history_path.relative_to(project_root)),
            "best_epoch": checkpoint_metadata["best_epoch"],
            "best_validation_macro_f1": checkpoint_metadata["best_validation_macro_f1"],
            "epochs_completed": len(history),
            "model_kwargs": model_kwargs,
        }
        del best_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics["metadata"]["total_seconds"] = round(time.perf_counter() - run_started, 3)
    (results_dir / "neural_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "neural_training_metadata.json").write_text(
        json.dumps(training_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_leaderboard(metrics, results_dir / "neural_leaderboard.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model", choices=sorted(MODEL_CLASSES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_neural_models(args.config, args.sample_size, args.epochs, args.model)


if __name__ == "__main__":
    main()
