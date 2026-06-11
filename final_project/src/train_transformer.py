"""Train and evaluate multilingual transformer models on saved data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

try:
    from .evaluate import (
        build_leaderboard,
        compute_classification_metrics,
        save_classification_report,
        save_confusion_matrix,
    )
    from .neural_utils import set_seed, get_device
    from .train_baseline import (
        REQUIRED_DATA_COLUMNS,
        build_prediction_frame,
        resolve_project_path,
    )
    from .utils import load_config
    from .explanation_assistant import export_explanation_samples
except ImportError:
    from evaluate import (
        build_leaderboard,
        compute_classification_metrics,
        save_classification_report,
        save_confusion_matrix,
    )
    from neural_utils import set_seed, get_device
    from train_baseline import (
        REQUIRED_DATA_COLUMNS,
        build_prediction_frame,
        resolve_project_path,
    )
    from utils import load_config
    from explanation_assistant import export_explanation_samples


EVALUATION_SPLITS = ("validation", "test")


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_splits_transformer(
    project_root: Path, config: Mapping[str, Any], sample_size: int | None = None
) -> dict[str, pd.DataFrame]:
    """Load only saved splits and optionally sample the training split."""
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
        # Stratify sample train split to keep class distribution
        _, sample = train_test_split(
            frames["train"],
            test_size=sample_size,
            stratify=frames["train"][config["transformer_models"]["label_column"]],
            random_state=int(config["transformer_models"]["training"]["random_seed"]),
        )
        frames["train"] = sample.sort_index().reset_index(drop=True)
    return frames


def compute_class_weights_with_smoothing(
    labels: Sequence[int], num_classes: int, smoothing: float = 0.0
) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights with optional smoothing."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        counts = np.maximum(counts, 1.0)
    total_samples = len(labels)
    if smoothing > 0.0:
        uniform_count = total_samples / num_classes
        counts = (1.0 - smoothing) * counts + smoothing * uniform_count
    weights = total_samples / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _build_hf_dataset(df: pd.DataFrame, text_col: str, label_col: str, label_to_id: dict) -> Dataset:
    """Build a Hugging Face Dataset from a pandas DataFrame."""
    renamed = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "labels"})
    renamed["labels"] = renamed["labels"].map(label_to_id)
    return Dataset.from_pandas(renamed, preserve_index=False)


def _make_metric_fn(labels_list: list[str]):
    """Create metric function for Trainer evaluation."""
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true = [labels_list[int(idx)] for idx in labels]
        y_pred = [labels_list[int(idx)] for idx in preds]
        metrics = compute_classification_metrics(y_true, y_pred, labels=labels_list)
        return {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["macro_f1"],
            "f1_weighted": metrics["weighted_f1"],
        }
    return _compute


def train_transformer_models(
    config_path: str | Path = "config.yaml",
    sample_size: int | None = None,
    epochs_override: int | None = None,
    selected_model: str | None = None,
) -> dict[str, Any]:
    """Train enabled transformer models and save evaluation outputs."""
    config_file = Path(config_path).resolve()
    project_root = config_file.parent
    config = load_config(config_file)
    
    t_config = config["transformer_models"]
    if not t_config.get("enabled", True):
        raise ValueError("transformer_models.enabled is false")
        
    training_cfg = dict(t_config["training"])
    if epochs_override is not None:
        training_cfg["epochs"] = epochs_override
        
    seed = int(training_cfg["random_seed"])
    set_seed(seed)
    
    device = get_device(str(training_cfg["device"]))
    print(f"Using device: {device}")
    
    # Load Splits
    frames = load_splits_transformer(project_root, config, sample_size)
    
    text_column = t_config["text_column"]
    label_column = t_config["label_column"]
    
    # Build label mapping
    unique_labels = sorted(frames["train"][label_column].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(unique_labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    
    # Setup Output directories
    output_cfg = config["outputs"]
    models_dir = resolve_project_path(project_root, output_cfg["models_dir"])
    results_dir = resolve_project_path(project_root, output_cfg["results_dir"])
    predictions_dir = resolve_project_path(project_root, output_cfg["predictions_dir"])
    error_analysis_dir = resolve_project_path(project_root, output_cfg["error_analysis_dir"])
    
    for directory in (models_dir, results_dir, predictions_dir, error_analysis_dir):
        directory.mkdir(parents=True, exist_ok=True)
        
    # Save label mapping
    label_mapping_path = models_dir / "transformer_label_mapping.json"
    payload = {
        "label_to_id": label_to_id,
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
    }
    label_mapping_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Compute Class Weights
    train_int_labels = frames["train"][label_column].map(label_to_id).tolist()
    smoothing = float(training_cfg.get("class_weight_smoothing", 0.0))
    if training_cfg.get("use_class_weights", True):
        class_weights = compute_class_weights_with_smoothing(
            train_int_labels, len(unique_labels), smoothing
        )
    else:
        class_weights = torch.ones(len(unique_labels), dtype=torch.float32)
        
    enabled_models = [
        name
        for name, m_cfg in t_config["models"].items()
        if m_cfg.get("enabled", False)
    ]
    if selected_model:
        if selected_model not in enabled_models:
            raise ValueError(f"Model is not enabled or unknown: {selected_model}")
        enabled_models = [selected_model]
        
    run_started = time.perf_counter()
    
    metrics_path = results_dir / "transformer_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            # update metadata just in case
            metrics["metadata"].update({
                "random_seed": seed,
                "device": str(device),
                "sample_size": sample_size,
                "split_sizes": {split: int(len(frame)) for split, frame in frames.items()},
                "class_weights": class_weights.tolist(),
                "labels": unique_labels,
            })
        except Exception:
            metrics = {
                "metadata": {
                    "random_seed": seed,
                    "device": str(device),
                    "sample_size": sample_size,
                    "split_sizes": {split: int(len(frame)) for split, frame in frames.items()},
                    "class_weights": class_weights.tolist(),
                    "labels": unique_labels,
                },
                "models": {},
            }
    else:
        metrics = {
            "metadata": {
                "random_seed": seed,
                "device": str(device),
                "sample_size": sample_size,
                "split_sizes": {split: int(len(frame)) for split, frame in frames.items()},
                "class_weights": class_weights.tolist(),
                "labels": unique_labels,
            },
            "models": {},
        }
    
    metadata_path = results_dir / "transformer_training_metadata.json"
    if metadata_path.exists():
        try:
            training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            training_metadata.update({
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            })
            if "models" not in training_metadata:
                training_metadata["models"] = {}
        except Exception:
            training_metadata = {
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "models": {},
            }
    else:
        training_metadata = {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "models": {},
        }
    
    for model_key in enabled_models:
        model_name = t_config["models"][model_key]["model_name"]
        print(f"\nTraining {model_key} ({model_name})...")
        
        # Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(unique_labels),
            id2label=id_to_label,
            label2id=label_to_id,
        )
        
        # Tokenize datasets
        max_seq_len = int(t_config["data"]["max_sequence_length"])
        def _tok(batch):
            return tokenizer(
                batch["text"], padding=False, truncation=True, max_length=max_seq_len
            )
            
        train_ds = _build_hf_dataset(frames["train"], text_column, label_column, label_to_id).map(_tok, batched=True)
        val_ds = _build_hf_dataset(frames["validation"], text_column, label_column, label_to_id).map(_tok, batched=True)
        test_ds = _build_hf_dataset(frames["test"], text_column, label_column, label_to_id).map(_tok, batched=True)
        
        collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
        metric_fn = _make_metric_fn(unique_labels)
        
        model_out_dir = models_dir / f"transformer_{model_key}"
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        training_kwargs = dict(
            output_dir=str(model_out_dir / "hf_checkpoints"),
            num_train_epochs=int(training_cfg["epochs"]),
            learning_rate=float(training_cfg["learning_rate"]),
            per_device_train_batch_size=int(t_config["data"]["batch_size"]),
            per_device_eval_batch_size=int(t_config["data"]["batch_size"]) * 2,
            gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
            warmup_ratio=float(training_cfg["warmup_ratio"]),
            weight_decay=float(training_cfg["weight_decay"]),
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=50,
            report_to=["none"],
            fp16=bool(training_cfg["fp16"]) and device.type == "cuda",
            dataloader_pin_memory=(device.type == "cuda"),
            dataloader_num_workers=int(t_config["data"]["num_workers"]),
            seed=seed,
        )
        
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            training_kwargs["eval_strategy"] = "epoch"
        else:
            training_kwargs["evaluation_strategy"] = "epoch"
            
        args = TrainingArguments(**training_kwargs)
        
        early_stopping = EarlyStoppingCallback(early_stopping_patience=int(training_cfg["early_stopping_patience"]))
        
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=metric_fn,
            class_weights=class_weights,
            callbacks=[early_stopping],
        )
        
        model_started = time.perf_counter()
        trainer.train()
        fit_time = time.perf_counter() - model_started
        
        # Save best model to stable path
        best_dir = model_out_dir / "best"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        
        metrics["models"][model_key] = {}
        history = trainer.state.log_history
        # Save training history as csv
        history_df = pd.DataFrame([h for h in history if "eval_loss" in h or "loss" in h])
        history_df.to_csv(results_dir / f"transformer_{model_key}_training_history.csv", index=False)
        
        # Evaluate validation & test splits using best checkpoint
        for split in EVALUATION_SPLITS:
            ds = val_ds if split == "validation" else test_ds
            prediction_output = trainer.predict(ds)
            
            logits = prediction_output.predictions
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            y_pred_ids = np.argmax(probs, axis=-1)
            confidence = probs.max(axis=-1)
            
            y_true = [unique_labels[int(idx)] for idx in prediction_output.label_ids]
            y_pred = [unique_labels[int(idx)] for idx in y_pred_ids]
            
            split_metrics = compute_classification_metrics(y_true, y_pred, labels=unique_labels)
            metrics["models"][model_key][split] = split_metrics
            
            # Save predictions
            prediction_frame = build_prediction_frame(
                frames[split], y_pred, confidence, split, model_key
            )
            prediction_frame.to_csv(
                predictions_dir / f"transformer_{model_key}_{split}_predictions.csv",
                index=False,
                encoding="utf-8",
            )
            
            # Save classification report and confusion matrix
            save_classification_report(
                y_true,
                y_pred,
                results_dir / f"classification_report_transformer_{model_key}_{split}.json",
            )
            save_confusion_matrix(
                y_true,
                y_pred,
                unique_labels,
                results_dir / f"confusion_matrix_transformer_{model_key}_{split}.csv",
            )
            
            print(
                f"  {split}: accuracy={split_metrics['accuracy']:.4f}, "
                f"macro-F1={split_metrics['macro_f1']:.4f}, "
                f"Neutral-F1={split_metrics['per_class'].get('Neutral', {}).get('f1', 0):.4f}"
            )
            
            # Export explanation samples for test split
            if split == "test":
                export_explanation_samples(
                    prediction_frame,
                    error_analysis_dir / f"explanation_samples_{model_key}.json",
                    model_key,
                    max_samples=5
                )
                # If this is the main enabled model or mbert, also save to explanation_samples.json as required
                export_explanation_samples(
                    prediction_frame,
                    error_analysis_dir / "explanation_samples.json",
                    model_key,
                    max_samples=5
                )
                
        training_metadata["models"][model_key] = {
            "checkpoint": str(best_dir.relative_to(project_root)),
            "fit_seconds": round(fit_time, 3),
            "model_name": model_name,
        }
        
        # Clean memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics["metadata"]["total_seconds"] = round(time.perf_counter() - run_started, 3)
    (results_dir / "transformer_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "transformer_training_metadata.json").write_text(
        json.dumps(training_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    
    build_leaderboard(metrics, results_dir / "transformer_leaderboard.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model", choices=["mbert", "xlm_roberta", "urdu_roberta"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_transformer_models(args.config, args.sample_size, args.epochs, args.model)


if __name__ == "__main__":
    main()
