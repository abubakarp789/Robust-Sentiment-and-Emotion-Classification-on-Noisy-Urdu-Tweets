"""HuggingFace fine-tuning utilities for mBERT, XLM-R, and Urdu-RoBERTa.

Wraps `Trainer` so the notebook can fine-tune all three transformers with a
single uniform function call per task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass
class TransformerCfg:
    name: str
    model_name: str
    lr: float
    epochs: int
    batch_size: int
    grad_accum: int = 1
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_len: int = 96
    fp16: bool = True


def _build_dataset(df: pd.DataFrame, text_col: str, label_col: str) -> Dataset:
    return Dataset.from_pandas(
        df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "labels"}),
        preserve_index=False,
    )


def _make_metric_fn():
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float((preds == labels).mean()),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        }
    return _compute


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


def finetune_transformer(
    cfg: TransformerCfg,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    text_col: str,
    label_col: str,
    id2label: dict,
    label2id: dict,
    class_weights: Sequence[float],
    output_root: Path,
    task_name: str,
) -> dict:
    """Fine-tune a single transformer and return test predictions + metrics."""
    out_dir = Path(output_root) / f"{cfg.name}_{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    def _tok(batch):
        return tokenizer(
            batch["text"], padding=False, truncation=True, max_length=cfg.max_len
        )

    train_ds = _build_dataset(train_df, text_col, label_col).map(_tok, batched=True)
    val_ds   = _build_dataset(val_df,   text_col, label_col).map(_tok, batched=True)
    test_ds  = _build_dataset(test_df,  text_col, label_col).map(_tok, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    metric_fn = _make_metric_fn()

    args = TrainingArguments(
        output_dir=str(out_dir / "hf"),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=200,
        report_to=["none"],
        fp16=cfg.fp16 and torch.cuda.is_available(),
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metric_fn,
        class_weights=torch.tensor(class_weights, dtype=torch.float32),
    )

    trainer.train()
    val_metrics = trainer.evaluate(val_ds)
    test_pred = trainer.predict(test_ds)

    preds = np.argmax(test_pred.predictions, axis=-1)
    metrics = {
        "task": task_name,
        "model": cfg.name,
        "val": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "test": {k: float(v) for k, v in (test_pred.metrics or {}).items()
                 if isinstance(v, (int, float))},
    }
    # Persist tokenizer + best model under a stable directory for inference reuse.
    save_root = out_dir / "best"
    trainer.save_model(str(save_root))
    tokenizer.save_pretrained(str(save_root))

    pd.DataFrame({"y_true": test_df[label_col].values, "y_pred": preds}).to_csv(
        out_dir / "test_predictions.csv", index=False
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {"preds": preds, "metrics": metrics, "output_dir": str(out_dir)}


__all__ = ["TransformerCfg", "finetune_transformer", "WeightedTrainer"]
