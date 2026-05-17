"""Training utilities for the CNN and BiLSTM models.

Designed for reusability between the sentiment and emotion tasks. Includes:
- a token-id vocabulary builder
- a Dataset / collate function for variable-length tweets
- a fastText embedding matrix loader
- a generic train/eval loop with class-weighted cross-entropy + early stopping
"""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from preprocessing import tokenize


PAD = "<pad>"
UNK = "<unk>"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
@dataclass
class Vocab:
    stoi: dict
    itos: list

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int = 80_000, min_freq: int = 2) -> "Vocab":
        c: Counter = Counter()
        for t in texts:
            c.update(tokenize(t))
        most = [w for w, n in c.most_common() if n >= min_freq][: max_size - 2]
        itos = [PAD, UNK] + most
        stoi = {w: i for i, w in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Vocab":
        itos = json.loads(path.read_text(encoding="utf-8"))["itos"]
        stoi = {w: i for i, w in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


# ---------------------------------------------------------------------------
# fastText loader
# ---------------------------------------------------------------------------
def load_fasttext_vectors(path: Path, vocab: Vocab, dim: int = 300) -> torch.Tensor:
    """Load fastText vectors into an embedding matrix aligned with `vocab`."""
    matrix = np.random.normal(0.0, 0.1, size=(len(vocab), dim)).astype("float32")
    matrix[vocab.pad_id] = 0.0
    found = 0

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().split()
        if len(header) == 2 and header[0].isdigit():
            pass  # standard fastText header
        else:
            f.seek(0)
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            tok = parts[0]
            if tok in vocab.stoi:
                matrix[vocab.stoi[tok]] = np.asarray(parts[1:], dtype="float32")
                found += 1
    print(f"  fastText: matched {found:,} / {len(vocab):,} vocab tokens "
          f"({found/len(vocab)*100:.1f}%)")
    return torch.from_numpy(matrix)


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocab, max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        ids = self.vocab.encode(tokenize(self.texts[idx]))[: self.max_len]
        return ids, int(self.labels[idx])


class Collate:
    """Picklable collate callable (Windows DataLoader workers require this)."""

    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, batch):
        ids_list, y_list = zip(*batch)
        bsz = len(ids_list)
        L = max(1, min(self.max_len, max(len(s) for s in ids_list)))
        input_ids = torch.full((bsz, L), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, L), dtype=torch.bool)
        for i, ids in enumerate(ids_list):
            n = min(len(ids), L)
            if n > 0:
                input_ids[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
                attention_mask[i, :n] = True
        y = torch.tensor(y_list, dtype=torch.long)
        return input_ids, attention_mask, y


def make_collate(pad_id: int, max_len: int) -> Collate:
    return Collate(pad_id, max_len)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype("float32")
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (num_classes * counts)
    return torch.from_numpy(w.astype("float32"))


@torch.no_grad()
def evaluate(model, loader, device, needs_mask: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct, total = 0, 0
    all_pred, all_true = [], []
    for input_ids, mask, y in loader:
        input_ids, mask, y = input_ids.to(device), mask.to(device), y.to(device)
        logits = model(input_ids, mask) if needs_mask else model(input_ids)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_pred.append(pred.cpu().numpy()); all_true.append(y.cpu().numpy())
    return correct / max(total, 1), np.concatenate(all_true), np.concatenate(all_pred)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    *,
    lr: float = 1e-3,
    epochs: int = 8,
    patience: int = 2,
    device: Optional[str] = None,
    needs_mask: bool = False,
    log_every: int = 100,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_state, best_val_acc, bad_epochs = None, -1.0, 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_steps, t0 = 0.0, 0, time.time()
        for step, (input_ids, mask, y) in enumerate(train_loader, 1):
            input_ids, mask, y = input_ids.to(device), mask.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(input_ids, mask) if needs_mask else model(input_ids)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += loss.item(); n_steps += 1
            if step % log_every == 0:
                print(f"  epoch {epoch:>2}  step {step:>5}  loss {running/n_steps:.4f}")
        train_loss = running / max(n_steps, 1)
        val_acc, *_ = evaluate(model, val_loader, device, needs_mask)
        history["train_loss"].append(train_loss); history["val_acc"].append(val_acc)
        sched.step(val_acc)
        print(f"  epoch {epoch:>2} done  ({time.time()-t0:.1f}s)   "
              f"train_loss={train_loss:.4f}   val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                print(f"  early stopping (no improvement for {bad_epochs} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_acc": best_val_acc}


@torch.no_grad()
def predict(model, loader, device, needs_mask: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for input_ids, mask, y in loader:
        input_ids, mask = input_ids.to(device), mask.to(device)
        logits = model(input_ids, mask) if needs_mask else model(input_ids)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


__all__ = [
    "Vocab",
    "TweetDataset",
    "Collate",
    "make_collate",
    "load_fasttext_vectors",
    "compute_class_weights",
    "train_model",
    "evaluate",
    "predict",
]
