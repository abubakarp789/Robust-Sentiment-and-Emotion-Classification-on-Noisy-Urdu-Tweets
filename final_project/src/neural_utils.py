"""Reusable vocabulary, encoding, device, and checkpoint helpers."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Type

import numpy as np
import torch
import torch.nn as nn


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(text: str) -> list[str]:
    """Tokenize already-cleaned Urdu text using whitespace boundaries."""
    return str(text).split()


def build_vocab(
    texts: Iterable[str], max_vocab_size: int, min_frequency: int
) -> dict[str, int]:
    """Build a deterministic vocabulary using training texts only."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    ordered = sorted(
        ((token, count) for token, count in counts.items() if count >= min_frequency),
        key=lambda item: (-item[1], item[0]),
    )
    tokens = [PAD_TOKEN, UNK_TOKEN] + [token for token, _ in ordered[: max_vocab_size - 2]]
    return {token: index for index, token in enumerate(tokens)}


def encode_text(
    text: str, vocab: Mapping[str, int], max_sequence_length: int
) -> list[int]:
    """Convert text into fixed-length padded token ids."""
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]
    encoded = [vocab.get(token, unk_id) for token in tokenize(text)][:max_sequence_length]
    if not encoded:
        encoded = [unk_id]
    return encoded + [pad_id] * (max_sequence_length - len(encoded))


def encode_labels(labels: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create stable alphabetical label mappings."""
    ordered = sorted(set(str(label) for label in labels))
    label_to_id = {label: index for index, label in enumerate(ordered)}
    return label_to_id, {index: label for label, index in label_to_id.items()}


def compute_class_weights(
    labels: Sequence[int], num_classes: int | None = None
) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights."""
    values = np.asarray(labels, dtype=np.int64)
    class_count = num_classes if num_classes is not None else int(values.max()) + 1
    counts = np.bincount(values, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Cannot compute weights for a class absent from training labels")
    weights = len(values) / (class_count * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_device(device_config: str) -> torch.device:
    """Return CUDA when requested/available, otherwise CPU."""
    normalized = device_config.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(normalized)


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict) -> None:
    """Atomically save model state, retrying transient Windows file locks."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_path.parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save({"state_dict": model.state_dict(), "metadata": metadata}, temporary_path)
        for attempt in range(10):
            try:
                os.replace(temporary_path, checkpoint_path)
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(0.5)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model_class: Type[nn.Module],
    model_kwargs: dict,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    """Load a model checkpoint and return the model plus metadata."""
    resolved_device = torch.device(device)
    checkpoint = torch.load(Path(path), map_location=resolved_device, weights_only=True)
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device)
    return model, checkpoint.get("metadata", {})


def save_vocab(vocab: Mapping[str, int], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(dict(vocab), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_label_mapping(
    label_to_id: Mapping[str, int], id_to_label: Mapping[int, str], path: str | Path
) -> None:
    payload = {
        "label_to_id": dict(label_to_id),
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pretrained_embeddings(
    path: str | Path, vocab: Mapping[str, int], embedding_dim: int, seed: int
) -> tuple[torch.Tensor, int]:
    """Load text-format fastText vectors aligned to the current vocabulary."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab[PAD_TOKEN]] = 0.0
    matched = 0
    embedding_path = Path(path)
    opener = __import__("gzip").open if embedding_path.suffix == ".gz" else open
    with opener(embedding_path, "rt", encoding="utf-8", errors="ignore") as handle:
        first_position = handle.tell()
        header = handle.readline().split()
        if not (len(header) == 2 and header[0].isdigit()):
            handle.seek(first_position)
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) != embedding_dim + 1:
                continue
            token = parts[0]
            token_id = vocab.get(token)
            if token_id is not None:
                matrix[token_id] = np.asarray(parts[1:], dtype=np.float32)
                matched += 1
    return torch.from_numpy(matrix), matched
