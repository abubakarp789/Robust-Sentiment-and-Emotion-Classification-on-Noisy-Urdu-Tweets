from __future__ import annotations

import json

import pytest
import torch

from src.models_dl import BiLSTMAttention, TextCNN
from src.neural_utils import (
    build_vocab,
    compute_class_weights,
    encode_labels,
    encode_text,
    get_device,
    load_checkpoint,
    save_checkpoint,
)


def test_vocab_encoding_and_labels_are_deterministic() -> None:
    vocab = build_vocab(["الف ب ب", "ج ب"], max_vocab_size=5, min_frequency=1)
    encoded = encode_text("ب نامعلوم", vocab, max_sequence_length=4)
    label_to_id, id_to_label = encode_labels(["Positive", "Negative", "Neutral"])

    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert encoded == [vocab["ب"], vocab["<unk>"], 0, 0]
    assert label_to_id == {"Negative": 0, "Neutral": 1, "Positive": 2}
    assert id_to_label == {0: "Negative", 1: "Neutral", 2: "Positive"}


def test_class_weights_are_inverse_frequency() -> None:
    weights = compute_class_weights([0, 0, 0, 1, 2], num_classes=3)
    assert weights.shape == (3,)
    assert weights[0] < weights[1]
    assert weights[1] == pytest.approx(weights[2])


def test_model_forward_shapes_and_padding_masks() -> None:
    inputs = torch.tensor([[2, 3, 4, 0, 0], [4, 3, 2, 5, 6]])
    mask = inputs.ne(0)
    cnn = TextCNN(
        vocab_size=10,
        embedding_dim=8,
        num_classes=3,
        padding_idx=0,
        num_filters=4,
        kernel_sizes=[2, 3],
        dropout=0.1,
    )
    bilstm = BiLSTMAttention(
        vocab_size=10,
        embedding_dim=8,
        num_classes=3,
        padding_idx=0,
        hidden_dim=4,
        num_layers=1,
        bidirectional=True,
        dropout=0.1,
        attention=True,
    )

    assert cnn(inputs, mask).shape == (2, 3)
    assert bilstm(inputs, mask).shape == (2, 3)


def test_checkpoint_round_trip_preserves_metadata(tmp_path) -> None:
    kwargs = {
        "vocab_size": 10,
        "embedding_dim": 8,
        "num_classes": 3,
        "padding_idx": 0,
        "num_filters": 4,
        "kernel_sizes": [2, 3],
        "dropout": 0.1,
    }
    model = TextCNN(**kwargs)
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, {"model_name": "text_cnn", "model_kwargs": kwargs})

    loaded, metadata = load_checkpoint(path, TextCNN, kwargs)

    assert isinstance(loaded, TextCNN)
    assert metadata["model_name"] == "text_cnn"
    assert path.exists()


def test_get_device_rejects_unavailable_cuda() -> None:
    assert get_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            get_device("cuda")
