"""PyTorch neural architectures for Urdu tweet classification."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Text CNN model for Urdu tweet classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        padding_idx: int,
        num_filters: int = 128,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, embedding_dim):
                raise ValueError("Pretrained embedding shape does not match model configuration")
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=size)
            for size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(tuple(kernel_sizes)), num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = [
            F.adaptive_max_pool1d(F.relu(conv(embedded)), 1).squeeze(-1)
            for conv in self.convolutions
        ]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class AdditiveAttention(nn.Module):
    """Additive attention pooling with padding-mask support."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, feature_dim)
        self.score = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(torch.tanh(self.projection(hidden_states))).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(weights * hidden_states, dim=1)


class BiLSTMAttention(nn.Module):
    """BiLSTM with attention mechanism for Urdu tweet classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        padding_idx: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        attention: bool = True,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.use_attention = attention
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, embedding_dim):
                raise ValueError("Pretrained embedding shape does not match model configuration")
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = AdditiveAttention(output_dim) if attention else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.padding_idx)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(input_ids), lengths, batch_first=True, enforce_sorted=False
        )
        packed_hidden, _ = self.lstm(packed)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden, batch_first=True, total_length=input_ids.size(1)
        )
        if self.use_attention:
            pooled = self.attention(hidden_states, attention_mask)
        else:
            last_indices = lengths.to(hidden_states.device) - 1
            pooled = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device), last_indices
            ]
        return self.classifier(self.dropout(pooled))


__all__ = ["TextCNN", "BiLSTMAttention", "AdditiveAttention"]
