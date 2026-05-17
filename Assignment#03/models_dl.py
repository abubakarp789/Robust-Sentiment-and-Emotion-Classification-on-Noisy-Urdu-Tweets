"""PyTorch model definitions for the CNN and BiLSTM-attention classifiers.

The architectures realise the equations in Sections 4.3 and 4.4 of
`04_architecture_math.md`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-kernel 1D CNN (Kim, 2014)
# ---------------------------------------------------------------------------
class TextCNN(nn.Module):
    """Multi-kernel 1-D convolutional sentence classifier."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 300,
        filter_sizes: Sequence[int] = (3, 4, 5),
        num_filters: int = 128,
        dropout: float = 0.5,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, embed_dim)
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=h)
                for h in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L)
        x = self.embedding(input_ids)                # (B, L, D)
        x = x.transpose(1, 2)                        # (B, D, L)  — conv expects channels first
        feats = []
        for conv in self.convs:
            c = F.relu(conv(x))                      # (B, F, L - h + 1)
            c = F.adaptive_max_pool1d(c, 1).squeeze(2)  # (B, F)
            feats.append(c)
        z = torch.cat(feats, dim=1)                  # (B, 3F)
        z = self.dropout(z)
        return self.classifier(z)


# ---------------------------------------------------------------------------
# BiLSTM with additive attention pooling
# ---------------------------------------------------------------------------
class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over time steps."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # H: (B, L, 2H), mask: (B, L) with 1 for real tokens
        scores = self.u(torch.tanh(self.W(H))).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        alphas = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        return torch.sum(alphas * H, dim=1)                  # (B, 2H)


class BiLSTMAttn(nn.Module):
    """Two-layer BiLSTM with additive attention pooling."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 300,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, embed_dim)
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AdditiveAttention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        H, _ = self.lstm(x)            # (B, L, 2H)
        z = self.attn(H, attention_mask)
        z = self.dropout(z)
        return self.classifier(z)


__all__ = ["TextCNN", "BiLSTMAttn", "AdditiveAttention"]
