"""Sequence encoders and the multi-task head.

Every encoder maps token ids ``(batch, length)`` to one vector ``(batch, d_model)``. The
architecture arguments have no defaults here: the paper's settings are stated once, in the
signatures of :mod:`dsni.lit`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from dsni.features import GATE_FEATURES, relative_position
from dsni.schema import PAD, TASKS, VOCAB_SIZE

HEAD_HIDDEN = (128, 64)  # shared layers of the multi-task head
GATE_BINS = 32  # position bins of the learned gate
GATE_INIT_STD = 0.02  # "small random weights" of the gate (paper, gate analysis)


def sinusoid(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal position encoding of shape ``(length, d_model)``."""
    position = torch.arange(length, device=device).unsqueeze(1)
    rate = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    table = torch.zeros(length, d_model, device=device)
    table[:, 0::2] = torch.sin(position * rate)
    table[:, 1::2] = torch.cos(position * rate)
    return table


class CNNEncoder(nn.Module):
    """The paper's ``CNN Encoder``: stacked 1-D convolutions and a masked mean over positions."""

    def __init__(
        self,
        d_model: int,
        channels: Sequence[int],
        kernels: Sequence[int],
        dropout: float,
        embed_dim: int = 128,
    ) -> None:
        """Build the convolution stack; ``channels`` and ``kernels`` set one layer each."""
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD)
        self.blocks = nn.ModuleList()
        width = embed_dim
        for out, kernel in zip(channels, kernels, strict=True):
            conv = nn.Conv1d(width, out, kernel, padding=kernel // 2)
            block = nn.Sequential(conv, nn.BatchNorm1d(out), nn.ReLU(), nn.Dropout(dropout))
            self.blocks.append(block)
            width = out
        self.project = nn.Linear(width, d_model)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a batch of token ids; ``offsets`` is accepted for a uniform interface."""
        keep = (tokens != PAD).unsqueeze(1).float()
        hidden = self.embed(tokens).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden) * keep  # padded positions stay zero for the next layer
        return self.project(hidden.sum(dim=2) / keep.sum(dim=2).clamp(min=1.0))


class BioGate(nn.Module):
    """Variability gate ``g = sigmoid(W f + b)`` over per-position biological features.

    With the four features (region tag, relative position, local GC, local entropy) the gate
    has ``5 * d_model`` parameters, 1,280 at ``d_model = 256``. As in the paper, the bias starts
    at zero and the weights small, so every gate value starts near 0.5 and any deviation from it
    is learned.
    """

    def __init__(self, d_model: int, features: Sequence[str], dropout: float) -> None:
        """``features`` names entries of :data:`dsni.features.GATE_FEATURES`."""
        super().__init__()
        self.features = tuple(features)
        self.linear = nn.Linear(len(self.features), d_model)
        nn.init.normal_(self.linear.weight, std=GATE_INIT_STD)
        nn.init.zeros_(self.linear.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        """Gate values ``(batch, length, d_model)`` in (0, 1)."""
        stacked = torch.stack([GATE_FEATURES[f](tokens, offsets) for f in self.features], dim=-1)
        return self.dropout(torch.sigmoid(self.linear(stacked)))


class LearnedGate(nn.Module):
    """Ablation gate without biological features: one learned gate per position bin."""

    def __init__(self, d_model: int, dropout: float) -> None:
        """One embedding per equal-width bin of the relative position."""
        super().__init__()
        self.embed = nn.Embedding(GATE_BINS, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        """Gate values ``(batch, length, d_model)`` in (0, 1)."""
        index = (relative_position(tokens) * (GATE_BINS - 1)).round().long()
        return self.dropout(torch.sigmoid(self.embed(index)))


class TransformerEncoder(nn.Module):
    """``Transformer (Flat)`` with ``gate="none"``, ``Transformer + Gating`` with ``gate="bio"``.

    The gate multiplies the position-encoded embeddings before the Transformer layers; a
    learned [CLS] token is pooled at the output.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        gate: str,
        gate_features: Sequence[str],
        gate_dropout: float,
    ) -> None:
        """Build the encoder.

        Args:
            d_model: Embedding and hidden width (even).
            n_layers: Transformer layers.
            n_heads: Attention heads.
            d_ff: Feed-forward width.
            dropout: Dropout in the embeddings and the Transformer layers.
            gate: ``"none"``, ``"bio"`` (biological features) or ``"learned"`` (position bins).
            gate_features: Features of the ``"bio"`` gate; drop entries for the ablations.
            gate_dropout: Dropout on the gate values.
        """
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.gate: nn.Module | None
        if gate == "bio":
            self.gate = BioGate(d_model, gate_features, gate_dropout)
        elif gate == "learned":
            self.gate = LearnedGate(d_model, gate_dropout)
        elif gate == "none":
            self.gate = None
        else:
            raise ValueError(f"gate must be 'none', 'bio' or 'learned', got {gate!r}")
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a batch of token ids; ``offsets`` locates cropped inputs on the gene."""
        padding = tokens == PAD
        position = sinusoid(tokens.shape[1], self.embed.embedding_dim, tokens.device)
        hidden = self.dropout(self.embed(tokens) + position)
        if self.gate is not None:
            hidden = hidden * self.gate(tokens, offsets)
        hidden = torch.cat([self.cls.expand(len(hidden), -1, -1), hidden], dim=1)
        padding = torch.cat([padding.new_zeros(len(padding), 1), padding], dim=1)
        return self.layers(hidden, src_key_padding_mask=padding)[:, 0]


@torch.no_grad()
def gate_profile(
    encoder: TransformerEncoder, tokens: torch.Tensor, offsets: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean gate value at every input position, over embedding dimensions and sequences.

    This is the quantity of the paper's gate analysis. An untrained gate is near 0.5 everywhere
    (zero bias, small weights), so clear deviations from 0.5 come from training. Padding
    positions are excluded; a position no sequence reaches is NaN.
    """
    if encoder.gate is None:
        raise ValueError("the encoder has no gate")
    was_training = encoder.training
    encoder.eval()
    values = encoder.gate(tokens, offsets).mean(dim=-1)  # (batch, length)
    encoder.train(was_training)
    keep = (tokens != PAD).float()
    return (values * keep).sum(dim=0) / keep.sum(dim=0)


class MultiTaskHead(nn.Module):
    """A shared MLP followed by one linear classifier per cultivation condition."""

    def __init__(self, d_model: int, n_classes: dict[str, int], dropout: float) -> None:
        """``n_classes`` maps every task to its number of classes."""
        super().__init__()
        layers: list[nn.Module] = []
        width = d_model
        for size in HEAD_HIDDEN:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.shared = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({task: nn.Linear(width, n_classes[task]) for task in TASKS})

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits of every task."""
        shared = self.shared(encoded)
        return {task: head(shared) for task, head in self.heads.items()}
