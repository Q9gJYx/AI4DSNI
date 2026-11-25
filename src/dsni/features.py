"""Per-position biological features that condition the variability gate.

Every feature maps token ids ``(batch, length)`` to values of the same shape. ``offsets``
gives, per row, the gene position of the first token; it is non-zero only when the input
was cropped, and only the region tags depend on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from dsni.schema import PAD, VOCAB_SIZE

WINDOW = 21  # width of the sliding window of the GC and entropy features

# Hypervariable regions V1 to V9 in E. coli 16S coordinates (0-based, end exclusive). They are
# applied to raw positions, so the tags are approximate for sequences that do not start near
# the 5' end of the gene; an alignment-based annotation is on the roadmap.
V_REGIONS: tuple[tuple[int, int], ...] = (
    (69, 99),
    (137, 242),
    (433, 497),
    (576, 682),
    (822, 879),
    (986, 1043),
    (1117, 1173),
    (1243, 1294),
    (1435, 1465),
)


def region_tags(tokens: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
    """1 inside a hypervariable region, 0 in conserved regions."""
    position = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
    if offsets is not None:
        position = position + offsets[:, None]
    tags = torch.zeros(tokens.shape, device=tokens.device)
    for start, end in V_REGIONS:
        tags += (position >= start) & (position < end)
    return tags


def relative_position(tokens: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
    """Position divided by input length, in [0, 1]; padding positions are clamped to 1."""
    length = (tokens != PAD).sum(dim=1, keepdim=True).clamp(min=2)
    index = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
    return (index / (length - 1)).clamp(max=1.0)


def _window_counts(tokens: torch.Tensor) -> torch.Tensor:
    """Counts of A, C, G, T in a centred window: ``(batch, 4, length)``; N and PAD count as none."""
    onehot = F.one_hot(tokens, num_classes=VOCAB_SIZE)[..., 1:5].float().transpose(1, 2)
    return F.avg_pool1d(onehot, WINDOW, stride=1, padding=WINDOW // 2) * WINDOW


def local_gc(tokens: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
    """Share of G and C among the bases of a centred window."""
    counts = _window_counts(tokens)
    return (counts[:, 1] + counts[:, 2]) / counts.sum(dim=1).clamp(min=1.0)


def local_entropy(tokens: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
    """Shannon entropy of the base composition of a centred window, scaled to [0, 1]."""
    counts = _window_counts(tokens)
    p = counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.special.entr(p).sum(dim=1) / math.log(4)


Feature = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
GATE_FEATURES: dict[str, Feature] = {
    "region": region_tags,
    "position": relative_position,
    "gc": local_gc,
    "entropy": local_entropy,
}
