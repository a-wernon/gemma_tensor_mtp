"""MTPC-FF cluster head — fully factorised (rank-1 baseline).

Per MTPC §3.2: input units are n independent unembedding layers, one per
window position. q(c_{+1..+n} | h) = Π_i softmax(W_i h)[c_{+i}].

Equivalent to MTPC-CP at r=1. This is the honest baseline against which
we measure the CP joint-structure gain in G1.

Parameter count: n × n_clusters × d.
For n=8, V=2048, d=1536: ~25M params.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import MTPCHead


class FFClusterHead(MTPCHead):
    def __init__(self, hidden_size: int, n_clusters: int, n_future: int):
        super().__init__(hidden_size, n_clusters, n_future)
        # One linear per future offset. We stack them into a single tensor
        # W ∈ [n, V, d] for vectorised forward.
        self.W = nn.Parameter(
            torch.empty(n_future, n_clusters, hidden_size)
        )
        self.b = nn.Parameter(torch.zeros(n_future, n_clusters))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def per_offset_logits(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, L, d]; W: [n, V, d] → [B, L, n, V]
        # einsum: blD, nVD -> blnV
        return torch.einsum("bld,nvd->blnv", h.float(), self.W) + self.b
