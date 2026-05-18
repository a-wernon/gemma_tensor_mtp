"""Token ↔ cluster mapping derived from Gemma 4's pretrained token_ordering.

The masked embedder exposes a `token_ordering` buffer of shape
[num_centroids, tokens_per_centroid]. For Gemma 4 E2B/E4B:
    num_centroids = 2048
    tokens_per_centroid = 128
    → covers 2048 * 128 = 262144 tokens (full vocab)

Each row contains the token ids belonging to that centroid. We invert
this map once at load time to support:
    token_id → cluster_id        (training labels)
    cluster_id → [token_ids]     (within-cluster scoring at G2+)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger


@dataclass
class ClusterSystem:
    num_centroids: int
    tokens_per_centroid: int
    vocab_size: int
    token_ordering: torch.Tensor      # [num_centroids, tokens_per_centroid] long
    token_to_cluster: torch.Tensor    # [vocab_size] long

    def cluster_of(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map any tensor of token ids to their cluster ids.

        Token ids outside [0, vocab_size) (e.g. pad ids that fall above
        the clustered range) are mapped to -1.
        """
        out = torch.full_like(token_ids, fill_value=-1, dtype=torch.long)
        valid = (token_ids >= 0) & (token_ids < self.vocab_size)
        out[valid] = self.token_to_cluster.to(token_ids.device)[token_ids[valid]]
        return out


def build_cluster_system(token_ordering: torch.Tensor) -> ClusterSystem:
    """Compute inverse mapping once. token_ordering: [num_centroids, tpc]."""
    num_c, tpc = token_ordering.shape
    vocab = num_c * tpc

    flat = token_ordering.reshape(-1).long().cpu()                # [num_c * tpc]
    cluster_for_row = (
        torch.arange(num_c).unsqueeze(1).expand(num_c, tpc).reshape(-1).long()
    )                                                              # [num_c * tpc]
    inv = torch.full((vocab,), fill_value=-1, dtype=torch.long)

    # Some Gemma builds may not cover every token id in token_ordering (if
    # vocab > num_c * tpc). If max(flat) < vocab, the gap is fine — inv
    # entries stay -1 and we treat those tokens as "no cluster" downstream.
    valid_mask = (flat >= 0) & (flat < vocab)
    inv[flat[valid_mask]] = cluster_for_row[valid_mask]

    covered = int((inv >= 0).sum().item())
    if covered < vocab:
        logger.warning(
            f"token_ordering covers {covered}/{vocab} token ids; uncovered "
            "ids will be labeled cluster=-1 and skipped by the loss."
        )

    return ClusterSystem(
        num_centroids=num_c,
        tokens_per_centroid=tpc,
        vocab_size=vocab,
        token_ordering=token_ordering.long().cpu(),
        token_to_cluster=inv,
    )
