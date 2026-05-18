"""Abstract base for cluster-level MTPC heads.

Inputs:
  h            [B, L, d]   target hidden states
  cluster_lbl  [B, L, n]   ground-truth cluster ids for offsets +1..+n
                            (-1 = ignore in loss)
  loss_mask    [B, L]      bool, True where this position contributes
                            to the joint NLL

A head implements:
  joint_logprob(h, cluster_lbl) -> [B, L]
      log q(c_{+1}, ..., c_{+n} | h) at every position (or 0 where
      cluster_lbl == -1 in any of the n offsets).

  loss(h, cluster_lbl, loss_mask, gamma) -> scalar
      mean masked NLL with exponential per-offset discount γ^{k} as in
      MTPC paper Eq. (4).

  per_offset_nll(h, cluster_lbl, loss_mask) -> [n]
      diagnostic — per-offset NLL averaged over masked positions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class MTPCHead(nn.Module, ABC):
    def __init__(self, hidden_size: int, n_clusters: int, n_future: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_clusters = n_clusters
        self.n_future = n_future

    @abstractmethod
    def per_offset_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Return [B, L, n_future, n_clusters] logits per offset.

        For mixture heads, this is the *marginal* per-offset distribution
        (Σ_α w_α · q_{i,α}). The joint logprob may differ from a
        product of marginals; subclasses override joint_logprob().
        """
        ...

    def joint_logprob(
        self, h: torch.Tensor, cluster_lbl: torch.Tensor
    ) -> torch.Tensor:
        """Default: factorised joint = sum of per-offset logprobs."""
        logits = self.per_offset_logits(h)                # [B, L, n, V]
        logp = torch.log_softmax(logits.float(), dim=-1)  # [B, L, n, V]
        # Gather at labels; mask -1 → 0 contribution.
        safe = cluster_lbl.clamp_min(0)
        gathered = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # [B,L,n]
        gathered = gathered.masked_fill(cluster_lbl == -1, 0.0)
        return gathered.sum(dim=-1)                       # [B, L]

    def loss(
        self,
        h: torch.Tensor,
        cluster_lbl: torch.Tensor,
        loss_mask: torch.Tensor,
        gamma: float = 0.9,
    ) -> tuple[torch.Tensor, dict]:
        logits = self.per_offset_logits(h)                # [B, L, n, V]
        logp = torch.log_softmax(logits.float(), dim=-1)
        safe = cluster_lbl.clamp_min(0)
        gathered = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # [B,L,n]
        valid = (cluster_lbl != -1) & loss_mask.unsqueeze(-1)

        nll = -gathered                                     # [B, L, n]
        n = self.n_future
        discount = torch.tensor(
            [gamma ** k for k in range(n)], device=nll.device
        ).view(1, 1, n)
        weighted = nll * discount * valid.float()

        # Normalise by number of valid (pos, offset) entries.
        denom = valid.float().sum().clamp_min(1.0)
        total = weighted.sum() / denom
        per_offset = (nll * valid.float()).sum(dim=(0, 1)) / valid.float().sum(
            dim=(0, 1)
        ).clamp_min(1.0)

        stats = {
            "loss": float(total.detach()),
            "per_offset_nll": per_offset.detach().tolist(),
            "n_valid": int(valid.sum().item()),
        }
        return total, stats

    @torch.no_grad()
    def per_offset_nll(
        self, h: torch.Tensor, cluster_lbl: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.per_offset_logits(h)
        logp = torch.log_softmax(logits.float(), dim=-1)
        safe = cluster_lbl.clamp_min(0)
        gathered = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        valid = (cluster_lbl != -1) & loss_mask.unsqueeze(-1)
        nll = (-gathered * valid.float()).sum(dim=(0, 1))
        denom = valid.float().sum(dim=(0, 1)).clamp_min(1.0)
        return nll / denom                                 # [n]

    @torch.no_grad()
    def joint_nll(
        self, h: torch.Tensor, cluster_lbl: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean joint NLL across masked positions. Scalar."""
        jlp = self.joint_logprob(h, cluster_lbl)              # [B, L]
        # Only positions with FULL valid window contribute meaningfully.
        full_valid = ((cluster_lbl != -1).all(dim=-1)) & loss_mask
        nll = (-jlp * full_valid.float()).sum()
        denom = full_valid.float().sum().clamp_min(1.0)
        return nll / denom
