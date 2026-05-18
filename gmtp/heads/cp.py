"""MTPC-CP cluster head — rank-r shallow CP mixture.

Per MTPC §3.2:
    q(c_{+1..+n} | h) = Σ_α w_α(h) · Π_i q_{i,α}(c_{+i} | h)

where:
  * α ∈ {1..r} is a discrete latent (mixture component)
  * q_{i,α}(c) = softmax(W_{i,α} h)[c]
  * w_α(h) = softmax(R h)[α]    (mixture weights conditional on h)

Two parameterisations are supported:

  full        W ∈ R^{n×r×V×d}   — exact MTPC parameterisation
                                   For n=8, r=32, V=2048, d=1536: ~800M.
  shared_trunk  W_{i,α} = U_i diag(s_{i,α}) V
                                   U_i ∈ R^{V × k}, V ∈ R^{k × d},
                                   s ∈ R^{n × r × k}
                                   For k=256: ~70M.

For G1 default = shared_trunk (cheaper, still expressive).

CP-r=1 reduces exactly to FF (the rank-1 mixture has one component with
weight 1, identical to a factorised head). MTPC paper Section 3.2
confirms this — we use it as the FF↔CP sanity check.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import MTPCHead


class CPClusterHead(MTPCHead):
    def __init__(
        self,
        hidden_size: int,
        n_clusters: int,
        n_future: int,
        rank: int = 32,
        parameterisation: str = "shared_trunk",
        trunk_dim: int = 256,
    ):
        super().__init__(hidden_size, n_clusters, n_future)
        self.rank = rank
        self.parameterisation = parameterisation
        self.trunk_dim = trunk_dim

        if parameterisation == "full":
            self.W = nn.Parameter(
                torch.empty(n_future, rank, n_clusters, hidden_size)
            )
            self.b = nn.Parameter(torch.zeros(n_future, rank, n_clusters))
            nn.init.normal_(self.W, mean=0.0, std=0.02)
        elif parameterisation == "shared_trunk":
            # U: [V, k]; V: [n, k, d]; s: [n, r, k]
            self.U = nn.Parameter(torch.empty(n_clusters, trunk_dim))
            self.V = nn.Parameter(torch.empty(n_future, trunk_dim, hidden_size))
            self.s = nn.Parameter(torch.ones(n_future, rank, trunk_dim))
            self.b = nn.Parameter(torch.zeros(n_future, rank, n_clusters))
            nn.init.normal_(self.U, mean=0.0, std=0.02)
            nn.init.normal_(self.V, mean=0.0, std=0.02)
            nn.init.normal_(self.s, mean=1.0, std=0.02)
        else:
            raise ValueError(f"Unknown parameterisation: {parameterisation}")

        # Mixture weight head: w_α(h) = softmax(R h)[α]
        self.R = nn.Parameter(torch.empty(rank, hidden_size))
        nn.init.normal_(self.R, mean=0.0, std=0.02)

    def _per_offset_per_component_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Return [B, L, n, r, V] logits q_{i,α}(c | h) before softmax."""
        h32 = h.float()
        if self.parameterisation == "full":
            # W: [n, r, V, d] → einsum: bld, nrVd -> blnrV
            return torch.einsum("bld,nrvd->blnrv", h32, self.W) + self.b
        # shared_trunk:
        #   z_{i,α}(h) = U @ diag(s_{i,α}) @ V_i @ h   ∈ R^V
        #   Compute Vh: [n, k, d] · [B, L, d] -> [B, L, n, k]
        Vh = torch.einsum("nkd,bld->blnk", self.V, h32)
        # scaled: s[n, r, k] * Vh[B, L, n, k] -> [B, L, n, r, k]
        scaled = self.s.unsqueeze(0).unsqueeze(0) * Vh.unsqueeze(3)
        # logits: U[V, k] @ scaled[..., k] -> [B, L, n, r, V]
        logits = torch.einsum("vk,blnrk->blnrv", self.U, scaled)
        return logits + self.b

    def per_offset_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Marginal per-offset distribution (Σ_α w_α · q_{i,α}).

        NOTE: this is NOT used directly by joint_logprob() — we override
        that below to compute the proper joint via logsumexp over α.
        per_offset_logits exists for diagnostics / Pareto plots.
        """
        comp_logits = self._per_offset_per_component_logits(h)    # [B,L,n,r,V]
        comp_logp = torch.log_softmax(comp_logits, dim=-1)
        log_w = torch.log_softmax(self.R @ h.float().transpose(-1, -2), dim=-2)
        # log_w: [B, L, r]   (computed via [r, d] @ [B, d, L] then transposed)
        # Recompute cleanly:
        log_w = torch.log_softmax(
            torch.einsum("rd,bld->blr", self.R, h.float()), dim=-1
        )
        # log( Σ_α w_α · q_{i,α}(c) ) per (i, c)
        # = logsumexp over α of (log_w[α] + comp_logp[α, c]) for each i
        # comp_logp: [B, L, n, r, V] ; log_w: [B, L, r]
        marg = torch.logsumexp(
            comp_logp + log_w.unsqueeze(2).unsqueeze(-1), dim=3
        )                                                          # [B, L, n, V]
        return marg

    def joint_logprob(
        self, h: torch.Tensor, cluster_lbl: torch.Tensor
    ) -> torch.Tensor:
        """log Σ_α w_α(h) · Π_i q_{i,α}(c_{+i} | h).

        Cleanly tractable: gather log q_{i,α}(c_{+i}) for each α, sum
        over i, add log w_α, logsumexp over α.
        """
        comp_logits = self._per_offset_per_component_logits(h)    # [B,L,n,r,V]
        comp_logp = torch.log_softmax(comp_logits, dim=-1)
        # Gather at labels per (i, α). cluster_lbl: [B, L, n], -1 → ignore.
        safe = cluster_lbl.clamp_min(0)                            # [B, L, n]
        # broadcast to [B, L, n, r, 1]
        idx = safe.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.rank, 1)
        gathered = comp_logp.gather(-1, idx).squeeze(-1)           # [B, L, n, r]
        # Mask out positions where label is -1: contribution = 0.
        mask = (cluster_lbl != -1).unsqueeze(-1).float()           # [B, L, n, 1]
        per_component_logp_per_offset = gathered * mask            # [B, L, n, r]
        sum_over_offsets = per_component_logp_per_offset.sum(dim=2)  # [B, L, r]

        log_w = torch.log_softmax(
            torch.einsum("rd,bld->blr", self.R, h.float()), dim=-1
        )                                                          # [B, L, r]
        return torch.logsumexp(log_w + sum_over_offsets, dim=-1)   # [B, L]

    def loss(
        self,
        h: torch.Tensor,
        cluster_lbl: torch.Tensor,
        loss_mask: torch.Tensor,
        gamma: float = 0.9,
    ) -> tuple[torch.Tensor, dict]:
        """Joint NLL with exponential per-offset discount.

        Subtlety: the MTPC joint NLL is on the full window. Exponential
        discount γ^k is a per-offset weighting that does not factor
        cleanly into the joint logsumexp. We follow MTPC paper Eq. (4)
        and compute per-offset NLLs from the *marginals* with discount,
        then add the joint NLL term (no discount) as a single scalar
        regulariser. In practice for r ≥ 2 the joint term is the one
        that drives mixture learning; the per-offset discounted term
        keeps the marginals well-calibrated.
        """
        # Per-offset marginal NLL with discount (for marginal calibration).
        marg_logits = self.per_offset_logits(h)                    # [B,L,n,V]
        safe = cluster_lbl.clamp_min(0)
        marg_gathered = marg_logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        # marg_logits is already log-marginals via logsumexp → these ARE
        # log-probs already. No additional log_softmax needed.
        valid = (cluster_lbl != -1) & loss_mask.unsqueeze(-1)
        nll = -marg_gathered                                       # [B, L, n]

        n = self.n_future
        discount = torch.tensor(
            [gamma ** k for k in range(n)], device=nll.device
        ).view(1, 1, n)
        weighted = nll * discount * valid.float()
        denom_marg = valid.float().sum().clamp_min(1.0)
        marg_loss = weighted.sum() / denom_marg

        # Joint NLL (no discount).
        jlp = self.joint_logprob(h, cluster_lbl)                   # [B, L]
        full_valid = ((cluster_lbl != -1).all(dim=-1)) & loss_mask
        denom_joint = full_valid.float().sum().clamp_min(1.0)
        joint_loss = (-jlp * full_valid.float()).sum() / denom_joint

        total = marg_loss + joint_loss
        per_offset = (nll * valid.float()).sum(dim=(0, 1)) / valid.float().sum(
            dim=(0, 1)
        ).clamp_min(1.0)

        stats = {
            "loss": float(total.detach()),
            "marg_loss": float(marg_loss.detach()),
            "joint_loss": float(joint_loss.detach()),
            "per_offset_nll": per_offset.detach().tolist(),
            "n_valid_offset": int(valid.sum().item()),
            "n_valid_joint": int(full_valid.sum().item()),
        }
        return total, stats
