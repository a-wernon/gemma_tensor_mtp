"""MTPC-CP cluster head — rank-r shallow CP mixture, memory-efficient.

Per MTPC §3.2:
    q(c_{+1..+n} | h) = Σ_α w_α(h) · Π_i q_{i,α}(c_{+i} | h)

with q_{i,α}(c) = softmax(W_{i,α} h)[c] and w_α(h) = softmax(R h)[α].

Parameterisations:

  full          W ∈ R^{n×r×V×d}                            (~800M @ default)
  shared_trunk  W_{i,α} = U diag(s_{i,α}) V_i               (~70M @ default)
                  U ∈ R^{V × k}, V_i ∈ R^{k × d}, s ∈ R^{n × r × k}

CP-r=1 reduces exactly to FF (single mixture component, weight 1).

Memory note. The naïve forward materialises [B, L, n, r, V] for the
per-(offset, component) logits. At default shape (B=2, L=512, n=8, r=32,
V=2048) that's ~8.6 GB in fp32 and another tensor of the same size for
backward — total ~17 GB just for this head. Instead we:

  * compute the partition function log Σ_v exp(W_{i,α} h)_v by chunked
    logsumexp over V (peak chunk = [B, L, n, r, V_chunk]);
  * compute the per-(offset, component) logit AT THE LABEL via
    label-gather on U, never touching the full V dimension.

The big tensor never exists. Peak intermediate becomes [B, L, n, r, k]
(~270 MB at k=256) plus one [B, L, n, r, V_chunk] chunk inside the
partition loop, which is freed at the end of each chunk by autograd
since the chunk's contribution to the running logsumexp is incorporated
via a stable max-shifted update.

per_offset_logits() (full marginals, only used for diagnostics / eval)
is kept but computed with the same chunked-over-V path so peak memory
during eval is also bounded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .base import MTPCHead


_V_CHUNK_DEFAULT = 256        # chunk size over V for partition / marginals


def _chunk_logsumexp(
    scaled: torch.Tensor,                     # [B, L, n, r, k]
    U_chunk: torch.Tensor,                    # [Vc, k]
    bias_chunk: torch.Tensor | None,          # [n, r, Vc] or None
) -> torch.Tensor:
    """logsumexp over the V dim of (scaled @ U_chunk.T + bias_chunk).

    Returns [B, L, n, r]. Wrapped in checkpoint() at the call site so
    the intermediate [B, L, n, r, Vc] tensor is freed after forward and
    recomputed in backward.
    """
    logits = torch.einsum("blnrk,vk->blnrv", scaled, U_chunk)
    if bias_chunk is not None:
        logits = logits + bias_chunk.unsqueeze(0).unsqueeze(0)
    return torch.logsumexp(logits, dim=-1)


def _chunked_logsumexp_over_V(
    scaled: torch.Tensor,                     # [B, L, n, r, k]
    U: torch.Tensor,                          # [V, k]
    bias: torch.Tensor | None,                # [n, r, V] or None
    v_chunk: int,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Return log Σ_v exp(scaled @ U[v].T + bias[v]) per (B, L, n, r).

    Stable across chunks: maintains running max, combines via
    M + log(exp(prev - M) + exp(chunk - M)).

    With use_checkpoint=True the per-chunk [B, L, n, r, Vc] tensor is
    discarded after forward and recomputed in backward. Peak memory
    inside backward is one chunk-sized tensor.
    """
    V = U.shape[0]
    accum_max = None
    accum_lse = None
    for v_start in range(0, V, v_chunk):
        v_end = min(v_start + v_chunk, V)
        U_chunk = U[v_start:v_end]                                   # [Vc, k]
        bias_chunk = bias[:, :, v_start:v_end] if bias is not None else None

        if use_checkpoint and scaled.requires_grad:
            chunk_lse = checkpoint(
                _chunk_logsumexp, scaled, U_chunk, bias_chunk,
                use_reentrant=False,
            )
        else:
            chunk_lse = _chunk_logsumexp(scaled, U_chunk, bias_chunk)

        if accum_lse is None:
            accum_lse = chunk_lse
            accum_max = chunk_lse.detach()
        else:
            new_max = torch.maximum(accum_max, chunk_lse.detach())
            accum_lse = new_max + torch.log(
                torch.exp(accum_lse - new_max) + torch.exp(chunk_lse - new_max)
            )
            accum_max = new_max
    return accum_lse


def _logit_at_label(
    scaled: torch.Tensor,                     # [B, L, n, r, k]
    U: torch.Tensor,                          # [V, k]
    bias: torch.Tensor | None,                # [n, r, V] or None
    label: torch.Tensor,                      # [B, L, n], values in [0, V) or -1
) -> torch.Tensor:
    """Per-(offset, component) logit evaluated at the ground-truth label.

    Returns [B, L, n, r]. Positions with label = -1 are still gathered
    (with safe index 0); downstream masking handles them.
    """
    safe = label.clamp_min(0)                                       # [B, L, n]
    U_at_label = U[safe]                                            # [B, L, n, k]
    logit = torch.einsum("blnk,blnrk->blnr", U_at_label, scaled)    # [B, L, n, r]
    if bias is not None:
        # bias[n, r, V] gathered at label: advanced indexing. The
        # intermediate index tensors are int and small; no [B,L,n,r,V]
        # materialised.
        B, L, n = label.shape
        r = bias.shape[1]
        n_idx = torch.arange(n, device=label.device).view(1, 1, n, 1).expand(B, L, n, r)
        r_idx = torch.arange(r, device=label.device).view(1, 1, 1, r).expand(B, L, n, r)
        label_idx = safe.unsqueeze(-1).expand(B, L, n, r)
        logit = logit + bias[n_idx, r_idx, label_idx]
    return logit


def _full_marginals_chunked(
    scaled: torch.Tensor,                     # [B, L, n, r, k]
    U: torch.Tensor,                          # [V, k]
    bias: torch.Tensor | None,                # [n, r, V] or None
    log_w: torch.Tensor,                      # [B, L, r]
    v_chunk: int,
) -> torch.Tensor:
    """Return [B, L, n, V] log-marginals log Σ_α w_α(h) q_{i,α}(v | h).

    Used for diagnostics / eval. Computed in V chunks; never materialises
    the full [B, L, n, r, V] slab. Peak intermediate is one chunk.
    """
    V = U.shape[0]
    # First compute log_partition over V for normalisation.
    log_Z = _chunked_logsumexp_over_V(scaled, U, bias, v_chunk)         # [B,L,n,r]

    out_chunks: list[torch.Tensor] = []
    for v_start in range(0, V, v_chunk):
        v_end = min(v_start + v_chunk, V)
        U_chunk = U[v_start:v_end]
        logits_chunk = torch.einsum("blnrk,vk->blnrv", scaled, U_chunk)
        if bias is not None:
            logits_chunk = logits_chunk + bias[:, :, v_start:v_end].unsqueeze(0).unsqueeze(0)
        # log q_{i,α}(v) = logits - log_Z
        log_q_chunk = logits_chunk - log_Z.unsqueeze(-1)               # [B,L,n,r,Vc]
        # log p_i(v) = logsumexp_α (log w_α + log q_{i,α}(v))
        log_w_b = log_w.unsqueeze(2).unsqueeze(-1)                     # [B,L,1,r,1]
        log_p_chunk = torch.logsumexp(log_w_b + log_q_chunk, dim=3)    # [B,L,n,Vc]
        out_chunks.append(log_p_chunk)
    return torch.cat(out_chunks, dim=-1)                               # [B,L,n,V]


class CPClusterHead(MTPCHead):
    def __init__(
        self,
        hidden_size: int,
        n_clusters: int,
        n_future: int,
        rank: int = 32,
        parameterisation: str = "shared_trunk",
        trunk_dim: int = 256,
        v_chunk: int = _V_CHUNK_DEFAULT,
    ):
        super().__init__(hidden_size, n_clusters, n_future)
        self.rank = rank
        self.parameterisation = parameterisation
        self.trunk_dim = trunk_dim
        self.v_chunk = v_chunk

        if parameterisation == "shared_trunk":
            # U: [V, k]; V: [n, k, d]; s: [n, r, k]
            self.U = nn.Parameter(torch.empty(n_clusters, trunk_dim))
            self.V = nn.Parameter(torch.empty(n_future, trunk_dim, hidden_size))
            self.s = nn.Parameter(torch.ones(n_future, rank, trunk_dim))
            self.b = nn.Parameter(torch.zeros(n_future, rank, n_clusters))
            nn.init.normal_(self.U, mean=0.0, std=0.02)
            nn.init.normal_(self.V, mean=0.0, std=0.02)
            nn.init.normal_(self.s, mean=1.0, std=0.02)
        elif parameterisation == "full":
            # W ∈ [n, r, V, d]; only viable at small V.
            self.W = nn.Parameter(torch.empty(n_future, rank, n_clusters, hidden_size))
            self.b = nn.Parameter(torch.zeros(n_future, rank, n_clusters))
            nn.init.normal_(self.W, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown parameterisation: {parameterisation}")

        # Mixture weight head: w_α(h) = softmax(R h)[α]
        self.R = nn.Parameter(torch.empty(rank, hidden_size))
        nn.init.normal_(self.R, mean=0.0, std=0.02)

    # ---------------------------------------------------------------- forward primitives

    def _scaled(self, h: torch.Tensor) -> torch.Tensor:
        """Return scaled[B,L,n,r,k] for shared_trunk; raises for full mode
        (full mode forces materialising [B,L,n,r,V] and is only intended
        for small-V sanity checks)."""
        if self.parameterisation != "shared_trunk":
            raise RuntimeError(
                "_scaled() only supports shared_trunk. For full mode, use "
                "_per_offset_per_component_logits_full() (small-V only)."
            )
        h32 = h.float()
        Vh = torch.einsum("nkd,bld->blnk", self.V, h32)                 # [B,L,n,k]
        # s: [n, r, k]; broadcast over B, L
        scaled = self.s.unsqueeze(0).unsqueeze(0) * Vh.unsqueeze(3)     # [B,L,n,r,k]
        return scaled

    def _log_w(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(torch.einsum("rd,bld->blr", self.R, h.float()), dim=-1)

    def _per_offset_per_component_logits_full(self, h: torch.Tensor) -> torch.Tensor:
        """Materialise [B,L,n,r,V]. Only for small V or sanity checks."""
        assert self.parameterisation == "full", "use _scaled() for shared_trunk"
        return torch.einsum("bld,nrvd->blnrv", h.float(), self.W) + self.b

    # ---------------------------------------------------------------- public API

    def per_offset_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Marginal per-offset log-probability table [B, L, n, V].

        Computed in chunks over V; used for diagnostic plots / eval. The
        returned tensor is log p_i(v | h), already normalised.
        """
        if self.parameterisation == "full":
            comp_logits = self._per_offset_per_component_logits_full(h)
            comp_logp = torch.log_softmax(comp_logits, dim=-1)
            log_w = self._log_w(h)                                       # [B,L,r]
            return torch.logsumexp(
                comp_logp + log_w.unsqueeze(2).unsqueeze(-1), dim=3
            )

        scaled = self._scaled(h)
        log_w = self._log_w(h)
        return _full_marginals_chunked(scaled, self.U, self.b, log_w, self.v_chunk)

    def joint_logprob(self, h: torch.Tensor, cluster_lbl: torch.Tensor) -> torch.Tensor:
        """log Σ_α w_α(h) · Π_i q_{i,α}(c_{+i} | h).

        Memory-efficient: never materialises [B, L, n, r, V].
        """
        if self.parameterisation == "full":
            comp_logits = self._per_offset_per_component_logits_full(h)  # [B,L,n,r,V]
            comp_logp = torch.log_softmax(comp_logits, dim=-1)
            safe = cluster_lbl.clamp_min(0)
            idx = safe.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.rank, 1)
            gathered = comp_logp.gather(-1, idx).squeeze(-1)              # [B,L,n,r]
            mask = (cluster_lbl != -1).unsqueeze(-1).float()
            sum_over_offsets = (gathered * mask).sum(dim=2)               # [B,L,r]
            log_w = self._log_w(h)
            return torch.logsumexp(log_w + sum_over_offsets, dim=-1)

        scaled = self._scaled(h)
        log_w = self._log_w(h)
        log_Z = _chunked_logsumexp_over_V(scaled, self.U, self.b, self.v_chunk)
        logit_at_label = _logit_at_label(scaled, self.U, self.b, cluster_lbl)
        log_q_at_label = logit_at_label - log_Z                            # [B,L,n,r]
        mask = (cluster_lbl != -1).unsqueeze(-1).float()                  # [B,L,n,1]
        sum_over_offsets = (log_q_at_label * mask).sum(dim=2)             # [B,L,r]
        return torch.logsumexp(log_w + sum_over_offsets, dim=-1)          # [B,L]

    def _log_marg_at_label(
        self, h: torch.Tensor, cluster_lbl: torch.Tensor
    ) -> torch.Tensor:
        """Per-offset log p_i(c_i | h) at the label, [B, L, n]. Memory-efficient.

        Used for the MTPC per-offset NLL loss (paper Eq. 4) without
        materialising the full V dimension.
        """
        if self.parameterisation == "full":
            marg = self.per_offset_logits(h)                              # [B,L,n,V]
            safe = cluster_lbl.clamp_min(0)
            return marg.gather(-1, safe.unsqueeze(-1)).squeeze(-1)

        scaled = self._scaled(h)
        log_w = self._log_w(h)
        log_Z = _chunked_logsumexp_over_V(scaled, self.U, self.b, self.v_chunk)
        logit_at_label = _logit_at_label(scaled, self.U, self.b, cluster_lbl)
        log_q_at_label = logit_at_label - log_Z                            # [B,L,n,r]
        # log p_i(c_i) = logsumexp_α (log w_α + log q_{i,α}(c_i))
        return torch.logsumexp(
            log_w.unsqueeze(2) + log_q_at_label, dim=-1
        )                                                                  # [B,L,n]

    # ---------------------------------------------------------------- loss / eval

    def loss(
        self,
        h: torch.Tensor,
        cluster_lbl: torch.Tensor,
        loss_mask: torch.Tensor,
        gamma: float = 0.9,
    ) -> tuple[torch.Tensor, dict]:
        """MTPC Eq. (4) per-offset marginal NLL with exponential discount.

        Memory-efficient — no full [B,L,n,r,V] tensor in either forward
        or backward. Computes scaled / log_w / log_Z / logit_at_label
        ONCE and reuses them across the marginal and joint terms.
        Joint NLL is added (no discount) to keep mixture coupling alive
        during early training (otherwise marginal MLE can drift toward
        rank-1).
        """
        if self.parameterisation == "full":
            return self._loss_full(h, cluster_lbl, loss_mask, gamma)

        scaled = self._scaled(h)                                          # [B,L,n,r,k]
        log_w = self._log_w(h)                                            # [B,L,r]
        log_Z = _chunked_logsumexp_over_V(scaled, self.U, self.b, self.v_chunk)
        logit_at_label = _logit_at_label(scaled, self.U, self.b, cluster_lbl)
        log_q_at_label = logit_at_label - log_Z                           # [B,L,n,r]

        valid = (cluster_lbl != -1) & loss_mask.unsqueeze(-1)             # [B,L,n]
        mask_f = (cluster_lbl != -1).unsqueeze(-1).float()                # [B,L,n,1]

        # Per-offset marginal: log p_i(c_i) = logsumexp_α (log w_α + log q_{i,α}(c_i))
        log_marg = torch.logsumexp(log_w.unsqueeze(2) + log_q_at_label, dim=-1)  # [B,L,n]
        nll = -log_marg
        n = self.n_future
        discount = torch.tensor(
            [gamma ** k for k in range(n)], device=nll.device
        ).view(1, 1, n)
        weighted = nll * discount * valid.float()
        denom_marg = valid.float().sum().clamp_min(1.0)
        marg_loss = weighted.sum() / denom_marg

        # Joint NLL (no discount): log Σ_α w_α Π_i q_{i,α}(c_i)
        sum_over_offsets = (log_q_at_label * mask_f).sum(dim=2)           # [B,L,r]
        jlp = torch.logsumexp(log_w + sum_over_offsets, dim=-1)           # [B,L]
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

    def _loss_full(
        self,
        h: torch.Tensor,
        cluster_lbl: torch.Tensor,
        loss_mask: torch.Tensor,
        gamma: float,
    ) -> tuple[torch.Tensor, dict]:
        """Fallback for parameterisation='full'. Materialises [B,L,n,r,V]."""
        log_marg = self._log_marg_at_label(h, cluster_lbl)
        nll = -log_marg
        valid = (cluster_lbl != -1) & loss_mask.unsqueeze(-1)
        n = self.n_future
        discount = torch.tensor(
            [gamma ** k for k in range(n)], device=nll.device
        ).view(1, 1, n)
        weighted = nll * discount * valid.float()
        denom_marg = valid.float().sum().clamp_min(1.0)
        marg_loss = weighted.sum() / denom_marg

        jlp = self.joint_logprob(h, cluster_lbl)
        full_valid = ((cluster_lbl != -1).all(dim=-1)) & loss_mask
        denom_joint = full_valid.float().sum().clamp_min(1.0)
        joint_loss = (-jlp * full_valid.float()).sum() / denom_joint

        total = marg_loss + joint_loss
        per_offset = (nll * valid.float()).sum(dim=(0, 1)) / valid.float().sum(
            dim=(0, 1)
        ).clamp_min(1.0)
        return total, {
            "loss": float(total.detach()),
            "marg_loss": float(marg_loss.detach()),
            "joint_loss": float(joint_loss.detach()),
            "per_offset_nll": per_offset.detach().tolist(),
            "n_valid_offset": int(valid.sum().item()),
            "n_valid_joint": int(full_valid.sum().item()),
        }
