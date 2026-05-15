"""Gemma 4 target + assistant loading, plus masked-embedder hook for cluster-mask capture.

API ASSUMPTIONS (verify on first run; adjust if class names differ on the
installed transformers):

  * `AutoModelForCausalLM.from_pretrained(target_id)` returns a Gemma 4
    causal LM. `target.config` has `hidden_size`, `vocab_size`.
  * `AutoModelForCausalLM.from_pretrained(assistant_id)` returns the
    assistant. The assistant exposes a sparse LM head as a submodule named
    something like `masked_embedder` / `masked_embedding` — we discover it
    by walking modules and matching on class name containing "Masked"
    AND attribute presence (`centroids` and `token_ordering`).
  * The masked embedder, given hidden state `h: [B, L, d]`, internally:
      1. Computes centroid logits  C @ h.T          [B, L, num_centroids]
      2. Selects top-`centroid_intermediate_top_k`  [B, L, top_k]
      3. Gathers candidate token ids via `token_ordering`
      4. Computes exact logits inside the candidate set
      5. Scatters into a full [B, L, vocab_size] tensor (unselected = mask
         value, per gemma_idea.txt §6).
  * We install a forward-hook on the masked embedder that records, for
    each call, the per-position selected centroid ids. The corresponding
    candidate token ids are reconstructed via `token_ordering`.

If the discovery fails (no module matches), `find_masked_embedder` raises
with a clear message listing top-level submodule class names, so the user
can patch the matcher quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GemmaPair:
    target: nn.Module
    assistant: nn.Module
    tokenizer: object
    masked_embedder: nn.Module
    num_centroids: int
    centroid_top_k: int
    tokens_per_centroid: int
    vocab_size: int
    hidden_size_target: int
    hidden_size_assistant: int


def load_pair(
    target_id: str,
    assistant_id: str,
    dtype: torch.dtype,
    device: str,
) -> GemmaPair:
    logger.info(f"Loading target: {target_id}")
    target = AutoModelForCausalLM.from_pretrained(
        target_id, dtype=dtype, device_map=device
    ).eval()
    logger.info(f"Loading assistant: {assistant_id}")
    assistant = AutoModelForCausalLM.from_pretrained(
        assistant_id, dtype=dtype, device_map=device
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_id)

    masked = find_masked_embedder(assistant)
    num_centroids = int(masked.centroids.shape[0])
    token_ordering = _get_token_ordering(masked)
    tokens_per_centroid = int(token_ordering.shape[1])
    vocab_size = num_centroids * tokens_per_centroid

    cfg = getattr(assistant, "config", None)
    centroid_top_k = (
        getattr(cfg, "centroid_intermediate_top_k", None)
        or getattr(masked, "centroid_top_k", None)
        or getattr(masked, "top_k", None)
        or 32
    )

    pair = GemmaPair(
        target=target,
        assistant=assistant,
        tokenizer=tokenizer,
        masked_embedder=masked,
        num_centroids=num_centroids,
        centroid_top_k=int(centroid_top_k),
        tokens_per_centroid=tokens_per_centroid,
        vocab_size=int(getattr(target.config, "vocab_size", vocab_size)),
        hidden_size_target=int(target.config.hidden_size),
        hidden_size_assistant=int(assistant.config.hidden_size),
    )
    logger.info(
        f"Pair ready. target_d={pair.hidden_size_target} "
        f"assistant_d={pair.hidden_size_assistant} V={pair.vocab_size} "
        f"num_centroids={pair.num_centroids} top_k={pair.centroid_top_k} "
        f"tokens_per_centroid={pair.tokens_per_centroid}"
    )
    return pair


def find_masked_embedder(model: nn.Module) -> nn.Module:
    """Locate the assistant's sparse / centroid-masked LM head.

    Strategy: walk all submodules; pick the first whose class name contains
    "Masked" and which exposes both `centroids` and a token_ordering buffer
    (`token_ordering` or `ordered_token_ids`). Fall back to attribute-only
    match if the class-name heuristic misses.
    """
    candidates: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        has_centroids = hasattr(mod, "centroids")
        has_ordering = hasattr(mod, "token_ordering") or hasattr(mod, "ordered_token_ids")
        if has_centroids and has_ordering:
            candidates.append((name, mod))
            if "Masked" in cls or "Centroid" in cls:
                logger.info(f"Found masked embedder: {name} ({cls})")
                return mod

    if candidates:
        name, mod = candidates[0]
        logger.warning(
            f"Falling back on attribute-only match for masked embedder: "
            f"{name} ({type(mod).__name__})"
        )
        return mod

    sample = sorted({type(m).__name__ for m in model.modules()})[:30]
    raise RuntimeError(
        "Could not find a masked-embedder submodule on the assistant. "
        f"Top-level class names sampled: {sample}. "
        "Patch find_masked_embedder() in gmtp/gemma_io.py to match the "
        "actual class / attribute names."
    )


def _get_token_ordering(masked: nn.Module) -> torch.Tensor:
    if hasattr(masked, "token_ordering"):
        t = masked.token_ordering
    elif hasattr(masked, "ordered_token_ids"):
        t = masked.ordered_token_ids
    else:
        raise RuntimeError("masked embedder has no token_ordering buffer")
    if t.dim() == 1:
        # Flat [V] permutation; reshape to [num_centroids, tokens_per_centroid].
        n_c = int(masked.centroids.shape[0])
        t = t.view(n_c, -1)
    return t


# ---------------------------------------------------------------------------
# Cluster-mask capture
# ---------------------------------------------------------------------------


@dataclass
class ClusterCapture:
    """Records, per masked-embedder call, the selected centroid ids.

    `events` is a list of [B, L, top_k] long tensors on CPU. `clear()`
    resets between prompts.
    """

    centroids: torch.Tensor                      # [num_centroids, d_assist]
    token_ordering: torch.Tensor                 # [num_centroids, tokens_per_centroid]
    centroid_top_k: int
    events: list[torch.Tensor] = field(default_factory=list)
    _handle: Optional[object] = None

    def clear(self) -> None:
        self.events.clear()

    def candidate_token_ids(self, event_idx: int) -> torch.Tensor:
        """Materialize the per-position candidate token id set for one event.

        Returns: [B, L, top_k * tokens_per_centroid] long tensor on the same
        device as token_ordering.
        """
        cluster_ids = self.events[event_idx].to(self.token_ordering.device)  # [B, L, top_k]
        candidates = self.token_ordering[cluster_ids]  # [B, L, top_k, t_per_c]
        B, L, k, tpc = candidates.shape
        return candidates.reshape(B, L, k * tpc)


def install_cluster_capture(masked: nn.Module) -> ClusterCapture:
    """Install a forward pre-hook that captures top-k centroid ids per call.

    The pre-hook re-computes centroid logits from the input hidden state
    (cheap: 256 × 2048 matmul) and stores top-k indices. We do not rely on
    the embedder's internal tensor names, only its public `centroids`
    attribute and the captured input.
    """
    centroids = masked.centroids
    token_ordering = _get_token_ordering(masked)
    top_k = int(
        getattr(masked, "centroid_top_k", None)
        or getattr(masked, "top_k", None)
        or 32
    )
    cap = ClusterCapture(
        centroids=centroids,
        token_ordering=token_ordering,
        centroid_top_k=top_k,
    )

    def pre_hook(_module, args, kwargs):
        if args:
            h = args[0]
        else:
            h = kwargs.get("hidden_states") or kwargs.get("input")
        if h is None or not torch.is_tensor(h):
            return
        # h: [B, L, d_assist]. Compute centroid logits and top-k indices.
        with torch.no_grad():
            scores = torch.matmul(h.float(), centroids.float().T)  # [B, L, num_c]
            top = scores.topk(top_k, dim=-1).indices            # [B, L, top_k]
        cap.events.append(top.detach().cpu())

    cap._handle = masked.register_forward_pre_hook(pre_hook, with_kwargs=True)
    logger.info(
        f"Installed cluster-capture pre-hook on masked embedder "
        f"(top_k={top_k}, num_centroids={centroids.shape[0]})"
    )
    return cap


def remove_capture(cap: ClusterCapture) -> None:
    if cap._handle is not None:
        cap._handle.remove()
        cap._handle = None


# ---------------------------------------------------------------------------
# Forward-call counters
# ---------------------------------------------------------------------------


@dataclass
class CallCounter:
    target_calls: int = 0
    assistant_calls: int = 0
    _t_handle: Optional[object] = None
    _a_handle: Optional[object] = None

    def reset(self) -> None:
        self.target_calls = 0
        self.assistant_calls = 0


def install_call_counters(target: nn.Module, assistant: nn.Module) -> CallCounter:
    cc = CallCounter()

    def t_hook(_m, _a, _o):
        cc.target_calls += 1

    def a_hook(_m, _a, _o):
        cc.assistant_calls += 1

    cc._t_handle = target.register_forward_hook(t_hook)
    cc._a_handle = assistant.register_forward_hook(a_hook)
    return cc


def remove_counters(cc: CallCounter) -> None:
    if cc._t_handle is not None:
        cc._t_handle.remove()
    if cc._a_handle is not None:
        cc._a_handle.remove()
    cc._t_handle = cc._a_handle = None
