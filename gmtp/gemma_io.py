"""Gemma 4 target + assistant loading and centroid-system access.

For Phase G (cluster-MTPC), we only NEED the target — the head is bolted
on the target's hidden states directly. The assistant is preserved as
an *operational baseline* for G2 (compare our head's μ_acc against the
released drafter's). To use the assistant outside HF's generate(), it
must be called with target activations and shared_kv_states — that
recipe is documented in `invoke_assistant_with_target_conditioning`.

What this module exposes:
  load_target(...)                      — frozen Gemma 4 target only
  load_pair(...)                        — target + assistant + tokenizer
  find_masked_embedder(model)           — discover the sparse LM head
  get_token_ordering(masked)            — [num_centroids, tokens_per_centroid]
  get_centroid_weight_and_bias(masked)  — [num_centroids, d_assist] (+ bias)
  install_call_counters(target, assist) — forward-call counters (G2)
  invoke_assistant_with_target_conditioning(pair, full_ids)
                                        — single-step assistant invocation
                                          for offline analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class GemmaTarget:
    model: nn.Module
    tokenizer: object
    vocab_size: int
    hidden_size: int


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


def _load_model(model_id: str, dtype: torch.dtype, device: str) -> nn.Module:
    """Load Gemma-family checkpoints across HF auto classes.

    Newer Gemma 4 checkpoints may register under ImageTextToText auto class
    even for text-only prompting. Try CausalLM first, then fall back.
    """
    kwargs = {
        "dtype": dtype,
        "device_map": device,
        "attn_implementation": "sdpa",
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e_causal:
        logger.warning(
            f"CausalLM load failed for {model_id}; trying ImageTextToText. "
            f"Cause: {type(e_causal).__name__}: {e_causal}"
        )
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def load_target(target_id: str, dtype: torch.dtype, device: str) -> GemmaTarget:
    """Load only the target. Sufficient for G1 head-only training."""
    logger.info(f"Loading target: {target_id}")
    model = _load_model(target_id, dtype=dtype, device=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    H = _hidden_size_from_config(model.config)
    V = int(getattr(model.config, "vocab_size", 0))
    logger.info(f"Target loaded: hidden_size={H} vocab_size={V}")
    return GemmaTarget(model=model, tokenizer=tokenizer, vocab_size=V, hidden_size=H)


def load_pair(
    target_id: str,
    assistant_id: str,
    dtype: torch.dtype,
    device: str,
) -> GemmaPair:
    """Load target + assistant (for G2+ operational baseline comparison)."""
    logger.info(f"Loading target: {target_id}")
    target = _load_model(target_id, dtype=dtype, device=device).eval()
    logger.info(f"Loading assistant: {assistant_id}")
    assistant = _load_model(assistant_id, dtype=dtype, device=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_id)

    masked = find_masked_embedder(assistant)
    num_centroids = _num_centroids(masked)
    token_ordering = get_token_ordering(masked)
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
        hidden_size_target=_hidden_size_from_config(target.config),
        hidden_size_assistant=_hidden_size_from_config(assistant.config),
    )
    logger.info(
        f"Pair ready. target_d={pair.hidden_size_target} "
        f"assistant_d={pair.hidden_size_assistant} V={pair.vocab_size} "
        f"num_centroids={pair.num_centroids} top_k={pair.centroid_top_k} "
        f"tokens_per_centroid={pair.tokens_per_centroid}"
    )
    return pair


# ---------------------------------------------------------------------------
# Centroid-system discovery
# ---------------------------------------------------------------------------


def find_masked_embedder(model: nn.Module) -> nn.Module:
    """Locate the assistant's sparse centroid-masked LM head."""
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
        f"Top-level class names sampled: {sample}."
    )


def get_token_ordering(masked: nn.Module) -> torch.Tensor:
    """Return [num_centroids, tokens_per_centroid] long tensor."""
    if hasattr(masked, "token_ordering"):
        t = masked.token_ordering
    elif hasattr(masked, "ordered_token_ids"):
        t = masked.ordered_token_ids
    else:
        raise RuntimeError("masked embedder has no token_ordering buffer")
    if t.dim() == 1:
        n_c = _num_centroids(masked)
        t = t.view(n_c, -1)
    return t


def get_centroid_weight_and_bias(
    masked: nn.Module,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return centroid projection weight [num_centroids, d] and optional bias.

    `masked.centroids` may be a raw Tensor or an nn.Linear depending on
    transformers version.
    """
    c = getattr(masked, "centroids", None)
    if c is None:
        raise RuntimeError("masked embedder has no `centroids` attribute")
    if torch.is_tensor(c):
        return c, None
    if isinstance(c, nn.Linear):
        return c.weight, c.bias
    if hasattr(c, "weight") and torch.is_tensor(c.weight):
        b = c.bias if hasattr(c, "bias") and torch.is_tensor(c.bias) else None
        return c.weight, b
    raise RuntimeError(f"Unsupported `centroids` type: {type(c).__name__}")


def _num_centroids(masked: nn.Module) -> int:
    w, _ = get_centroid_weight_and_bias(masked)
    return int(w.shape[0])


def _hidden_size_from_config(cfg) -> int:
    if hasattr(cfg, "hidden_size") and getattr(cfg, "hidden_size") is not None:
        return int(cfg.hidden_size)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return int(text_cfg.hidden_size)
    raise RuntimeError(f"Unable to infer hidden size from config {type(cfg).__name__}")


# ---------------------------------------------------------------------------
# Forward-call counters (G2 operational baseline)
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


# ---------------------------------------------------------------------------
# Assistant invocation with target conditioning
# (For G2 operational-baseline analysis. Not used in G1.)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def invoke_assistant_with_target_conditioning(
    pair: GemmaPair,
    input_ids: torch.Tensor,
) -> object:
    """Single-step assistant forward with proper target conditioning.

    The Gemma 4 assistant expects:
      * inputs_embeds = concat(token_embeds(input_ids), target_last_hidden)
        along the feature dim (not seq dim).
      * shared_kv_states from a target forward with
        `return_shared_kv_states=True`.
    Standalone `assistant(input_ids=...)` is NOT supported.

    Returns the assistant's forward output (logits over vocab).
    """
    target_out = pair.target(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_shared_kv_states=True,
    )
    token_embeds = pair.target.get_input_embeddings()(input_ids)
    last_hidden = target_out.hidden_states[-1]
    inputs_embeds = torch.cat([token_embeds, last_hidden], dim=-1)
    return pair.assistant(
        inputs_embeds=inputs_embeds,
        shared_kv_states=target_out.shared_kv_states,
        use_cache=False,
    )
