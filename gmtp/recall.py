"""E0a — cluster recall.

For each prompt:
  1. Generate a ground-truth continuation X of length N with the target
     (greedy or sampled).
  2. For each position i in X, run the assistant on (prompt + X[:i]) so
     the masked-embedder pre-hook captures the per-position cluster mask
     for the *first draft step*. Recall_i = 1[X[i] in candidate set].

This isolates the "is the target-preferred token in the assistant's
top-k cluster mask" diagnostic, with no spec-decoding feedback loop.

Why first-draft only: it is the most informative position. Real spec
decoding then drafts further tokens autoregressively, where the assistant
no longer has the latest target hidden state — so first-draft recall is
an upper bound on per-step recall in actual spec decoding. If even
first-draft recall is high, downstream positions only get harder.

FIRST-RUN GOTCHA — the isolated assistant call. The Gemma 4 assistant
documented usage is target.generate(..., assistant_model=assistant); HF
internally feeds it target hidden states + KV cache. We assume the
assistant also accepts a bare pair.assistant(input_ids=...) call — if
that errors (cross-attn requires target activations), the fix is to
instead capture cluster events DURING a target.generate(...,
assistant_model=...) run and align them to verified positions. That is
strictly more invasive; we ship the simple path first and revise if it
breaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger
from tqdm import tqdm

from .gemma_io import ClusterCapture, GemmaPair, install_cluster_capture, remove_capture


@dataclass
class RecallTrace:
    prompt_idx: int
    source: str
    n_positions: int
    hits_greedy: int
    hits_sampled: int
    target_tokens_greedy: list[int]
    target_tokens_sampled: list[int]


def _maybe_apply_chat_template(tokenizer, text: str) -> str:
    """Apply the model's chat template if available; otherwise return text."""
    tmpl = getattr(tokenizer, "apply_chat_template", None)
    if tmpl is None:
        return text
    try:
        return tmpl(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return text


@torch.inference_mode()
def _generate_continuation(
    pair: GemmaPair,
    input_ids: torch.Tensor,
    n_new: int,
    do_sample: bool,
    temperature: float,
) -> torch.Tensor:
    """Plain target-only generation. Returns the new tokens [n_new]."""
    out = pair.target.generate(
        input_ids,
        max_new_tokens=n_new,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=1.0 if not do_sample else 0.95,
        top_k=0 if not do_sample else 64,
        use_cache=True,
        pad_token_id=pair.tokenizer.eos_token_id,
    )
    new = out[0, input_ids.shape[1]:]
    return new


@torch.inference_mode()
def _candidate_token_set_at_first_draft(
    pair: GemmaPair,
    cap: ClusterCapture,
    full_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the assistant once on `full_ids` (= prompt + revealed prefix) and
    return the set of candidate token ids at the LAST position — this
    matches "what would the assistant propose as draft token #1 given
    this prefix".

    Returns: long tensor of unique candidate token ids.
    """
    cap.clear()
    _ = pair.assistant(input_ids=full_ids, use_cache=False)
    if not cap.events:
        raise RuntimeError(
            "Cluster-capture pre-hook produced no events on the assistant "
            "forward. Either the masked embedder was not invoked, or the "
            "discovery picked the wrong submodule. See gmtp/gemma_io.py."
        )
    # Concatenate candidate ids across all masked-embedder calls in this
    # forward (most assistants invoke it once at the LM-head; if it fires
    # per-layer we union them).
    candidates: list[torch.Tensor] = []
    for i in range(len(cap.events)):
        c = cap.candidate_token_ids(i)   # [B, L, top_k * tpc]
        candidates.append(c[0, -1])
    return torch.unique(torch.cat(candidates))


def measure_recall(
    pair: GemmaPair,
    prompts,
    n_positions: int,
    do_sample: bool,
    temperature: float,
    device: str,
    label: str,
) -> list[RecallTrace]:
    """Returns one RecallTrace per prompt for the given decoding mode."""
    logger.info(
        f"[{label}] cluster recall over {len(prompts)} prompts, "
        f"n_positions={n_positions}, sample={do_sample}, T={temperature}"
    )
    cap = install_cluster_capture(pair.masked_embedder)
    traces: list[RecallTrace] = []

    try:
        for p_idx, p in enumerate(tqdm(prompts, desc=f"recall:{label}")):
            text = _maybe_apply_chat_template(pair.tokenizer, p.text)
            enc = pair.tokenizer(text, return_tensors="pt").to(device)
            prompt_ids = enc.input_ids                               # [1, L0]
            new_tokens = _generate_continuation(
                pair, prompt_ids, n_positions, do_sample, temperature
            )                                                        # [n_positions]

            hits = 0
            for i in range(new_tokens.shape[0]):
                full_ids = torch.cat([prompt_ids, new_tokens[:i].unsqueeze(0)], dim=1)
                cand = _candidate_token_set_at_first_draft(pair, cap, full_ids)
                if (cand == new_tokens[i]).any():
                    hits += 1

            traces.append(
                RecallTrace(
                    prompt_idx=p_idx,
                    source=p.source,
                    n_positions=int(new_tokens.shape[0]),
                    hits_greedy=hits if not do_sample else 0,
                    hits_sampled=hits if do_sample else 0,
                    target_tokens_greedy=new_tokens.tolist() if not do_sample else [],
                    target_tokens_sampled=new_tokens.tolist() if do_sample else [],
                )
            )
    finally:
        remove_capture(cap)

    return traces


def aggregate_recall(traces: list[RecallTrace], mode: str) -> dict:
    """mode in {"greedy", "sampled"}."""
    if not traces:
        return {"recall": None, "n_prompts": 0, "n_positions": 0}
    total_hits = sum(t.hits_greedy if mode == "greedy" else t.hits_sampled for t in traces)
    total_positions = sum(t.n_positions for t in traces)
    by_source: dict[str, dict] = {}
    for t in traces:
        s = by_source.setdefault(t.source, {"hits": 0, "positions": 0, "prompts": 0})
        s["hits"] += t.hits_greedy if mode == "greedy" else t.hits_sampled
        s["positions"] += t.n_positions
        s["prompts"] += 1
    for s in by_source.values():
        s["recall"] = (s["hits"] / s["positions"]) if s["positions"] else None
    return {
        "recall": (total_hits / total_positions) if total_positions else None,
        "n_prompts": len(traces),
        "n_positions": total_positions,
        "n_hits": total_hits,
        "by_source": by_source,
    }
