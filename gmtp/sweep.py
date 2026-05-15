"""E1 — top_k sweep over centroid_intermediate_top_k.

Three pieces:

  E1a — recall sweep. Capture top-K_max centroid indices per (prompt,
        position) ONCE; derive recall@k for any k ≤ K_max in post-
        processing (top-k is a prefix of top-K_max). Total assistant work
        is identical to E0a — the sweep is free.

  E1b — LM-head latency microbench. Time the masked embedder forward in
        isolation at each top_k cell (monkey-patched). Cheap.

  E1c — end-to-end timing per top_k. target.generate(...,
        assistant_model=...) at each cell, with cluster top_k monkey-
        patched on the assistant. The expensive piece in proportion to
        cells × prompts.

ASSUMPTION on monkey-patching: the masked embedder's selected-cluster
count is read from one of {centroid_top_k, top_k} attributes at each
forward pass. If it is baked at init or compile time, latency / E2E
will return identical numbers across cells — sanity-checked at the end
of E1b and the user is warned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from .gemma_io import GemmaPair, _get_token_ordering, install_call_counters, remove_counters
from .timing import _maybe_apply_chat_template
from .utils import cuda_time_ms, summary_stats


# ---------------------------------------------------------------------------
# E1a — top-K_max cluster capture
# ---------------------------------------------------------------------------


@dataclass
class TopKMaxClusterCapture:
    """Records per-call top-K_max centroid ids on the masked embedder."""

    masked: nn.Module
    K_max: int
    token_ordering: torch.Tensor
    events: list[torch.Tensor] = field(default_factory=list)
    _handle: Optional[object] = None

    def install(self) -> None:
        centroids = self.masked.centroids
        K_max = self.K_max

        def hook(_module, args, kwargs):
            if args:
                h = args[0]
            else:
                h = kwargs.get("hidden_states") or kwargs.get("input")
            if h is None or not torch.is_tensor(h):
                return
            with torch.no_grad():
                scores = torch.matmul(h.float(), centroids.float().T)  # [B, L, num_c]
                top = scores.topk(K_max, dim=-1).indices               # [B, L, K_max]
            self.events.append(top.detach().cpu())

        self._handle = self.masked.register_forward_pre_hook(hook, with_kwargs=True)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self) -> None:
        self.events.clear()


@dataclass
class PositionTrace:
    target_token: int
    clusters_top_kmax: list[int]


@dataclass
class PromptTrace:
    prompt_idx: int
    source: str
    positions: list[PositionTrace]


@torch.inference_mode()
def _generate_continuation(pair, input_ids, n_new, do_sample, temperature):
    out = pair.target.generate(
        input_ids,
        max_new_tokens=n_new,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=0.95 if do_sample else 1.0,
        top_k=64 if do_sample else 0,
        use_cache=True,
        pad_token_id=pair.tokenizer.eos_token_id,
    )
    return out[0, input_ids.shape[1]:]


@torch.inference_mode()
def measure_recall_topkmax(
    pair: GemmaPair,
    prompts,
    n_positions: int,
    do_sample: bool,
    temperature: float,
    K_max: int,
    device: str,
    label: str,
) -> list[PromptTrace]:
    """Run target generation + per-position assistant capture for top-K_max."""
    logger.info(
        f"[{label}] top-K_max recall capture over {len(prompts)} prompts, "
        f"n_positions={n_positions}, K_max={K_max}, sample={do_sample}, T={temperature}"
    )
    cap = TopKMaxClusterCapture(
        masked=pair.masked_embedder,
        K_max=K_max,
        token_ordering=_get_token_ordering(pair.masked_embedder),
    )
    cap.install()
    traces: list[PromptTrace] = []

    try:
        for p_idx, p in enumerate(tqdm(prompts, desc=f"E1a:{label}")):
            text = _maybe_apply_chat_template(pair.tokenizer, p.text)
            enc = pair.tokenizer(text, return_tensors="pt").to(device)
            new = _generate_continuation(pair, enc.input_ids, n_positions, do_sample, temperature)

            positions: list[PositionTrace] = []
            for i in range(new.shape[0]):
                full_ids = torch.cat([enc.input_ids, new[:i].unsqueeze(0)], dim=1)
                cap.clear()
                _ = pair.assistant(input_ids=full_ids, use_cache=False)
                if not cap.events:
                    raise RuntimeError(
                        "TopKMaxClusterCapture: no events fired on assistant forward."
                    )
                last = cap.events[-1]                  # [B, L, K_max]
                clusters = last[0, -1].tolist()
                positions.append(
                    PositionTrace(target_token=int(new[i]), clusters_top_kmax=clusters)
                )
            traces.append(PromptTrace(prompt_idx=p_idx, source=p.source, positions=positions))
    finally:
        cap.remove()
    return traces


def derive_recall_at_k(
    traces: list[PromptTrace],
    k_values: list[int],
    token_ordering: torch.Tensor,
) -> dict:
    """Compute recall@k for each k in k_values from captured top-K_max traces.

    Returns: dict[k -> {overall_recall, n, by_source}].
    """
    # token_ordering is on the model device; for set-membership we move to CPU.
    to_cpu = token_ordering.detach().cpu()
    out: dict[int, dict] = {}
    for k in k_values:
        hits_total = 0
        n_total = 0
        by_source: dict[str, dict] = {}
        for t in traces:
            src = by_source.setdefault(t.source, {"hits": 0, "n": 0, "prompts": 0})
            src["prompts"] += 1
            for pos in t.positions:
                cluster_ids = torch.tensor(pos.clusters_top_kmax[:k], dtype=torch.long)
                cand = to_cpu[cluster_ids].flatten()  # [k * tpc]
                hit = int((cand == pos.target_token).any().item())
                hits_total += hit
                n_total += 1
                src["hits"] += hit
                src["n"] += 1
        for s in by_source.values():
            s["recall"] = (s["hits"] / s["n"]) if s["n"] else None
        out[k] = {
            "overall_recall": (hits_total / n_total) if n_total else None,
            "n": n_total,
            "n_hits": hits_total,
            "by_source": by_source,
        }
    return out


# ---------------------------------------------------------------------------
# E1b — LM-head latency microbench
# ---------------------------------------------------------------------------


def _set_top_k(masked: nn.Module, k: int) -> Optional[str]:
    """Try to set the masked embedder's top_k attribute. Returns the
    attribute name set, or None if neither candidate exists."""
    for attr in ("centroid_top_k", "top_k", "centroid_intermediate_top_k"):
        if hasattr(masked, attr):
            setattr(masked, attr, k)
            return attr
    return None


def _read_top_k(masked: nn.Module) -> Optional[int]:
    for attr in ("centroid_top_k", "top_k", "centroid_intermediate_top_k"):
        if hasattr(masked, attr):
            v = getattr(masked, attr)
            if isinstance(v, int):
                return v
    return None


@torch.inference_mode()
def microbench_masked_embedder(
    pair: GemmaPair,
    top_k_values: list[int],
    ctx_len: int,
    iters: int,
    warmup: int,
    device: str,
    dtype: torch.dtype,
) -> dict[int, dict]:
    """Time the masked embedder forward in isolation at each top_k cell.

    Uses a fixed random hidden-state input of shape [1, ctx_len, d_assist].
    Monkey-patches `masked.{centroid_top_k|top_k|centroid_intermediate_top_k}`
    before each cell. Restores original at the end.
    """
    masked = pair.masked_embedder
    h = torch.randn(
        1, ctx_len, pair.hidden_size_assistant, dtype=dtype, device=device
    )
    orig = _read_top_k(masked)
    results: dict[int, dict] = {}
    set_attr_name: Optional[str] = None

    for k in top_k_values:
        attr = _set_top_k(masked, k)
        if attr is None:
            logger.warning(
                f"[E1b] cannot set top_k attribute on masked embedder; "
                f"skipping latency for k={k}"
            )
            continue
        set_attr_name = attr

        def fn() -> None:
            _ = masked(h)

        try:
            times = cuda_time_ms(fn, iters=iters, warmup=warmup)
            stats = summary_stats(times)
            results[k] = stats
            logger.info(
                f"[E1b] top_k={k:>3}  median={stats['median_ms']:.3f}ms  "
                f"mean={stats['mean_ms']:.3f}±{stats['stdev_ms']:.3f}ms"
            )
        except Exception as e:
            logger.warning(f"[E1b] top_k={k}: forward failed ({e!r}); skipping")
            continue

    if orig is not None and set_attr_name is not None:
        setattr(masked, set_attr_name, orig)

    if len(results) >= 2:
        ms = [r["median_ms"] for r in results.values()]
        spread = (max(ms) - min(ms)) / max(min(ms), 1e-6)
        if spread < 0.05:
            logger.warning(
                f"[E1b] latency spread across top_k cells is only {spread:.1%}. "
                "The monkey-patched attribute may not control the actual kernel; "
                "verify before trusting E1c throughput numbers."
            )
    return results


# ---------------------------------------------------------------------------
# E1c — end-to-end throughput per top_k cell
# ---------------------------------------------------------------------------


@dataclass
class CellTimingTrace:
    top_k: int
    prompt_idx: int
    source: str
    new_tokens: int
    target_calls: int
    assistant_calls: int
    wall_seconds: float


@torch.inference_mode()
def time_e2e_per_topk(
    pair: GemmaPair,
    prompts,
    top_k_values: list[int],
    n_new_tokens: int,
    do_sample: bool,
    temperature: float,
    device: str,
) -> dict[int, list[CellTimingTrace]]:
    """For each top_k cell, run target.generate(..., assistant_model=...)
    over the given prompts and record per-prompt (new_tokens, target_calls,
    assistant_calls, wall_seconds).
    """
    masked = pair.masked_embedder
    orig = _read_top_k(masked)
    pad_id = pair.tokenizer.eos_token_id
    results: dict[int, list[CellTimingTrace]] = {}

    cc = install_call_counters(pair.target, pair.assistant)

    try:
        # Single warm-up at the first cell to avoid lazy-init contamination.
        if prompts and top_k_values:
            _set_top_k(masked, top_k_values[0])
            warm = pair.tokenizer(
                _maybe_apply_chat_template(pair.tokenizer, prompts[0].text),
                return_tensors="pt",
            ).to(device)
            _ = pair.target.generate(
                warm.input_ids,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                assistant_model=pair.assistant,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        for k in top_k_values:
            if _set_top_k(masked, k) is None:
                logger.warning(f"[E1c] cannot set top_k={k}; skipping cell")
                continue
            logger.info(f"[E1c] cell top_k={k} over {len(prompts)} prompts")
            traces: list[CellTimingTrace] = []

            for p_idx, p in enumerate(tqdm(prompts, desc=f"E1c:k={k}")):
                text = _maybe_apply_chat_template(pair.tokenizer, p.text)
                enc = pair.tokenizer(text, return_tensors="pt").to(device)
                prompt_len = enc.input_ids.shape[1]
                cc.reset()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = pair.target.generate(
                    enc.input_ids,
                    max_new_tokens=n_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    top_p=0.95 if do_sample else 1.0,
                    top_k=64 if do_sample else 0,
                    use_cache=True,
                    pad_token_id=pad_id,
                    assistant_model=pair.assistant,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                new = int(out.shape[1] - prompt_len)
                traces.append(
                    CellTimingTrace(
                        top_k=k,
                        prompt_idx=p_idx,
                        source=p.source,
                        new_tokens=new,
                        target_calls=cc.target_calls,
                        assistant_calls=cc.assistant_calls,
                        wall_seconds=elapsed,
                    )
                )
            results[k] = traces
    finally:
        remove_counters(cc)
        if orig is not None:
            _set_top_k(masked, orig)

    return results


def aggregate_e2e_cell(traces: list[CellTimingTrace]) -> dict:
    if not traces:
        return {"n_prompts": 0}
    new = sum(t.new_tokens for t in traces)
    tc = sum(t.target_calls for t in traces)
    ac = sum(t.assistant_calls for t in traces)
    secs = sum(t.wall_seconds for t in traces)
    return {
        "n_prompts": len(traces),
        "new_tokens_total": new,
        "target_calls_total": tc,
        "assistant_calls_total": ac,
        "wall_seconds_total": secs,
        "tok_per_sec": (new / secs) if secs else None,
        "new_tokens_per_target_call": (new / tc) if tc else None,
        "target_calls_per_new_token": (tc / new) if new else None,
        "assistant_calls_per_new_token": (ac / new) if new else None,
    }
