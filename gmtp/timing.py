"""E0b — end-to-end speculative-decoding timing.

For each prompt, run target.generate(..., assistant_model=assistant) with
forward-call counters on both models. Reports:

  * wall-clock tok/s (new tokens / total time)
  * target_calls / new_token (lower is better)
  * accepted_tokens_per_target_call = (new_tokens - target_calls) /
    target_calls + 1 cleanly accounts for the one bonus token target
    emits per accepted block; we report the simpler ratio
    new_tokens / target_calls and let the analysis stage interpret.
  * assistant_calls / new_token

This is a black-box wrapper around HF's assisted decoding. No claims
about per-step latency split — that requires hooks the harness can't
cleanly install around `generate`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from loguru import logger
from tqdm import tqdm

from .gemma_io import GemmaPair, install_call_counters, remove_counters


@dataclass
class TimingTrace:
    prompt_idx: int
    source: str
    new_tokens: int
    target_calls: int
    assistant_calls: int
    wall_seconds: float


def _maybe_apply_chat_template(tokenizer, text: str) -> str:
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
def measure_timing(
    pair: GemmaPair,
    prompts,
    n_new_tokens: int,
    do_sample: bool,
    temperature: float,
    device: str,
    label: str,
) -> list[TimingTrace]:
    logger.info(
        f"[{label}] end-to-end timing over {len(prompts)} prompts, "
        f"max_new={n_new_tokens}, sample={do_sample}, T={temperature}"
    )
    cc = install_call_counters(pair.target, pair.assistant)
    traces: list[TimingTrace] = []
    pad_id = pair.tokenizer.eos_token_id

    try:
        # Warm one prompt so first-call lazy init does not poison the timing.
        if prompts:
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
            torch.cuda.synchronize() if torch.cuda.is_available() else None

        for p_idx, p in enumerate(tqdm(prompts, desc=f"timing:{label}")):
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
                TimingTrace(
                    prompt_idx=p_idx,
                    source=p.source,
                    new_tokens=new,
                    target_calls=cc.target_calls,
                    assistant_calls=cc.assistant_calls,
                    wall_seconds=elapsed,
                )
            )
    finally:
        remove_counters(cc)

    return traces


def aggregate_timing(traces: list[TimingTrace]) -> dict:
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
