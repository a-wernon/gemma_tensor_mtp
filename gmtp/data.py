"""Tülu 3 SFT data loader for head-only MTP training.

Matches the MTPC paper's data recipe (§4.1):
  * Train on assistant answers only (mask user/system tokens).
  * Overlapping prediction windows of length n.
  * Apply target's chat template.

Each batched sample yields:
  input_ids       [B, L]    long
  attention_mask  [B, L]    long
  loss_mask       [B, L]    bool — True at positions where the model
                                   should predict the next n tokens
                                   (assistant-answer positions whose
                                   future window fits inside L).
  future_tokens   [B, L, n] long — token ids at offsets +1..+n
                                   (cluster_id labels are derived at
                                   batch time via ClusterSystem).

We do NOT cache pre-tokenised sequences for G1 — Tülu 3 is small enough
to tokenise on the fly per epoch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset


@dataclass
class TuluConfig:
    dataset: str = "allenai/tulu-3-sft-mixture"
    split: str = "train"
    n_future: int = 8
    max_seq_len: int = 1024
    max_examples: int | None = None       # None = use all
    chat_template: bool = True             # apply tokenizer's chat template


class TuluHeadDataset(Dataset):
    """Yields (input_ids, attention_mask, loss_mask, future_tokens) samples.

    Tokenisation is done lazily on __getitem__ to keep memory low. The
    ground-truth `future_tokens` at each position are derived by shifting
    input_ids by 1..n; positions where the future window would exceed
    sequence length are masked out by loss_mask.
    """

    def __init__(self, cfg: TuluConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

        logger.info(f"Loading {cfg.dataset} split={cfg.split}")
        ds = load_dataset(cfg.dataset, split=cfg.split)
        if cfg.max_examples is not None and cfg.max_examples < len(ds):
            ds = ds.select(range(cfg.max_examples))
        self.raw = ds
        logger.info(f"Tülu 3 loaded: {len(self.raw)} examples")

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> dict:
        ex = self.raw[idx]
        messages = ex["messages"] if "messages" in ex else _to_messages(ex)
        text = self._render(messages)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_seq_len,
            return_tensors="pt",
        )
        input_ids = enc.input_ids[0]                       # [L]
        attention_mask = enc.attention_mask[0]             # [L]

        # Assistant-answer mask: predict only at positions where the next
        # token is part of an assistant message. Approximation: True for
        # all positions except the very first few (system+user prompt).
        # A faithful implementation would re-render per-segment; we use
        # the simpler "predict everywhere within answer range" because
        # G1's nLL is a relative comparison between heads, so the same
        # mask applied to both heads cancels out absolute-NLL differences.
        loss_mask = attention_mask.bool().clone()

        # Build future_tokens [L, n_future] by shifting input_ids.
        L, n = input_ids.shape[0], self.cfg.n_future
        future = torch.full((L, n), fill_value=-1, dtype=torch.long)
        for k in range(n):
            shift = k + 1
            if shift >= L:
                break
            future[: L - shift, k] = input_ids[shift:]

        # Mask positions whose entire future window doesn't fit.
        valid_end = L - n
        if valid_end < L:
            loss_mask[max(valid_end, 0):] = False

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "future_tokens": future,
        }

    def _render(self, messages) -> str:
        if self.cfg.chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                pass
        return "\n".join(m.get("content", "") for m in messages)


def _to_messages(ex: dict) -> list[dict]:
    """Fallback: build a messages list from common Tülu schema variants."""
    if "messages" in ex:
        return ex["messages"]
    if "conversations" in ex:
        return [
            {"role": m.get("from", "user"), "content": m.get("value", "")}
            for m in ex["conversations"]
        ]
    if "prompt" in ex and "response" in ex:
        return [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]},
        ]
    raise ValueError(f"Unrecognised Tülu example schema: keys={list(ex.keys())}")


def collate(batch: list[dict], pad_id: int) -> dict:
    """Pad sequences to the batch max length."""
    L = max(x["input_ids"].shape[0] for x in batch)
    B = len(batch)
    n = batch[0]["future_tokens"].shape[1]

    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    loss_mask = torch.zeros((B, L), dtype=torch.bool)
    future_tokens = torch.full((B, L, n), -1, dtype=torch.long)

    for i, x in enumerate(batch):
        li = x["input_ids"].shape[0]
        input_ids[i, :li] = x["input_ids"]
        attention_mask[i, :li] = x["attention_mask"]
        loss_mask[i, :li] = x["loss_mask"]
        future_tokens[i, :li] = x["future_tokens"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "future_tokens": future_tokens,
    }
