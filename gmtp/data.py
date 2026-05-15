"""Fixed eval set for E0: ~200 prompts from GSM8K-train + HumanEval.

Builds and caches a deterministic JSONL of prompts so repeat runs use
identical inputs. The tokenizer is applied lazily by the caller (we keep
the cache tokenizer-agnostic so the same prompt set can be reused if we
swap target models later).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from loguru import logger


@dataclass
class Prompt:
    source: str   # "gsm8k" | "humaneval"
    idx: int      # original index in the source split
    text: str     # raw prompt text (no chat template applied)

    def to_dict(self) -> dict:
        return {"source": self.source, "idx": self.idx, "text": self.text}

    @staticmethod
    def from_dict(d: dict) -> "Prompt":
        return Prompt(source=d["source"], idx=d["idx"], text=d["text"])


def _gsm8k_prompts(n: int, seed: int) -> list[Prompt]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    out: list[Prompt] = []
    for i in idxs:
        q = ds[i]["question"].strip()
        if not q:
            continue
        out.append(Prompt(source="gsm8k", idx=int(i), text=q))
        if len(out) >= n:
            break
    return out


def _humaneval_prompts(n: int) -> list[Prompt]:
    # Deterministic order — HumanEval is small (164 tasks); take the first n.
    ds = load_dataset("openai_humaneval", split="test")
    out: list[Prompt] = []
    for i in range(min(n, len(ds))):
        text = ds[i]["prompt"]
        if not text:
            continue
        out.append(Prompt(source="humaneval", idx=int(i), text=text))
    return out


def build_eval_set(
    n_gsm8k: int,
    n_humaneval: int,
    seed: int,
    cache_path: Path,
) -> list[Prompt]:
    """Build (or load) the eval set. Cache is keyed by (counts, seed)."""
    key = f"gsm8k{n_gsm8k}_humaneval{n_humaneval}_seed{seed}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    target = cache_path.parent / f"{cache_path.stem}_{digest}.jsonl"

    if target.exists():
        logger.info(f"Loading cached eval set from {target}")
        prompts = [Prompt.from_dict(json.loads(l)) for l in target.read_text().splitlines()]
        return prompts

    logger.info(f"Building eval set (gsm8k={n_gsm8k}, humaneval={n_humaneval}, seed={seed})")
    prompts: list[Prompt] = []
    prompts.extend(_gsm8k_prompts(n_gsm8k, seed))
    prompts.extend(_humaneval_prompts(n_humaneval))

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        for p in prompts:
            f.write(json.dumps(p.to_dict()) + "\n")
    logger.info(f"Wrote {len(prompts)} prompts to {target}")
    return prompts
