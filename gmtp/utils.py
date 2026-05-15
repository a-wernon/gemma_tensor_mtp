"""Run-dir, logging, and timing helpers. Adapted from r2_latency/r2lat/utils.py."""

from __future__ import annotations

import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger


def env_threads() -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_str(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def make_run_dir(root: str, name: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = Path(root) / f"{name}_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_logger(run_dir: Path) -> None:
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO")
    logger.add(str(run_dir / "run.log"), level="DEBUG", rotation="10 MB")
    logger.info(f"Logging to {run_dir / 'run.log'}")


def cuda_time_ms(fn, iters: int, warmup: int = 5) -> list[float]:
    assert torch.cuda.is_available(), "cuda_time_ms requires CUDA"
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def summary_stats(times: list[float]) -> dict:
    if not times:
        return {"n_samples": 0}
    quantiles = statistics.quantiles(times, n=20) if len(times) >= 20 else None
    return {
        "n_samples": len(times),
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "p95_ms": quantiles[18] if quantiles is not None else max(times),
    }
