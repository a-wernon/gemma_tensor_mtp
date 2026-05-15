"""E0 entry point — instrumented baseline for Gemma 4 MTP drafter.

Usage:
    python run_e0.py --config configs/e0_baseline.yaml
    python run_e0.py --config configs/e0_baseline.yaml --smoke
    python run_e0.py --config configs/e0_baseline.yaml --skip-timing

Outputs under runs/e0/<run_name>_<ts>/:
    config.json
    run.log
    summary.json
    eval_set.jsonl
    tb/                (TensorBoard scalars, if enabled)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from gmtp.data import build_eval_set
from gmtp.gemma_io import load_pair
from gmtp.recall import aggregate_recall, measure_recall
from gmtp.timing import aggregate_timing, measure_timing
from gmtp.utils import (
    configure_logger,
    dtype_from_str,
    dump_json,
    env_threads,
    load_yaml,
    make_run_dir,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick sanity pass: 8 prompts, n_positions=4, max_new=16. ~minutes.",
    )
    p.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip E0b end-to-end timing (run only cluster recall).",
    )
    return p.parse_args()


def _apply_smoke(cfg: dict) -> None:
    cfg["data"]["n_gsm8k"] = 4
    cfg["data"]["n_humaneval"] = 4
    cfg["recall"]["n_positions"] = 4
    cfg["recall"]["max_prompts"] = 8
    cfg["timing"]["n_new_tokens"] = 16
    cfg["timing"]["max_prompts"] = 4
    cfg["run_name"] = cfg["run_name"] + "_smoke"


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.smoke:
        _apply_smoke(cfg)

    env_threads()
    set_seed(cfg.get("seed", 42))

    run_dir = make_run_dir(cfg["output"]["root"], cfg["run_name"])
    configure_logger(run_dir)
    dump_json(run_dir / "config.json", cfg)

    tb = (
        SummaryWriter(str(run_dir / "tb"))
        if cfg["output"].get("tensorboard", True)
        else None
    )

    device = cfg["device"]
    dtype = dtype_from_str(cfg["dtype"])

    # ---- eval set -----------------------------------------------------------
    cache_path = Path(cfg["data"].get("cache_dir", "cache/eval")) / "eval"
    prompts = build_eval_set(
        n_gsm8k=cfg["data"]["n_gsm8k"],
        n_humaneval=cfg["data"]["n_humaneval"],
        seed=cfg["seed"],
        cache_path=cache_path,
    )
    # Mirror the eval set into the run dir for reproducibility.
    with open(run_dir / "eval_set.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p.to_dict()) + "\n")

    # ---- model pair ---------------------------------------------------------
    pair = load_pair(cfg["target"], cfg["assistant"], dtype, device)

    # ---- E0a — cluster recall -----------------------------------------------
    recall_cfg = cfg["recall"]
    recall_prompts = prompts
    if recall_cfg.get("max_prompts"):
        recall_prompts = prompts[: int(recall_cfg["max_prompts"])]

    summary: dict = {
        "config_resolved": cfg,
        "vocab_size": pair.vocab_size,
        "num_centroids": pair.num_centroids,
        "centroid_top_k": pair.centroid_top_k,
        "tokens_per_centroid": pair.tokens_per_centroid,
    }

    if recall_cfg.get("greedy", True):
        traces = measure_recall(
            pair,
            recall_prompts,
            n_positions=int(recall_cfg["n_positions"]),
            do_sample=False,
            temperature=1.0,
            device=device,
            label="greedy",
        )
        agg = aggregate_recall(traces, mode="greedy")
        summary["recall_greedy"] = agg
        dump_json(run_dir / "traces_recall_greedy.json", [asdict(t) for t in traces])
        if tb:
            tb.add_scalar("recall/greedy/overall", agg["recall"] or 0.0, 0)
            for src, s in agg["by_source"].items():
                tb.add_scalar(f"recall/greedy/{src}", s["recall"] or 0.0, 0)
        logger.info(f"[recall:greedy] overall = {agg['recall']:.4f} "
                    f"({agg['n_hits']}/{agg['n_positions']})")

    if recall_cfg.get("sampled", True):
        traces = measure_recall(
            pair,
            recall_prompts,
            n_positions=int(recall_cfg["n_positions"]),
            do_sample=True,
            temperature=float(recall_cfg.get("temperature", 1.0)),
            device=device,
            label="sampled",
        )
        agg = aggregate_recall(traces, mode="sampled")
        summary["recall_sampled"] = agg
        dump_json(run_dir / "traces_recall_sampled.json", [asdict(t) for t in traces])
        if tb:
            tb.add_scalar("recall/sampled/overall", agg["recall"] or 0.0, 0)
            for src, s in agg["by_source"].items():
                tb.add_scalar(f"recall/sampled/{src}", s["recall"] or 0.0, 0)
        logger.info(f"[recall:sampled] overall = {agg['recall']:.4f} "
                    f"({agg['n_hits']}/{agg['n_positions']})")

    # ---- E0b — end-to-end timing --------------------------------------------
    timing_cfg = cfg.get("timing", {})
    if timing_cfg.get("enabled", True) and not args.skip_timing:
        timing_prompts = prompts
        if timing_cfg.get("max_prompts"):
            timing_prompts = prompts[: int(timing_cfg["max_prompts"])]

        if timing_cfg.get("greedy", True):
            traces = measure_timing(
                pair,
                timing_prompts,
                n_new_tokens=int(timing_cfg["n_new_tokens"]),
                do_sample=False,
                temperature=1.0,
                device=device,
                label="greedy",
            )
            agg = aggregate_timing(traces)
            summary["timing_greedy"] = agg
            dump_json(run_dir / "traces_timing_greedy.json", [asdict(t) for t in traces])
            if tb:
                for k, v in agg.items():
                    if isinstance(v, (int, float)) and v is not None:
                        tb.add_scalar(f"timing/greedy/{k}", v, 0)
            logger.info(
                f"[timing:greedy] tok/s={agg['tok_per_sec']:.2f}  "
                f"new/target_call={agg['new_tokens_per_target_call']:.3f}"
            )

        if timing_cfg.get("sampled", False):
            traces = measure_timing(
                pair,
                timing_prompts,
                n_new_tokens=int(timing_cfg["n_new_tokens"]),
                do_sample=True,
                temperature=float(timing_cfg.get("temperature", 1.0)),
                device=device,
                label="sampled",
            )
            agg = aggregate_timing(traces)
            summary["timing_sampled"] = agg
            dump_json(run_dir / "traces_timing_sampled.json", [asdict(t) for t in traces])
            if tb:
                for k, v in agg.items():
                    if isinstance(v, (int, float)) and v is not None:
                        tb.add_scalar(f"timing/sampled/{k}", v, 0)
            logger.info(
                f"[timing:sampled] tok/s={agg['tok_per_sec']:.2f}  "
                f"new/target_call={agg['new_tokens_per_target_call']:.3f}"
            )

    # ---- kill criterion -----------------------------------------------------
    kill = cfg.get("kill_criterion", {})
    g_thr = float(kill.get("greedy_recall_threshold", 0.99))
    s_thr = float(kill.get("sampled_recall_threshold", 0.95))
    g = summary.get("recall_greedy", {}).get("recall")
    s = summary.get("recall_sampled", {}).get("recall")
    if g is not None and s is not None:
        decision = "KILL_GEMMA_TRACK" if (g >= g_thr and s >= s_thr) else "GO"
    else:
        decision = "UNAVAILABLE"
    summary["kill_criterion"] = {
        "greedy_recall_threshold": g_thr,
        "sampled_recall_threshold": s_thr,
        "greedy_recall": g,
        "sampled_recall": s,
        "decision": decision,
        "interpretation": (
            "KILL_GEMMA_TRACK = recall already saturated; cluster modelling "
            "has no headroom; default back to EvaByte. "
            "GO = headroom exists; proceed to E1 (top-k sweep)."
        ),
    }
    dump_json(run_dir / "summary.json", summary)
    if tb:
        tb.close()

    print("\n" + "=" * 72)
    print(
        f"DECISION: {decision}   "
        f"(greedy_recall={g}  sampled_recall={s}; "
        f"thresholds g≥{g_thr}, s≥{s_thr})"
    )
    print(f"Outputs: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
