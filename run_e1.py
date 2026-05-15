"""E1 entry point — top_k sweep.

Usage:
    python run_e1.py --config configs/e1_topk_sweep.yaml
    python run_e1.py --config configs/e1_topk_sweep.yaml --smoke
    # part-by-part:
    python run_e1.py --config configs/e1_topk_sweep.yaml --only recall
    python run_e1.py --config configs/e1_topk_sweep.yaml --only microbench
    python run_e1.py --config configs/e1_topk_sweep.yaml --only e2e

Outputs under runs/e1/<run_name>_<ts>/:
    config.json, run.log, summary.json, eval_set.jsonl
    traces_recall_topkmax_{greedy,sampled}.json
    traces_e2e_k{K}.json
    pareto.png  (recall vs latency vs throughput)
    tb/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from gmtp.data import build_eval_set
from gmtp.gemma_io import _get_token_ordering, load_pair
from gmtp.sweep import (
    aggregate_e2e_cell,
    derive_recall_at_k,
    measure_recall_topkmax,
    microbench_masked_embedder,
    time_e2e_per_topk,
)
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
    p.add_argument("--smoke", action="store_true",
                   help="Tiny pass: 8 prompts, 4 positions, 16 new tokens, k_values=[16,64].")
    p.add_argument("--only", choices=["recall", "microbench", "e2e"], default=None)
    return p.parse_args()


def _apply_smoke(cfg: dict) -> None:
    cfg["data"]["n_gsm8k"] = 4
    cfg["data"]["n_humaneval"] = 4
    cfg["recall"]["n_positions"] = 4
    cfg["recall"]["max_prompts"] = 8
    cfg["recall"]["K_max"] = 64
    cfg["recall"]["k_values"] = [16, 64]
    cfg["microbench"]["k_values"] = [16, 64]
    cfg["microbench"]["iters"] = 20
    cfg["microbench"]["warmup"] = 5
    cfg["e2e_timing"]["k_values"] = [16, 64]
    cfg["e2e_timing"]["n_new_tokens"] = 16
    cfg["e2e_timing"]["max_prompts"] = 4
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
    tb = SummaryWriter(str(run_dir / "tb")) if cfg["output"].get("tensorboard", True) else None

    device = cfg["device"]
    dtype = dtype_from_str(cfg["dtype"])

    # ---- eval set ----------------------------------------------------------
    cache_path = Path(cfg["data"].get("cache_dir", "cache/eval")) / "eval"
    prompts = build_eval_set(
        n_gsm8k=cfg["data"]["n_gsm8k"],
        n_humaneval=cfg["data"]["n_humaneval"],
        seed=cfg["seed"],
        cache_path=cache_path,
    )
    with open(run_dir / "eval_set.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p.to_dict()) + "\n")

    # ---- model pair --------------------------------------------------------
    pair = load_pair(cfg["target"], cfg["assistant"], dtype, device)
    token_ordering = _get_token_ordering(pair.masked_embedder)

    summary: dict = {
        "config_resolved": cfg,
        "vocab_size": pair.vocab_size,
        "num_centroids": pair.num_centroids,
        "tokens_per_centroid": pair.tokens_per_centroid,
        "default_centroid_top_k": pair.centroid_top_k,
    }

    # ---- E1a — recall sweep ------------------------------------------------
    if cfg["recall"]["enabled"] and (args.only in (None, "recall")):
        recall_prompts = prompts
        if cfg["recall"].get("max_prompts"):
            recall_prompts = prompts[: int(cfg["recall"]["max_prompts"])]
        K_max = int(cfg["recall"]["K_max"])
        k_values = list(cfg["recall"]["k_values"])

        for mode_name, do_sample in [("greedy", False), ("sampled", True)]:
            if not cfg["recall"].get(mode_name, True):
                continue
            traces = measure_recall_topkmax(
                pair,
                recall_prompts,
                n_positions=int(cfg["recall"]["n_positions"]),
                do_sample=do_sample,
                temperature=float(cfg["recall"].get("temperature", 1.0)),
                K_max=K_max,
                device=device,
                label=mode_name,
            )
            dump_json(
                run_dir / f"traces_recall_topkmax_{mode_name}.json",
                [
                    {
                        "prompt_idx": t.prompt_idx,
                        "source": t.source,
                        "positions": [asdict(p) for p in t.positions],
                    }
                    for t in traces
                ],
            )
            recall_at_k = derive_recall_at_k(traces, k_values, token_ordering)
            summary[f"recall_{mode_name}"] = recall_at_k
            for k, r in recall_at_k.items():
                logger.info(
                    f"[E1a:{mode_name}] k={k:>3}  recall={r['overall_recall']:.4f}  "
                    f"({r['n_hits']}/{r['n']})"
                )
                if tb:
                    tb.add_scalar(f"recall_{mode_name}/overall", r["overall_recall"] or 0.0, k)
                    for src, s in r["by_source"].items():
                        tb.add_scalar(f"recall_{mode_name}/{src}", s["recall"] or 0.0, k)

    # ---- E1b — masked-embedder microbench ----------------------------------
    if cfg["microbench"]["enabled"] and (args.only in (None, "microbench")):
        results = microbench_masked_embedder(
            pair,
            top_k_values=list(cfg["microbench"]["k_values"]),
            ctx_len=int(cfg["microbench"]["ctx_len"]),
            iters=int(cfg["microbench"]["iters"]),
            warmup=int(cfg["microbench"]["warmup"]),
            device=device,
            dtype=dtype,
        )
        summary["microbench_lm_head"] = {str(k): v for k, v in results.items()}
        if tb:
            for k, v in results.items():
                tb.add_scalar("microbench/median_ms", v["median_ms"], k)

    # ---- E1c — end-to-end per top_k ----------------------------------------
    if cfg["e2e_timing"]["enabled"] and (args.only in (None, "e2e")):
        e2e_prompts = prompts
        if cfg["e2e_timing"].get("max_prompts"):
            e2e_prompts = prompts[: int(cfg["e2e_timing"]["max_prompts"])]

        for mode_name, do_sample in [("greedy", False), ("sampled", True)]:
            if not cfg["e2e_timing"].get(mode_name, False):
                continue
            traces_by_k = time_e2e_per_topk(
                pair,
                e2e_prompts,
                top_k_values=list(cfg["e2e_timing"]["k_values"]),
                n_new_tokens=int(cfg["e2e_timing"]["n_new_tokens"]),
                do_sample=do_sample,
                temperature=float(cfg["e2e_timing"].get("temperature", 1.0)),
                device=device,
            )
            agg_by_k = {k: aggregate_e2e_cell(ts) for k, ts in traces_by_k.items()}
            summary[f"e2e_{mode_name}"] = {str(k): v for k, v in agg_by_k.items()}
            for k, ts in traces_by_k.items():
                dump_json(
                    run_dir / f"traces_e2e_{mode_name}_k{k}.json",
                    [asdict(t) for t in ts],
                )
            for k, agg in agg_by_k.items():
                logger.info(
                    f"[E1c:{mode_name}] k={k:>3}  tok/s={agg['tok_per_sec']:.2f}  "
                    f"new/tcall={agg['new_tokens_per_target_call']:.3f}"
                )
                if tb:
                    tb.add_scalar(f"e2e_{mode_name}/tok_per_sec", agg["tok_per_sec"] or 0, k)
                    tb.add_scalar(
                        f"e2e_{mode_name}/new_per_target_call",
                        agg["new_tokens_per_target_call"] or 0,
                        k,
                    )

    # ---- decision summary --------------------------------------------------
    dec = cfg.get("decision", {})
    g_recall = summary.get("recall_greedy", {})
    s_recall = summary.get("recall_sampled", {})
    g_at_64 = (g_recall.get(64) or {}).get("overall_recall")
    g_at_128 = (g_recall.get(128) or {}).get("overall_recall")
    s_at_64 = (s_recall.get(64) or {}).get("overall_recall")
    label = "INSUFFICIENT_DATA"
    if g_at_64 is not None and g_at_128 is not None:
        if g_at_64 >= float(dec.get("easy_headroom_recall_at_k64", 0.95)):
            label = "EASY_HEADROOM"   # buy more clusters
        elif g_at_128 < float(dec.get("structural_problem_recall_at_k128", 0.85)):
            label = "STRUCTURAL_PROBLEM"  # joint selection plausibly worth it
        else:
            label = "MIXED"
    summary["decision"] = {
        "label": label,
        "greedy_recall_at_k64": g_at_64,
        "greedy_recall_at_k128": g_at_128,
        "sampled_recall_at_k64": s_at_64,
        "interpretation": (
            "EASY_HEADROOM = recall reaches the easy_headroom threshold by k=64; "
            "case for structured selection weakens. "
            "STRUCTURAL_PROBLEM = recall stays below structural_problem threshold "
            "even at k=128; strong case for E2/E3. "
            "MIXED = somewhere in between; decide on Pareto plot."
        ),
    }

    dump_json(run_dir / "summary.json", summary)
    _plot_pareto(summary, run_dir)
    if tb:
        tb.close()

    print("\n" + "=" * 72)
    print(
        f"DECISION: {label}   "
        f"(greedy recall@64={g_at_64}, recall@128={g_at_128}, "
        f"sampled@64={s_at_64})"
    )
    print(f"Outputs: {run_dir}")
    print("=" * 72)


def _plot_pareto(summary: dict, run_dir: Path) -> None:
    """Recall-vs-k curves + (if microbench available) recall-vs-latency Pareto."""
    g = summary.get("recall_greedy") or {}
    s = summary.get("recall_sampled") or {}
    micro = summary.get("microbench_lm_head") or {}
    if not g and not s:
        return

    fig, axes = plt.subplots(1, 2 if micro else 1, figsize=(11 if micro else 6, 4.5), squeeze=False)
    ax = axes[0][0]
    if g:
        ks = sorted(g.keys())
        ax.plot(ks, [g[k]["overall_recall"] for k in ks], marker="o", label="greedy")
    if s:
        ks = sorted(s.keys())
        ax.plot(ks, [s[k]["overall_recall"] for k in ks], marker="s", label="sampled")
    ax.set_xlabel("centroid top_k")
    ax.set_ylabel("first-draft cluster recall")
    ax.set_xscale("log", base=2)
    ax.set_title("E1a — recall vs top_k")
    ax.grid(alpha=0.3)
    ax.legend()

    if micro:
        ax2 = axes[0][1]
        ks_lat = sorted(int(k) for k in micro.keys())
        lat = [micro[str(k)]["median_ms"] for k in ks_lat]
        if g:
            recalls = [g.get(k, {}).get("overall_recall") for k in ks_lat]
            pairs = [(l, r, k) for l, r, k in zip(lat, recalls, ks_lat) if r is not None]
            if pairs:
                xs, ys, ks_p = zip(*pairs)
                ax2.plot(xs, ys, marker="o")
                for x, y, k in pairs:
                    ax2.annotate(f"k={k}", (x, y), fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax2.set_xlabel("LM-head latency (median ms)")
        ax2.set_ylabel("greedy first-draft recall")
        ax2.set_title("E1b/a — recall vs LM-head cost (Pareto)")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "pareto.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
