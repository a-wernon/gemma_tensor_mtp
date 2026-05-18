"""G1 entry point — factorisation-gap test at cluster level.

Trains cluster-MTPC-FF (rank-1) and cluster-MTPC-CP (rank-r) head-only
on frozen Gemma 4 E2B and reports held-out joint cluster NLL.

Decision gate:
    CP joint_nll gain over FF < 2% at n=8 → KILL the Phase G direction.
    Joint cluster structure does not exist meaningfully above rank-1.

Usage:
    uv run python run_g1.py --config configs/g1_factorisation.yaml
    uv run python run_g1.py --config configs/g1_factorisation.yaml --smoke

Outputs under runs/g1/<run_name>_<ts>/:
    config.json, run.log, summary.json, tb/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from gmtp.backbone import Backbone
from gmtp.clusters import build_cluster_system
from gmtp.data import TuluConfig, TuluHeadDataset, collate
from gmtp.gemma_io import find_masked_embedder, get_token_ordering, load_target
from gmtp.heads import build_head
from gmtp.train_head import train_one_head
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
        help="Tiny sanity pass: 200 examples, 1 epoch, CP rank=4.",
    )
    return p.parse_args()


def _apply_smoke(cfg: dict) -> None:
    cfg["data"]["max_examples"] = 200
    cfg["loader"]["batch_size"] = 2
    cfg["train"]["epochs"] = 1
    cfg["train"]["log_every"] = 10
    for h in cfg["heads"]:
        if h["name"] == "cp":
            h["kwargs"]["rank"] = 4
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

    # ---- backbone + clusters -----------------------------------------------
    target = load_target(cfg["target"], dtype, device)
    # The cluster system lives on the assistant's masked embedder. For G1
    # we don't run the assistant; we just need its `token_ordering` buffer
    # to derive cluster labels. We load the assistant briefly to extract
    # token_ordering, then drop it. (Cheap relative to the target.)
    from gmtp.gemma_io import _load_model
    assistant_id = cfg.get("assistant", cfg["target"] + "-assistant")
    logger.info(f"Loading assistant (for token_ordering only): {assistant_id}")
    assistant = _load_model(assistant_id, dtype=dtype, device="cpu").eval()
    masked = find_masked_embedder(assistant)
    token_ordering = get_token_ordering(masked).clone().cpu()
    del assistant
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clusters = build_cluster_system(token_ordering)
    logger.info(
        f"Cluster system: V_cluster={clusters.num_centroids} "
        f"tokens_per_centroid={clusters.tokens_per_centroid} "
        f"covered tokens={int((clusters.token_to_cluster >= 0).sum().item())}"
    )

    backbone = Backbone(
        model=target.model,
        hidden_size=target.hidden_size,
        dtype=dtype,
        device=device,
    )

    # ---- data --------------------------------------------------------------
    dcfg = cfg["data"]
    full_ds = TuluHeadDataset(
        TuluConfig(
            dataset=dcfg["dataset"],
            split=dcfg["split"],
            n_future=int(dcfg["n_future"]),
            max_seq_len=int(dcfg["max_seq_len"]),
            max_examples=dcfg.get("max_examples"),
            chat_template=bool(dcfg.get("chat_template", True)),
        ),
        tokenizer=target.tokenizer,
    )
    val_n = max(1, int(len(full_ds) * float(dcfg.get("val_fraction", 0.02))))
    train_n = len(full_ds) - val_n
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=gen)
    logger.info(f"Data split: train={train_n} val={val_n} (n_future={dcfg['n_future']})")

    pad_id = target.tokenizer.pad_token_id or target.tokenizer.eos_token_id
    lcfg = cfg["loader"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(lcfg["batch_size"]),
        shuffle=True,
        num_workers=int(lcfg.get("num_workers", 0)),
        pin_memory=bool(lcfg.get("pin_memory", True)),
        collate_fn=lambda b: collate(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(lcfg["batch_size"]),
        shuffle=False,
        num_workers=int(lcfg.get("num_workers", 0)),
        pin_memory=bool(lcfg.get("pin_memory", True)),
        collate_fn=lambda b: collate(b, pad_id),
    )

    # ---- train each head ---------------------------------------------------
    summary: dict = {
        "config_resolved": cfg,
        "vocab_cluster": clusters.num_centroids,
        "tokens_per_centroid": clusters.tokens_per_centroid,
        "hidden_size": backbone.hidden_size,
        "train_examples": train_n,
        "val_examples": val_n,
        "heads": {},
    }

    for spec in cfg["heads"]:
        name = spec["name"]
        kwargs = dict(spec.get("kwargs", {}))
        head = build_head(
            name,
            hidden_size=backbone.hidden_size,
            n_clusters=clusters.num_centroids,
            n_future=int(dcfg["n_future"]),
            **kwargs,
        )
        res = train_one_head(
            head_name=name,
            head=head,
            backbone=backbone,
            train_loader=train_loader,
            val_loader=val_loader,
            clusters=clusters,
            device=device,
            epochs=int(cfg["train"]["epochs"]),
            lr=float(cfg["train"]["lr"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
            gamma=float(cfg["train"]["gamma"]),
            log_every=int(cfg["train"]["log_every"]),
            tb_writer=tb,
            eval_max_batches=cfg["train"].get("eval_max_batches"),
        )
        summary["heads"][name] = asdict(res)
        # Free head before training the next one.
        del head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- decision gate -----------------------------------------------------
    dec = cfg["decision"]
    ff = summary["heads"].get(dec["ff_head_name"], {})
    cp = summary["heads"].get(dec["cp_head_name"], {})
    ff_nll = ff.get("val_joint_nll")
    cp_nll = cp.get("val_joint_nll")
    label = "INSUFFICIENT_DATA"
    rel_gain = None
    if ff_nll is not None and cp_nll is not None and ff_nll > 0:
        # NLL is lower-is-better. Gain = (FF − CP) / FF.
        rel_gain = (ff_nll - cp_nll) / ff_nll
        label = (
            "GO_CLUSTER_STRUCTURE_EXISTS"
            if rel_gain >= float(dec["min_relative_gain"])
            else "KILL_NO_JOINT_STRUCTURE"
        )

    summary["decision"] = {
        "label": label,
        "ff_val_joint_nll": ff_nll,
        "cp_val_joint_nll": cp_nll,
        "relative_gain": rel_gain,
        "threshold": float(dec["min_relative_gain"]),
        "interpretation": (
            "GO_CLUSTER_STRUCTURE_EXISTS = CP beat FF on val joint NLL by "
            "≥ threshold; cluster joint structure exists above rank-1. "
            "Proceed to G2. "
            "KILL_NO_JOINT_STRUCTURE = CP gain < threshold; joint cluster "
            "structure does not exist meaningfully. Phase G dies the same "
            "way Qwen3-CP did at large V."
        ),
    }
    dump_json(run_dir / "summary.json", summary)
    if tb:
        tb.close()

    print("\n" + "=" * 72)
    print(f"DECISION: {label}")
    if rel_gain is not None:
        print(
            f"   FF joint NLL = {ff_nll:.4f}\n"
            f"   CP joint NLL = {cp_nll:.4f}\n"
            f"   relative gain = {rel_gain*100:.2f}% "
            f"(threshold {dec['min_relative_gain']*100:.1f}%)"
        )
    print(f"Outputs: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
