"""Head-only training loop on top of a frozen Gemma 4 backbone.

For G1, we train each head on a fixed train slice of Tülu 3 and report
held-out joint cluster NLL. The backbone is frozen; gradients only flow
through the head parameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from .backbone import Backbone
from .clusters import ClusterSystem
from .data import TuluHeadDataset, collate
from .heads.base import MTPCHead


@dataclass
class TrainResult:
    head_name: str
    epoch_losses: list[float]
    val_joint_nll: float | None
    val_per_offset_nll: list[float] | None
    train_seconds: float
    n_steps: int


def _move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _labels_from_future_tokens(
    future_tokens: torch.Tensor, clusters: ClusterSystem
) -> torch.Tensor:
    """Convert [B, L, n] token ids → cluster ids on the same device."""
    return clusters.cluster_of(future_tokens)


@torch.no_grad()
def evaluate(
    head: MTPCHead,
    backbone: Backbone,
    val_loader: DataLoader,
    clusters: ClusterSystem,
    device: str,
    max_batches: int | None = None,
) -> dict:
    head.eval()
    joint_nll_sum = 0.0
    joint_n = 0
    per_offset_sum = torch.zeros(head.n_future)
    per_offset_n = torch.zeros(head.n_future)

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _move_batch(batch, device)
        h = backbone.extract(batch["input_ids"], batch["attention_mask"])
        lbl = _labels_from_future_tokens(batch["future_tokens"], clusters)

        jlp = head.joint_logprob(h, lbl)                  # [B, L]
        full_valid = ((lbl != -1).all(dim=-1)) & batch["loss_mask"]
        n_valid = int(full_valid.sum().item())
        joint_nll_sum += float((-jlp * full_valid.float()).sum().item())
        joint_n += n_valid

        per_off = head.per_offset_nll(h, lbl, batch["loss_mask"])
        valid_per_off = ((lbl != -1) & batch["loss_mask"].unsqueeze(-1)).float().sum(
            dim=(0, 1)
        )
        per_offset_sum += per_off.detach().cpu() * valid_per_off.detach().cpu()
        per_offset_n += valid_per_off.detach().cpu()

    head.train()
    joint = joint_nll_sum / max(joint_n, 1)
    per_off = (per_offset_sum / per_offset_n.clamp_min(1.0)).tolist()
    return {
        "joint_nll": joint,
        "per_offset_nll": per_off,
        "n_joint": joint_n,
    }


def train_one_head(
    head_name: str,
    head: MTPCHead,
    backbone: Backbone,
    train_loader: DataLoader,
    val_loader: DataLoader,
    clusters: ClusterSystem,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    gamma: float,
    log_every: int,
    tb_writer=None,
    eval_max_batches: int | None = None,
) -> TrainResult:
    head = head.to(device)
    head.train()
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    logger.info(
        f"[{head_name}] params={n_params/1e6:.2f}M  lr={lr}  γ={gamma}"
    )
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_losses: list[float] = []
    t0 = time.perf_counter()
    step = 0
    for ep in range(epochs):
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            h = backbone.extract(batch["input_ids"], batch["attention_mask"])
            lbl = _labels_from_future_tokens(batch["future_tokens"], clusters)

            loss, stats = head.loss(h, lbl, batch["loss_mask"], gamma=gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()

            running += stats["loss"]
            n_batches += 1
            step += 1
            if step % log_every == 0:
                logger.info(
                    f"[{head_name}] ep={ep} step={step} loss={stats['loss']:.4f} "
                    f"per_offset={['%.3f' % x for x in stats['per_offset_nll']]}"
                )
                if tb_writer:
                    tb_writer.add_scalar(f"{head_name}/train/loss", stats["loss"], step)
                    for k, v in enumerate(stats["per_offset_nll"]):
                        tb_writer.add_scalar(
                            f"{head_name}/train/per_offset_nll_{k}", v, step
                        )

        ep_loss = running / max(n_batches, 1)
        epoch_losses.append(ep_loss)
        logger.info(f"[{head_name}] ep={ep} mean_loss={ep_loss:.4f}")

    val = evaluate(head, backbone, val_loader, clusters, device, eval_max_batches)
    logger.info(
        f"[{head_name}] VAL  joint_nll={val['joint_nll']:.4f}  "
        f"per_offset={['%.3f' % x for x in val['per_offset_nll']]}  "
        f"(n_joint={val['n_joint']})"
    )
    if tb_writer:
        tb_writer.add_scalar(f"{head_name}/val/joint_nll", val["joint_nll"], step)
        for k, v in enumerate(val["per_offset_nll"]):
            tb_writer.add_scalar(f"{head_name}/val/per_offset_nll_{k}", v, step)

    return TrainResult(
        head_name=head_name,
        epoch_losses=epoch_losses,
        val_joint_nll=val["joint_nll"],
        val_per_offset_nll=val["per_offset_nll"],
        train_seconds=time.perf_counter() - t0,
        n_steps=step,
    )
