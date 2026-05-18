"""Frozen Gemma 4 backbone — extract per-position last-hidden states.

For G1 head-only training, we only need h_t per position. The target is
frozen; gradients flow through the head only. We run the target with
output_hidden_states=True and take the final layer.

Memory note: at d=1536, sequence 1024, bf16, one example produces
~3 MB of hidden state. For a batch of 8 sequences that's ~25 MB — fine.
For full Tülu fine-tuning at scale we'd cache hidden states; for G1 we
recompute on-the-fly (target forward is fast enough on H200).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class Backbone:
    model: nn.Module
    hidden_size: int
    dtype: torch.dtype
    device: str

    def __post_init__(self):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        logger.info(
            f"Backbone frozen: hidden_size={self.hidden_size} "
            f"dtype={self.dtype} device={self.device} "
            f"params={sum(p.numel() for p in self.model.parameters())/1e9:.2f}B"
        )

    @torch.inference_mode()
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return last-layer hidden states [B, L, hidden_size].

        Inference-mode — gradients NOT tracked through the backbone. The
        downstream head is trained on these hidden states as fixed
        features.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]
