"""SwiGLU activation (Shazeer 2020) — used in LLaMA, Mistral, PaLM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated linear unit with SiLU activation.

    Uses 2/3 of 4x expansion to match parameter count of standard FFN.
    """

    def __init__(self, d_in: int, d_hidden: int | None = None) -> None:
        super().__init__()
        d_hidden = d_hidden or int(2 * d_in * 4 / 3)
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w3 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
