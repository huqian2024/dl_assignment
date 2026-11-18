"""Prediction heads: residual MLP mapping pooled features to the 41-D state."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ResFCBlock


class ResMLPHead(nn.Module):
    """Linear lift -> residual MLP blocks -> linear readout (no output squash)."""

    def __init__(
        self,
        in_features: int,
        output_dim: int = 41,
        hidden: int = 256,
        blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(self.in_features, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList(
            [ResFCBlock(hidden, dropout) for _ in range(int(blocks))]
        )
        self.out = nn.Linear(hidden, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.proj(x)))
        for block in self.blocks:
            h = block(h)
        return self.out(h)


def build_head(cfg: dict, in_features: int) -> nn.Module:
    head_type = str(cfg.get("type", "res_mlp")).lower()
    if head_type in ("res_mlp", "resmlp", "mlp"):
        return ResMLPHead(
            in_features=in_features,
            output_dim=int(cfg.get("output_dim", 41)),
            hidden=int(cfg.get("hidden", 256)),
            blocks=int(cfg.get("blocks", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    raise ValueError(f"Unknown head type '{head_type}'")
