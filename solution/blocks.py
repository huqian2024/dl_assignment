"""Reusable 1D building blocks: normalizations, residual conv blocks, ResMLP."""

from __future__ import annotations

import torch
from torch import nn


def build_norm(kind: str, channels: int, groups: int = 8) -> nn.Module:
    """Normalization factory that is safe for small batches.

    ``group``  : GroupNorm with ``groups`` groups (fallback to 1 if needed)
    ``layer``  : per-position layer-norm over channels (batch independent)
    ``batch``  : BatchNorm1d (available, but never a project default)
    ``none``   : identity
    """
    key = str(kind).lower()
    if key == "group":
        num_groups = int(groups)
        while num_groups > 1 and channels % num_groups != 0:
            num_groups //= 2
        return nn.GroupNorm(num_groups, channels)
    if key in ("layer", "channel"):
        return nn.GroupNorm(1, channels)
    if key in ("batch", "batchnorm"):
        return nn.BatchNorm1d(channels)
    if key in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown normalization '{kind}'")


class ResBlock1D(nn.Module):
    """``x + f(x)`` with two conv/norm/GELU layers and an optional projection.

    Stride is fixed to 1: downsampling is the job of pooling modules, so the
    residual path stays shape-preserving and easy to reason about.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        norm: str = "group",
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ResBlock1D requires an odd kernel size")
        padding = int(dilation) * (int(kernel_size) // 2)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=int(dilation), bias=True,
        )
        self.norm1 = build_norm(norm, out_channels, groups)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=int(dilation), bias=True,
        )
        self.norm2 = build_norm(norm, out_channels, groups)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + residual)


class ResFCBlock(nn.Module):
    """Simple residual MLP block used by :class:`ResMLPHead`."""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x)))
        h = self.dropout(h)
        h = self.norm2(self.fc2(h))
        return self.act(h + x)
