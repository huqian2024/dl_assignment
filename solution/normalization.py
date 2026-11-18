"""Normalization layers and 41-dimensional state scaling.

Design rule: representation first, normalization second.  For the delay route
the complex IFFT is applied to the raw S before any statistics are computed.

Every statistic lives in a registered buffer, so a traced module needs no
external ``.npy``/``.npz`` file at inference time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .utils import resolve


class PerChannelNorm(nn.Module):
    """Per-channel global standardization, buffers shaped (1, C, 1)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1, 1).float(), persistent=True)
        self.register_buffer("std", std.view(1, -1, 1).float(), persistent=True)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


class PerChannelPositionNorm(nn.Module):
    """Per-channel, per-position standardization, buffers shaped (1, C, L)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", mean.view(1, *mean.shape).float(), persistent=True)
        self.register_buffer("std", std.view(1, *std.shape).float(), persistent=True)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


def build_normalizer(cfg: dict, stats: dict[str, np.ndarray]) -> nn.Module:
    """Instantiate the configured normalization from pre-computed statistics."""
    norm_type = str(cfg.get("type", "per_channel_position")).lower()
    eps = float(cfg.get("eps", 1e-6))
    mean = torch.as_tensor(np.asarray(stats["mean"], dtype=np.float32))
    std = torch.as_tensor(np.asarray(stats["std"], dtype=np.float32))
    if norm_type == "per_channel":
        return PerChannelNorm(mean, std, eps)
    if norm_type == "per_channel_position":
        return PerChannelPositionNorm(mean, std, eps)
    raise ValueError(f"Unknown normalization type '{norm_type}'")


class StateNormalizer(nn.Module):
    """Maps the 41-dimensional state vector into ``[-1, 1]`` using the bounds.

    ``u = (m - c) / r`` with ``c`` the box center and ``r`` the half range.  The
    inverse is exact up to the (tiny) minimum half-range guard applied when the
    buffers are built.
    """

    def __init__(self, center: torch.Tensor, half_range: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("center", center.view(1, -1).float(), persistent=True)
        self.register_buffer("half_range", half_range.view(1, -1).float(), persistent=True)

    def encode(self, m: torch.Tensor) -> torch.Tensor:
        return (m - self.center) / self.half_range

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return self.center + self.half_range * u


def state_buffers_from_bounds(lower: np.ndarray, upper: np.ndarray, min_half_range: float = 1e-3):
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    center = 0.5 * (lower + upper)
    half_range = np.maximum(0.5 * (upper - lower), float(min_half_range))
    return (
        torch.as_tensor(center, dtype=torch.float32),
        torch.as_tensor(half_range, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Statistics estimation (training split only!)
# ---------------------------------------------------------------------------

def representation_stats_key(rep_cfg: dict) -> str:
    """Stable key used in ``norm_stats.npz`` for a representation."""
    rep_type = str(rep_cfg.get("type", "frequency")).lower()
    if rep_type == "frequency":
        return "freq"
    window = str(rep_cfg.get("window", "tukey")).lower()
    concat = bool(rep_cfg.get("delay_concat_raw", True))
    has_window = window not in ("none", "rect", "rectangular", "")
    return f"delay_{window}{'_concat' if (concat and has_window) else ''}"


def estimate_stats(feature_batches) -> dict[str, np.ndarray]:
    """Estimate per-channel and per-channel-per-position statistics.

    ``feature_batches`` yields float tensors/arrays shaped (B, C, L).  A single
    pass accumulates sums and sums of squares in float64; mean/var are formed
    afterwards, so memory stays proportional to ``C * L``.
    """
    count = 0
    total = None
    total_sq = None
    position_sum = None
    position_sq = None
    length = 0
    for batch in feature_batches:
        x = torch.as_tensor(batch, dtype=torch.float64)
        if x.dim() != 3:
            raise ValueError(f"Expected (B,C,L) feature batch, got {tuple(x.shape)}")
        if total is None:
            channels = int(x.shape[1])
            length = int(x.shape[2])
            total = torch.zeros(channels, dtype=torch.float64)
            total_sq = torch.zeros(channels, dtype=torch.float64)
            position_sum = torch.zeros(channels, length, dtype=torch.float64)
            position_sq = torch.zeros(channels, length, dtype=torch.float64)
        count += int(x.shape[0])
        total += x.sum(dim=(0, 2))
        total_sq += (x * x).sum(dim=(0, 2))
        position_sum += x.sum(dim=0)
        position_sq += (x * x).sum(dim=0)

    if count == 0 or total is None or position_sum is None:
        raise ValueError("No feature batches were provided")

    scale = float(count * length)
    per_channel_mean = total / scale
    per_channel_var = total_sq / scale - per_channel_mean**2
    position_mean = position_sum / float(count)
    position_var = position_sq / float(count) - position_mean**2
    return {
        "per_channel_mean": per_channel_mean.numpy().astype(np.float64),
        "per_channel_std": np.sqrt(np.maximum(per_channel_var.numpy(), 0.0)),
        "position_mean": position_mean.numpy().astype(np.float64),
        "position_std": np.sqrt(np.maximum(position_var.numpy(), 0.0)),
    }


def save_stats(path: str, key: str, stats: dict[str, np.ndarray]) -> None:
    """Merge a representation entry into ``norm_stats.npz`` (atomic rewrite)."""
    from pathlib import Path

    import os

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {}
    if target.exists():
        with np.load(target) as existing:
            archive = {name: existing[name] for name in existing.files}
    for name, value in stats.items():
        archive[f"{key}/{name}"] = np.asarray(value, dtype=np.float64)
    tmp = target.with_suffix(".tmp.npz")
    np.savez(tmp, **archive)
    os.replace(tmp, target)


def load_stats(path: str, key: str, norm_type: str) -> dict[str, np.ndarray]:
    """Read one representation/normalization entry from ``norm_stats.npz``."""
    with np.load(resolve(path)) as archive:
        if norm_type == "per_channel":
            mean_key, std_key = "per_channel_mean", "per_channel_std"
        else:
            mean_key, std_key = "position_mean", "position_std"
        mean_name, std_name = f"{key}/{mean_key}", f"{key}/{std_key}"
        if mean_name not in archive or std_name not in archive:
            available = ", ".join(sorted(archive.files))
            raise KeyError(f"Missing '{mean_name}' in {path}. Available: {available}")
        return {
            "mean": np.array(archive[mean_name]),
            "std": np.array(archive[std_name]),
        }
