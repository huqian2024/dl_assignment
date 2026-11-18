"""Information-preserving pooling / downsampling.

For each local window the module builds three summaries:

* ``avg``  : the stable low-frequency/background component,
* ``y+``   : a learned softmax-weighted salient (peak-like) summary,
* ``y-``   : the same weights negated (dip-like / opposite) summary.

A 1x1 convolution fuses them and maps to the requested output width.  Softmax
is always local to a pooling window; weights sum to one by construction.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class InformationPreservingPool1D(nn.Module):
    """Average + learned salient/anti-salient fusion downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 4,
        stride: int = 2,
        tau: float = 1.0,
        learnable_tau: bool = True,
        tau_min: float = 0.05,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.tau_min = float(tau_min)
        self.score = nn.Conv1d(self.in_channels, 1, kernel_size=1, bias=True)
        self.fuse = nn.Conv1d(
            3 * self.in_channels, self.out_channels, kernel_size=1, bias=True
        )
        # softplus(log_tau) + tau_min keeps tau strictly positive and bounded
        # away from zero, preventing saturated/one-hot softmax degeneracy.
        safe_tau = max(float(tau) - self.tau_min, 1e-3)
        initial = math.log(math.expm1(safe_tau))
        self.log_tau = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.log_tau.requires_grad = bool(learnable_tau)

    # -- helpers ---------------------------------------------------------
    def _num_windows(self, length: int) -> int:
        if length < self.kernel_size:
            raise ValueError(
                f"sequence length {length} shorter than pooling kernel {self.kernel_size}"
            )
        return (length - self.kernel_size) // self.stride + 1

    def _pad_for_windows(self, x: torch.Tensor, num_windows: int) -> torch.Tensor:
        needed = (num_windows - 1) * self.stride + self.kernel_size
        missing = needed - int(x.shape[-1])
        if missing > 0:
            x = F.pad(x, (0, missing), mode="constant", value=0.0)
        return x

    def _windows(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_windows = self._num_windows(int(x.shape[-1]))
        x = self._pad_for_windows(x, num_windows)
        return x.unfold(2, self.kernel_size, self.stride), num_windows

    def current_tau(self) -> torch.Tensor:
        return F.softplus(self.log_tau) + self.tau_min

    def softmax_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha_plus, alpha_minus) shaped (B, n_win, K).

        Exposed for tests and analysis; ``forward`` recomputes them internally.
        """
        tau = self.current_tau()
        _, num_windows = self._windows(x)
        scores = self.score(x).unfold(2, self.kernel_size, self.stride).squeeze(1)
        if int(scores.shape[-2]) != num_windows:
            raise RuntimeError("score/window misalignment during pooling")
        alpha_plus = torch.softmax(scores / tau, dim=-1)
        alpha_minus = torch.softmax(-scores / tau, dim=-1)
        return alpha_plus, alpha_minus

    # -- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B,C,L) input, got {tuple(x.shape)}")
        tau = self.current_tau()
        windows, num_windows = self._windows(x)
        average = windows.mean(dim=-1)

        scores = self.score(x).unfold(2, self.kernel_size, self.stride).squeeze(1)
        if int(scores.shape[-2]) != num_windows:
            raise RuntimeError("score/window misalignment during pooling")
        weights_plus = torch.softmax(scores / tau, dim=-1)
        weights_minus = torch.softmax(-scores / tau, dim=-1)
        salient = (windows * weights_plus.unsqueeze(1)).sum(dim=-1)
        anti_salient = (windows * weights_minus.unsqueeze(1)).sum(dim=-1)

        fused = torch.cat([average, salient, anti_salient], dim=1)
        return self.fuse(fused)


class AveragePool1D(nn.Module):
    """Plain average-pool reference used only for debugging/ablation hooks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 4,
        stride: int = 2,
    ):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        self.out_channels = int(out_channels or in_channels)
        self.proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if out_channels not in (None, in_channels)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


def build_pooling(cfg: dict, in_channels: int) -> nn.Module:
    pool_type = str(cfg.get("type", "info_preserving")).lower()
    out_channels = int(cfg.get("out_channels", in_channels))
    kernel = int(cfg.get("kernel_size", 4))
    stride = int(cfg.get("stride", 2))
    if pool_type in ("info_preserving", "info", "softmax"):
        return InformationPreservingPool1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            tau=float(cfg.get("tau", 1.0)),
            learnable_tau=bool(cfg.get("learnable_tau", True)),
            tau_min=float(cfg.get("tau_min", 0.05)),
        )
    if pool_type == "avg":
        return AveragePool1D(in_channels, out_channels, kernel, stride)
    raise ValueError(f"Unknown pooling type '{pool_type}'")


class MultiScaleAggregator(nn.Module):
    """Combine feature maps from several resolutions into one sequence."""

    def __init__(self, mode: str = "multiscale", length: int = 8) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.length = int(length)
        if self.mode not in ("multiscale", "last", "global"):
            raise ValueError(f"Unknown aggregation mode '{mode}'")

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if self.mode == "last":
            return F.adaptive_avg_pool1d(features[-1], self.length)
        if self.mode == "global":
            pooled = [F.adaptive_avg_pool1d(f, 1) for f in features]
        else:
            pooled = [F.adaptive_avg_pool1d(f, self.length) for f in features]
        return torch.cat(pooled, dim=1)
