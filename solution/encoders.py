"""Composable encoders: residual conv stack and 1D Fourier neural operator.

Both encoders return a *list* of feature maps (one per resolution stage) so the
same code supports single-scale and multi-scale heads.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from .blocks import ResBlock1D, build_norm
from .dsp import crop_delay, pad_delay
from .pooling import build_pooling


class ResNetEncoder(nn.Module):
    """Dilated residual Conv1d encoder with information-preserving pooling."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        blocks_per_stage: int = 2,
        kernel_size: int = 7,
        dilation_cycle: List[int] | tuple[int, ...] = (1, 2, 4),
        norm: str = "group",
        groups: int = 8,
        dropout: float = 0.0,
        pool_cfg: dict | None = None,
        use_pooling: bool = True,
        multiscale: bool = True,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("ResNetEncoder needs at least one stage width")
        self.use_pooling = bool(use_pooling)
        self.multiscale = bool(multiscale)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size, padding=kernel_size // 2),
            build_norm(norm, channels[0], groups),
            nn.GELU(),
        )
        dilations = tuple(int(d) for d in dilation_cycle) or (1,)
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for stage, width in enumerate(channels):
            stage_blocks = nn.ModuleList(
                [
                    ResBlock1D(
                        in_channels=width,
                        out_channels=width,
                        kernel_size=kernel_size,
                        dilation=dilations[(stage * blocks_per_stage + idx) % len(dilations)],
                        norm=norm,
                        groups=groups,
                        dropout=dropout,
                    )
                    for idx in range(blocks_per_stage)
                ]
            )
            self.blocks.append(stage_blocks)
            if use_pooling and stage < len(channels) - 1:
                stage_pool_cfg = dict(pool_cfg or {})
                stage_pool_cfg["out_channels"] = channels[stage + 1]
                self.pools.append(build_pooling(stage_pool_cfg, channels[stage]))
            else:
                self.pools.append(nn.Identity())
        self.final_pool = None
        if use_pooling:
            final_cfg = dict(pool_cfg or {})
            final_cfg.setdefault("out_channels", channels[-1])
            self.final_pool = build_pooling(final_cfg, channels[-1])

        feature_channels = list(channels)
        if self.final_pool is not None:
            feature_channels.append(int(self.final_pool.out_channels))
        self.feature_channels = feature_channels if self.multiscale else [feature_channels[-1]]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.stem(x)
        features: List[torch.Tensor] = []
        for stage, stage_blocks in enumerate(self.blocks):
            for block in stage_blocks:
                h = block(h)
            features.append(h)
            if stage < len(self.blocks) - 1:
                h = self.pools[stage](h)
        if self.final_pool is not None:
            features.append(self.final_pool(features[-1]))
        return features if self.multiscale else features[-1:]


class SpectralConv1D(nn.Module):
    """Low-mode complex multiplication in the Fourier domain.

    The complex weight is stored as two real parameters so the module can be
    traced and serialized without complex parameters.
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        super().__init__()
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = int(n_modes)
        scale = 1.0 / max(self.in_channels * self.out_channels, 1)
        shape = (self.in_channels, self.out_channels, self.n_modes)
        self.weight_re = nn.Parameter(scale * torch.randn(*shape))
        self.weight_im = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            batch, _, length = x32.shape
            spectrum = torch.fft.rfft(x32, dim=-1)
            freq_bins = int(spectrum.shape[-1])
            modes = min(self.n_modes, freq_bins)
            xr = spectrum.real[..., :modes]
            xi = spectrum.imag[..., :modes]
            wr = self.weight_re[..., :modes]
            wi = self.weight_im[..., :modes]
            out_r = torch.einsum("bci,coi->boi", xr, wr) - torch.einsum("bci,coi->boi", xi, wi)
            out_i = torch.einsum("bci,coi->boi", xr, wi) + torch.einsum("bci,coi->boi", xi, wr)
            zeros = torch.zeros(
                batch,
                self.out_channels,
                freq_bins - modes,
                dtype=out_r.dtype,
                device=out_r.device,
            )
            full_r = torch.cat([out_r, zeros], dim=-1)
            full_i = torch.cat([out_i, zeros], dim=-1)
            y = torch.fft.irfft(torch.complex(full_r, full_i), n=length, dim=-1)
        return y.to(x.dtype)


class FNOConvResidualBlock(nn.Module):
    """``x + phi(W_point x + F^-1[R F(x)] + C_local(x))``."""

    def __init__(
        self,
        channels: int,
        n_modes: int = 64,
        local_kernel: int = 5,
        norm: str = "group",
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if local_kernel % 2 == 0:
            raise ValueError("local_kernel must be odd")
        self.spectral = SpectralConv1D(channels, channels, n_modes)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.local = nn.Conv1d(
            channels, channels, local_kernel, padding=local_kernel // 2
        )
        self.norm = build_norm(norm, channels, groups)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x) + self.local(x)
        y = self.norm(y)
        y = self.act(y)
        return x + self.dropout(y)


class FNOEncoder(nn.Module):
    """Lifting + stacked Fourier blocks + information-preserving pooling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 96,
        n_modes: int = 64,
        fno_blocks: int = 4,
        local_kernel: int = 5,
        norm: str = "group",
        groups: int = 8,
        dropout: float = 0.0,
        padding_points: int = 0,
        pool_cfg: dict | None = None,
        use_pooling: bool = True,
        multiscale: bool = True,
    ) -> None:
        super().__init__()
        if padding_points < 0:
            raise ValueError("padding_points must be non-negative")
        self.padding_points = int(padding_points)
        self.use_pooling = bool(use_pooling)
        self.multiscale = bool(multiscale)
        self.lift = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1),
            build_norm(norm, hidden_channels, groups),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                FNOConvResidualBlock(
                    hidden_channels, n_modes, local_kernel, norm, groups, dropout
                )
                for _ in range(int(fno_blocks))
            ]
        )
        self.pools = nn.ModuleList()
        feature_channels = [hidden_channels]
        if use_pooling:
            widths = [hidden_channels]
            for _ in range(2):
                widths.append(max(hidden_channels // 2, 16))
            for idx in range(2):
                cfg = dict(pool_cfg or {})
                cfg["out_channels"] = widths[idx + 1]
                self.pools.append(build_pooling(cfg, widths[idx]))
                feature_channels.append(widths[idx + 1])
        self.feature_channels = (
            feature_channels if self.multiscale else [feature_channels[-1]]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = pad_delay(x, self.padding_points, self.padding_points)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        h = crop_delay(h, self.padding_points, self.padding_points)
        features: List[torch.Tensor] = [h]
        for pool in self.pools:
            h = pool(h)
            features.append(h)
        return features if self.multiscale else features[-1:]


def build_encoder(cfg: dict, in_channels: int) -> nn.Module:
    encoder_type = str(cfg.get("type", "resnet")).lower()
    norm = str(cfg.get("norm", "group"))
    groups = int(cfg.get("groups", 8))
    dropout = float(cfg.get("dropout", 0.0))
    use_pooling = bool(cfg.get("use_pooling", True))
    multiscale = bool(cfg.get("multiscale", True))
    pool_cfg = {
        "type": cfg.get("pooling_type", "info_preserving"),
        "kernel_size": int(cfg.get("pool_kernel", 4)),
        "stride": int(cfg.get("pool_stride", 2)),
        "tau": float(cfg.get("pool_tau", 1.0)),
        "learnable_tau": bool(cfg.get("pool_learnable_tau", True)),
        "tau_min": float(cfg.get("pool_tau_min", 0.05)),
    }
    if encoder_type == "resnet":
        return ResNetEncoder(
            in_channels=in_channels,
            channels=[int(c) for c in cfg.get("channels", [64, 96, 128])],
            blocks_per_stage=int(cfg.get("blocks_per_stage", 2)),
            kernel_size=int(cfg.get("kernel_size", 7)),
            dilation_cycle=cfg.get("dilation_cycle", [1, 2, 4]),
            norm=norm,
            groups=groups,
            dropout=dropout,
            pool_cfg=pool_cfg,
            use_pooling=use_pooling,
            multiscale=multiscale,
        )
    if encoder_type in ("fno", "fno_conv", "spectral"):
        return FNOEncoder(
            in_channels=in_channels,
            hidden_channels=int(cfg.get("hidden_channels", 96)),
            n_modes=int(cfg.get("n_modes", 64)),
            fno_blocks=int(cfg.get("fno_blocks", 4)),
            local_kernel=int(cfg.get("local_kernel", 5)),
            norm=norm,
            groups=groups,
            dropout=dropout,
            padding_points=int(cfg.get("padding_points", 0)),
            pool_cfg=pool_cfg,
            use_pooling=use_pooling,
            multiscale=multiscale,
        )
    raise ValueError(f"Unknown encoder type '{encoder_type}'")
