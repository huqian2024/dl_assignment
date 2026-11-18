"""Composable one-shot S -> delta_m models.

Architecture is a product of independent parts:

    Representation x Encoder x Pooling/Aggregation x Head

Every route config instantiates the same class; only the parts differ.  The
network predicts the *normalized current state* ``u`` and converts it to a
correction with a fixed compliant anchor:

    m_hat = center + half_range * u_hat
    delta_m = anchor - m_hat
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .dsp import build_representation
from .encoders import build_encoder
from .heads import build_head
from .normalization import (
    StateNormalizer,
    build_normalizer,
    load_stats,
    representation_stats_key,
    state_buffers_from_bounds,
)
from .pooling import MultiScaleAggregator
from .utils import resolve


class FeatureBranch(nn.Module):
    """One domain: representation -> normalization -> encoder -> aggregation."""

    def __init__(
        self,
        representation: nn.Module,
        normalization: nn.Module,
        encoder: nn.Module,
        aggregator: MultiScaleAggregator,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.normalization = normalization
        self.encoder = encoder
        self.aggregator = aggregator

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        features = self.representation(response)
        features = self.normalization(features)
        encoded: List[torch.Tensor] = self.encoder(features)
        return self.aggregator(encoded)


class OneShotStateModel(nn.Module):
    """Standalone ``raw S (B,4,2001) -> delta_m (B,41)`` module."""

    def __init__(
        self,
        branches: List[FeatureBranch],
        feature_dim: int,
        head_cfg: dict,
        center: torch.Tensor,
        half_range: torch.Tensor,
        anchor: torch.Tensor,
        output_clamp: bool = False,
        route_name: str = "custom",
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.head = build_head(head_cfg, feature_dim)
        self.state_normalizer = StateNormalizer(center, half_range)
        self.register_buffer("anchor", anchor.view(1, -1).float(), persistent=True)
        self.output_clamp = bool(output_clamp)
        self.route_name = str(route_name)
        self.feature_dim = int(feature_dim)

    # -- core ------------------------------------------------------------
    def forward_state(
        self, response: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(u_hat, m_hat, delta_m)``; used by the trainer/evaluator."""
        parts = [branch(response) for branch in self.branches]
        features = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        u_hat = self.head(features.flatten(start_dim=1))
        if self.output_clamp:
            u_hat = torch.clamp(u_hat, -1.0, 1.0)
        m_hat = self.state_normalizer.decode(u_hat)
        delta_m = self.anchor - m_hat
        return u_hat, m_hat, delta_m

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        return self.forward_state(response)[2]


def _build_branch(
    branch_cfg: dict,
    num_freq_points: int,
    default_stats_path: str,
    default_pooling_cfg: dict | None = None,
) -> Tuple[FeatureBranch, int]:
    rep_cfg = dict(branch_cfg["representation"])
    norm_cfg = dict(branch_cfg["normalization"])
    enc_cfg = dict(branch_cfg["encoder"])
    agg_cfg = dict(branch_cfg["aggregate"])
    pool_cfg = dict(branch_cfg.get("pooling") or default_pooling_cfg or {})
    if pool_cfg:
        enc_cfg["pooling_type"] = str(pool_cfg.get("type", "info_preserving"))
        enc_cfg["pool_kernel"] = int(pool_cfg.get("kernel_size", 4))
        enc_cfg["pool_stride"] = int(pool_cfg.get("stride", 2))
        enc_cfg["pool_tau"] = float(pool_cfg.get("tau", 1.0))
        enc_cfg["pool_learnable_tau"] = bool(pool_cfg.get("learnable_tau", True))
        enc_cfg["pool_tau_min"] = float(pool_cfg.get("tau_min", 0.05))

    representation = build_representation(rep_cfg, num_freq_points)
    stats_path = norm_cfg.get("stats_file", default_stats_path)
    stats_key = norm_cfg.get("stats_key") or representation_stats_key(rep_cfg)
    stats = load_stats(
        resolve(stats_path),
        stats_key,
        str(norm_cfg.get("type", "per_channel_position")),
    )
    normalization = build_normalizer(norm_cfg, stats)
    encoder = build_encoder(enc_cfg, int(representation.out_channels))
    agg_mode = str(agg_cfg.get("type", "multiscale"))
    agg_length = int(agg_cfg.get("length", 8))
    aggregator = MultiScaleAggregator(mode=agg_mode, length=agg_length)
    channels = int(sum(encoder.feature_channels))
    length = 1 if agg_mode == "global" else agg_length
    feature_dim = channels * length
    return (
        FeatureBranch(representation, normalization, encoder, aggregator),
        feature_dim,
    )


def build_model(cfg: dict, simulator) -> OneShotStateModel:
    """Instantiate a route from its JSON config using only public simulator info."""
    num_freq_points = int(len(simulator.frequency_hz))
    default_stats_path = str(
        cfg["normalization"].get("stats_file", "artifacts/common/norm_stats.npz")
    )

    if cfg.get("branches"):
        branch_specs = list(cfg["branches"])
    else:
        branch_specs = [
            {
                "representation": cfg["representation"],
                "normalization": cfg["normalization"],
                "encoder": cfg["encoder"],
                "aggregate": cfg["aggregate"],
            }
        ]

    branches: List[FeatureBranch] = []
    feature_dim = 0
    default_pooling = dict(cfg.get("pooling", {}))
    for spec in branch_specs:
        branch, dim = _build_branch(
            spec, num_freq_points, default_stats_path, default_pooling
        )
        branches.append(branch)
        feature_dim += dim

    anchor_path = resolve(cfg.get("anchor_file", "artifacts/common/anchor.npz"))
    with np.load(anchor_path) as archive:
        anchor = np.asarray(archive["m_star"], dtype=np.float32)
    center, half_range = state_buffers_from_bounds(
        simulator.lower_bounds, simulator.upper_bounds
    )
    return OneShotStateModel(
        branches=branches,
        feature_dim=feature_dim,
        head_cfg=dict(cfg.get("head", {})),
        center=center,
        half_range=half_range,
        anchor=torch.as_tensor(anchor, dtype=torch.float32),
        output_clamp=bool(cfg.get("output_clamp", False)),
        route_name=str(cfg.get("route", "custom")),
    )


def model_config_summary(model: OneShotStateModel) -> dict:
    """Human-readable architecture summary for logs/reports."""
    lines = []
    for branch in model.branches:
        lines.append(
            {
                "representation": type(branch.representation).__name__,
                "normalization": type(branch.normalization).__name__,
                "encoder": type(branch.encoder).__name__,
                "aggregator": type(branch.aggregator).__name__,
            }
        )
    return {
        "route": model.route_name,
        "feature_dim": model.feature_dim,
        "branches": lines,
        "head": type(model.head).__name__,
    }
