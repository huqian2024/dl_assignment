"""Shared test fixtures: temporary normalization statistics and anchors.

Tests must be independent of the real cache; they only need *valid* buffers.
No public evaluation seeds are used anywhere.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from solution.config import load_config  # noqa: E402
from solution.dsp import build_representation  # noqa: E402
from solution.normalization import estimate_stats, representation_stats_key, save_stats  # noqa: E402
from solution.simulator import load_simulator  # noqa: E402

ROUTE_CONFIGS = [
    "configs/route_freq_resnet.json",
    "configs/route_delay_resnet.json",
    "configs/route_delay_fno.json",
    "configs/route_dual_domain.json",
]


class TempArtifacts:
    """Context manager creating a temporary stats file and anchor file."""

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples
        self._tmp = None
        self.stats_path = None
        self.anchor_path = None
        self.simulator = None

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cm19_test_")
        base = Path(self._tmp.name)
        self.stats_path = base / "norm_stats.npz"
        self.anchor_path = base / "anchor.npz"
        self.simulator = load_simulator()
        torch.manual_seed(0)
        responses = torch.randn(self.num_samples, 4, 2001)
        for rep_cfg in (
            {"type": "frequency"},
            {"type": "delay", "window": "tukey", "delay_concat_raw": True},
        ):
            representation = build_representation(rep_cfg, 2001)
            with torch.no_grad():
                features = representation(responses)
            stats = estimate_stats([features])
            save_stats(str(self.stats_path), representation_stats_key(rep_cfg), stats)
        center = 0.5 * (self.simulator.lower_bounds + self.simulator.upper_bounds)
        np.savez(
            self.anchor_path,
            m_star=center.astype(np.float64),
            margins=np.zeros(7),
            min_margin=np.asarray(-1.0),
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self._tmp.cleanup()
        return False

    def config(self, route_path: str) -> dict:
        cfg = load_config(route_path)
        cfg["normalization"]["stats_file"] = str(self.stats_path)
        for branch in cfg.get("branches") or []:
            branch["normalization"]["stats_file"] = str(self.stats_path)
        cfg["anchor_file"] = str(self.anchor_path)
        return cfg
