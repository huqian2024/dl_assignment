"""Small JSON-config loader with deep-merge defaults.

No external configuration framework: plain JSON in, plain dicts out, so every
route is reproducible from a single file.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .utils import CONFIGS_DIR, load_json, resolve, save_json

# ---------------------------------------------------------------------------
# Defaults shared by every route.  Route configs only override what differs.
# ---------------------------------------------------------------------------

DATA_DEFAULTS: dict[str, Any] = {
    "train": {"start": 0, "stop": 8000},
    "validation": {"start": 20000, "stop": 21000},
    "internal_test": {"start": 30000, "stop": 31000},
    "dataset_dir": "cache/dataset",
    "batch_size": 32,
}

NORMALIZATION_DEFAULTS: dict[str, Any] = {
    "type": "per_channel_position",  # per_channel | per_channel_position
    "eps": 1e-6,
    "stats_file": "artifacts/common/norm_stats.npz",
}

REPRESENTATION_DEFAULTS: dict[str, Any] = {
    "type": "frequency",  # frequency | delay
    "window": "none",  # none | tukey | hann
    "tukey_alpha": 0.25,
    "delay_concat_raw": True,
    "delay_window": "tukey",
    # zero padding is applied *after* normalization inside the encoder
    "padding_points": 0,
}

ENCODER_DEFAULTS: dict[str, Any] = {
    "type": "resnet",  # resnet | fno
    "in_channels": 4,
    "channels": [64, 96, 128],
    "blocks_per_stage": 2,
    "kernel_size": 7,
    "dilation_cycle": [1, 2, 4],
    "norm": "group",
    "groups": 8,
    "dropout": 0.0,
    "pool_kernel": 4,
    "pool_stride": 2,
    "use_pooling": True,
    "multiscale": True,
    # FNO-only fields
    "n_modes": 64,
    "hidden_channels": 96,
    "local_kernel": 5,
    "fno_blocks": 4,
    "padding_points": 0,
}

POOLING_DEFAULTS: dict[str, Any] = {
    "type": "info_preserving",  # info_preserving | avg
    "kernel_size": 4,
    "stride": 2,
    "tau": 1.0,
    "learnable_tau": True,
    "tau_min": 0.05,
}

AGGREGATE_DEFAULTS: dict[str, Any] = {
    "type": "multiscale",  # multiscale | last | global
    "length": 8,
}

HEAD_DEFAULTS: dict[str, Any] = {
    "type": "res_mlp",
    "hidden": 256,
    "blocks": 2,
    "dropout": 0.0,
    "output_dim": 41,
}

MODEL_DEFAULTS: dict[str, Any] = {
    "route": "custom",
    "seed": 2026,
    "data": DATA_DEFAULTS,
    "representation": REPRESENTATION_DEFAULTS,
    "normalization": NORMALIZATION_DEFAULTS,
    "encoder": ENCODER_DEFAULTS,
    "pooling": POOLING_DEFAULTS,
    "aggregate": AGGREGATE_DEFAULTS,
    "head": HEAD_DEFAULTS,
    "anchor_file": "artifacts/common/anchor.npz",
    "output_clamp": False,
}

TRAIN_DEFAULTS: dict[str, Any] = {
    "seed": 2026,
    "epochs": 1,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "grad_clip": 1.0,
    "amp": True,
    "amp_dtype": "bf16",
    "device": "auto",
    "num_workers": 0,
    "scheduler": "cosine",
    "warmup_steps": 0,
    "loss": "huber",  # huber | mse
    "huber_delta": 1.0,
    "early_stopping_patience": 0,
    "checkpoint_every": 0,
    "output_dir": "runs",
    "run_name": None,
}

EVAL_DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "split": "internal_test",
    "batch_size": 64,
}

ANCHOR_SEARCH_DEFAULTS: dict[str, Any] = {
    "seed": 2026,
    "population": 256,
    "elite_frac": 0.08,
    "init_std": 0.35,
    "min_std": 0.01,
    "extra_std": 0.0,
    "smoothing": 0.7,
    "iterations": 200,
    "workers": 8,
    "objective": "softmin",  # min | softmin | mean
    "softmin_tau": 0.5,
    "early_stop_margin": 0.5,
    "local_refine_steps": 60,
    "local_refine_sigma": 0.02,
    "global_seed_pool": 0,
    "output": "artifacts/common/anchor.npz",
    "log_every": 5,
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a route config and merge it over :data:`MODEL_DEFAULTS`."""
    cfg: dict[str, Any] = copy.deepcopy(MODEL_DEFAULTS)
    if path is not None:
        cfg = deep_merge(cfg, load_json(resolve(path)))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def save_config(cfg: Mapping[str, Any], path: str) -> None:
    save_json(path, dict(cfg))


def route_config_path(name: str):
    """Resolve ``route_freq_resnet`` -> ``configs/route_freq_resnet.json``."""
    if not name.endswith(".json"):
        name = f"{name}.json"
    return CONFIGS_DIR / name


def seed_range(spec: Mapping[str, Any]) -> range:
    return range(int(spec["start"]), int(spec["stop"]))
