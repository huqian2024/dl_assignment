"""Shared utilities: project paths, seeding, IO, timing and model bookkeeping.

Only the Python standard library, numpy and torch are used here.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS_DIR = PROJECT_ROOT / "assignments"
CACHE_DIR = PROJECT_ROOT / "cache"
DATASET_DIR = CACHE_DIR / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
COMMON_ARTIFACTS_DIR = ARTIFACTS_DIR / "common"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Public evaluation seeds of the assignment.  They are *never* used anywhere in
# this project; the constant exists so every split can be checked against it.
PUBLIC_EVAL_SEEDS = tuple(range(10000, 10050))


def ensure_assignments_on_path() -> None:
    """Allow ``from cm19_simulator import CM19Simulator`` without installing."""
    path = str(ASSIGNMENTS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def resolve(path: str | os.PathLike[str], base: Path = PROJECT_ROOT) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (including CUDA) deterministically."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto"):
    """Return a torch device following ``auto``/``cpu``/``cuda``."""
    import torch

    if preference == "auto":
        preference = "cuda" if torch.cuda.is_available() else "cpu"
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(preference)


def describe_environment() -> dict[str, Any]:
    """Collect runtime information for the stage report / logs."""
    import platform

    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": None,
        "gpu_count": 0,
        "cuda_device_capability": None,
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_device_capability"] = ".".join(
            str(v) for v in torch.cuda.get_device_capability(0)
        )
    return info


def cuda_sanity_check() -> dict[str, Any]:
    """Run a tiny CUDA matmul and report timing/result."""
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "cuda": True,
        "matmul_max_abs_err_vs_cpu": float(
            (c.cpu() - (a.cpu() @ b.cpu())).abs().max()
        ),
        "elapsed_ms": 1e3 * elapsed,
        "allocated_mb": torch.cuda.memory_allocated() / 2**20,
    }


# ---------------------------------------------------------------------------
# JSON / NPZ IO
# ---------------------------------------------------------------------------

def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def save_json(path: str | os.PathLike[str], payload: Any, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, default=_json_default), encoding="utf-8")
    os.replace(tmp, target)


def load_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | os.PathLike[str], record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def count_parameters(model, trainable_only: bool = True) -> int:
    if trainable_only:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in model.parameters()))


@contextmanager
def timer():
    """Wall-clock context manager yielding a mutable dict."""
    record: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - start


def bytes_to_human(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def total_size(paths: Iterable[str | os.PathLike[str]]) -> int:
    total = 0
    for path in paths:
        total += Path(path).stat().st_size
    return total
