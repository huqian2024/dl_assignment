"""Dataset construction and caching for CM19.

Seeds are split by range and a manifest records exact counts, sizes, simulator
call count and wall-clock time.  Public evaluation seeds (10000-10049) are
rejected by construction.

Cache layout (``cache/dataset``)::

    manifest.json
    train_responses.npy   float32 (N, 4, 2001)
    train_m_current.npy   float32 (N, 41)
    train_seeds.npy       int64   (N,)
    validation_*          ...
    internal_test_*       ...
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .simulator import load_simulator
from .utils import (
    DATASET_DIR,
    PUBLIC_EVAL_SEEDS,
    bytes_to_human,
    load_json,
    resolve,
    save_json,
)

SPLITS = ("train", "validation", "internal_test")
_WORKER_SIM = None


def _worker_init() -> None:
    global _WORKER_SIM
    _WORKER_SIM = load_simulator()


def _worker_state(seed: int):
    m_current, response = _WORKER_SIM.sample_state(int(seed))
    return np.asarray(response, dtype=np.float32), np.asarray(m_current, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_seed(spec: dict) -> np.ndarray:
    start, stop = int(spec["start"]), int(spec["stop"])
    if stop <= start:
        raise ValueError(f"Invalid seed range [{start}, {stop})")
    seeds = np.arange(start, stop, dtype=np.int64)
    overlap = np.intersect1d(seeds, np.asarray(PUBLIC_EVAL_SEEDS))
    if overlap.size:
        raise ValueError(f"Seed range [{start}, {stop}) contains public eval seeds {overlap[:5]}")
    return seeds


def split_seeds_from_config(cfg: dict) -> dict[str, np.ndarray]:
    seed_cfg = cfg.get("data", cfg)
    return {name: split_seed(seed_cfg[name]) for name in SPLITS}


def build_dataset(
    data_cfg: dict,
    workers: int = 12,
    overwrite: bool = False,
    progress_every: int = 2000,
) -> dict:
    """Generate and cache all splits; returns the manifest."""
    dataset_dir = Path(resolve(data_cfg.get("dataset_dir", DATASET_DIR)))
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; pass overwrite=True to regenerate"
        )

    splits = split_seeds_from_config(data_cfg)
    simulator = load_simulator()
    response_shape = tuple(int(v) for v in simulator.response_shape)
    correction_dim = int(simulator.correction_dim)

    manifest: dict = {
        "created_unix": time.time(),
        "dataset_dir": str(dataset_dir),
        "response_shape": list(response_shape),
        "correction_dim": correction_dim,
        "dtype": "float32",
        "public_eval_seed_overlap": False,
        "splits": {},
        "simulator_calls": 0,
        "wall_seconds_total": 0.0,
    }

    total_start = time.perf_counter()
    for name in SPLITS:
        seeds = splits[name]
        count = len(seeds)
        responses = np.empty((count, *response_shape), dtype=np.float32)
        m_current = np.empty((count, correction_dim), dtype=np.float32)

        split_start = time.perf_counter()
        if workers and workers > 1 and count > 1:
            context = mp.get_context("fork")
            with context.Pool(processes=int(workers), initializer=_worker_init) as pool:
                for idx, (response, vector) in enumerate(
                    pool.imap(_worker_state, seeds.tolist(), chunksize=16)
                ):
                    responses[idx] = response
                    m_current[idx] = vector
                    if progress_every and (idx + 1) % progress_every == 0:
                        elapsed = time.perf_counter() - split_start
                        print(
                            f"[dataset] {name}: {idx + 1}/{count} "
                            f"({(idx + 1) / elapsed:.1f} states/s)",
                            flush=True,
                        )
        else:
            if _WORKER_SIM is None:
                _worker_init()
            for idx, seed in enumerate(seeds):
                response, vector = _worker_state(int(seed))
                responses[idx] = response
                m_current[idx] = vector
        elapsed = time.perf_counter() - split_start

        paths = {
            "responses": dataset_dir / f"{name}_responses.npy",
            "m_current": dataset_dir / f"{name}_m_current.npy",
            "seeds": dataset_dir / f"{name}_seeds.npy",
        }
        np.save(paths["responses"], responses)
        np.save(paths["m_current"], m_current)
        np.save(paths["seeds"], seeds)

        manifest["splits"][name] = {
            "start": int(seeds[0]),
            "stop": int(seeds[-1]) + 1,
            "count": int(count),
            "files": {key: path.name for key, path in paths.items()},
            "bytes": int(sum(path.stat().st_size for path in paths.values())),
            "sha256": {key: _sha256(path) for key, path in paths.items()},
            "generation_seconds": float(elapsed),
            "states_per_second": float(count / max(elapsed, 1e-9)),
        }
        manifest["simulator_calls"] += int(count)
        print(
            f"[dataset] {name}: {count} states in {elapsed:.1f}s "
            f"({bytes_to_human(manifest['splits'][name]['bytes'])})",
            flush=True,
        )

    manifest["wall_seconds_total"] = float(time.perf_counter() - total_start)
    manifest["cache_bytes_total"] = int(
        sum(split["bytes"] for split in manifest["splits"].values())
    )
    manifest["states_per_second"] = float(
        manifest["simulator_calls"] / max(manifest["wall_seconds_total"], 1e-9)
    )
    save_json(manifest_path, manifest)
    return manifest


class CM19ResponseDataset(Dataset):
    """Memmap-backed ``(response, m_current)`` dataset with an optional slice."""

    def __init__(
        self,
        dataset_dir: str = DATASET_DIR,
        split: str = "train",
        limit: int | None = None,
        offset: int = 0,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
        self.dataset_dir = Path(resolve(dataset_dir))
        manifest = load_json(self.dataset_dir / "manifest.json")
        meta = manifest["splits"][split]
        self.split = split
        self.seeds = np.load(self.dataset_dir / meta["files"]["seeds"])
        self.responses = np.load(self.dataset_dir / meta["files"]["responses"], mmap_mode="r")
        self.m_current = np.load(self.dataset_dir / meta["files"]["m_current"], mmap_mode="r")
        if not (len(self.seeds) == len(self.responses) == len(self.m_current)):
            raise RuntimeError("Cached split arrays have inconsistent lengths")
        stop = len(self.seeds) if limit is None else min(len(self.seeds), offset + int(limit))
        self._indices = np.arange(int(offset), stop, dtype=np.int64)
        self.response_shape = tuple(self.responses.shape[1:])
        self.correction_dim = int(self.m_current.shape[1])

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, index: int):
        row = int(self._indices[index])
        response = torch.from_numpy(np.array(self.responses[row], dtype=np.float32))
        state = torch.from_numpy(np.array(self.m_current[row], dtype=np.float32))
        return response, state

    def seeds_slice(self) -> np.ndarray:
        return self.seeds[self._indices]


def validate_cache(dataset_dir: str = DATASET_DIR, verify_hashes: bool = False) -> dict:
    """Structural + numerical integrity check of the cached dataset."""
    dataset_dir = Path(resolve(dataset_dir))
    manifest = load_json(dataset_dir / "manifest.json")
    simulator = load_simulator()
    report: dict = {"splits": {}, "ok": True}
    all_seeds: list[np.ndarray] = []
    for name in SPLITS:
        meta = manifest["splits"][name]
        seeds = np.load(dataset_dir / meta["files"]["seeds"])
        responses = np.load(dataset_dir / meta["files"]["responses"], mmap_mode="r")
        m_current = np.load(dataset_dir / meta["files"]["m_current"], mmap_mode="r")
        problems = []
        if len(seeds) != meta["count"]:
            problems.append("seed count mismatch")
        if tuple(responses.shape) != (meta["count"], *manifest["response_shape"]):
            problems.append(f"response shape {responses.shape}")
        if tuple(m_current.shape) != (meta["count"], manifest["correction_dim"]):
            problems.append(f"state shape {m_current.shape}")
        if np.intersect1d(seeds, np.asarray(PUBLIC_EVAL_SEEDS)).size:
            problems.append("contains public evaluation seeds")
        if not np.isfinite(responses[:: max(1, len(seeds) // 32)]).all():
            problems.append("non-finite responses")
        if not np.isfinite(m_current).all():
            problems.append("non-finite states")
        if np.any(m_current < simulator.lower_bounds - 1e-6) or np.any(
            m_current > simulator.upper_bounds + 1e-6
        ):
            problems.append("state outside published bounds")
        hashes = {}
        if verify_hashes:
            for key in ("responses", "m_current", "seeds"):
                hashes[key] = _sha256(dataset_dir / meta["files"][key])
                if hashes[key] != meta["sha256"][key]:
                    problems.append(f"{key} sha256 mismatch")
        report["splits"][name] = {
            "count": int(len(seeds)),
            "seed_range": [int(seeds[0]), int(seeds[-1]) + 1],
            "bytes": int(meta["bytes"]),
            "problems": problems,
            "hashes_verified": bool(verify_hashes),
        }
        if problems:
            report["ok"] = False
        all_seeds.append(seeds)
    overlap = np.intersect1d(all_seeds[0], all_seeds[1]).size + np.intersect1d(
        all_seeds[1], all_seeds[2]
    ).size
    report["split_overlap"] = int(overlap)
    if overlap:
        report["ok"] = False
    report["public_eval_seed_overlap"] = False
    return report


def build_normalization_stats(
    rep_cfg: dict,
    dataset_dir: str = DATASET_DIR,
    split: str = "train",
    batch_size: int = 128,
    max_samples: int | None = None,
) -> dict:
    """Estimate normalization statistics from the *training* split only."""
    import torch

    from .dsp import build_representation
    from .normalization import estimate_stats

    dataset = CM19ResponseDataset(dataset_dir, split=split, limit=max_samples)
    num_freq_points = int(dataset.response_shape[1])
    representation = build_representation(rep_cfg, num_freq_points)
    representation.eval()

    def batches():
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        with torch.no_grad():
            for response, _ in loader:
                yield representation(response)

    stats = estimate_stats(batches())
    stats["num_samples"] = np.asarray(len(dataset), dtype=np.float64)
    return stats
