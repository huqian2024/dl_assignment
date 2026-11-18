"""Build the cached dataset (all splits) and normalization statistics.

Usage:
    .venv/bin/python scripts/build_dataset.py --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Multiprocessing + BLAS threads oversubscribe badly for 19x19 matrix solves.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.config import load_config  # noqa: E402
from solution.data import build_dataset, build_normalization_stats, validate_cache  # noqa: E402
from solution.normalization import representation_stats_key, save_stats  # noqa: E402
from solution.utils import CONFIGS_DIR, bytes_to_human, load_json, resolve  # noqa: E402


def collect_representation_configs(config_paths: list[Path]) -> dict[str, dict]:
    """Unique representation configs keyed by their normalization stats key."""
    unique: dict[str, dict] = {}
    for path in config_paths:
        if not path.exists():
            continue
        cfg = load_json(path)
        reps = []
        if cfg.get("branches"):
            reps.extend(branch["representation"] for branch in cfg["branches"])
        else:
            reps.append(cfg["representation"])
        for rep in reps:
            unique.setdefault(representation_stats_key(rep), dict(rep))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIGS_DIR / "route_freq_resnet.json"))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--norm-configs", nargs="*", default=None)
    parser.add_argument("--max-norm-samples", type=int, default=None)
    parser.add_argument("--stats-file", default="artifacts/common/norm_stats.npz")
    parser.add_argument("--skip-norm-stats", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = dict(cfg["data"])
    if args.dataset_dir:
        data_cfg["dataset_dir"] = args.dataset_dir

    manifest_path = Path(resolve(data_cfg["dataset_dir"])) / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"[data] cache already present: {manifest_path} (use --overwrite to regenerate)")
        manifest = load_json(manifest_path)
    else:
        manifest = build_dataset(
            data_cfg, workers=args.workers, overwrite=args.overwrite
        )

    report = validate_cache(data_cfg["dataset_dir"], verify_hashes=args.verify_hashes)
    if not report["ok"]:
        print(json.dumps(report, indent=2))
        raise SystemExit("Dataset validation failed")

    stats_summary = {}
    if not args.skip_norm_stats:
        import glob

        patterns = args.norm_configs or ["configs/route_*.json"]
        config_paths: list[Path] = []
        for pattern in patterns:
            candidate = Path(pattern)
            search = str(candidate) if candidate.is_absolute() else str(Path(ROOT) / candidate)
            config_paths.extend(sorted(Path(p) for p in glob.glob(search) if Path(p).is_file()))
        reps = collect_representation_configs(config_paths)
        if not reps:
            raise SystemExit("No representation configs found for normalization stats")
        stats_path = resolve(args.stats_file)
        for key, rep_cfg in sorted(reps.items()):
            print(f"[norm] estimating statistics for '{key}'", flush=True)
            stats = build_normalization_stats(
                rep_cfg,
                dataset_dir=data_cfg["dataset_dir"],
                split="train",
                batch_size=128,
                max_samples=args.max_norm_samples,
            )
            save_stats(str(stats_path), key, stats)
            stats_summary[key] = {
                "channels": int(stats["per_channel_mean"].size),
                "positions": int(stats["position_mean"].shape[-1]),
                "samples": int(stats["num_samples"]),
            }

    summary = {
        "manifest": str(manifest_path),
        "splits": {
            name: {
                "seed_range": [meta["start"], meta["stop"]],
                "count": meta["count"],
                "bytes": meta["bytes"],
            }
            for name, meta in manifest["splits"].items()
        },
        "cache_bytes_total": manifest["cache_bytes_total"],
        "cache_human": bytes_to_human(manifest["cache_bytes_total"]),
        "simulator_calls": manifest["simulator_calls"],
        "generation_seconds": manifest["wall_seconds_total"],
        "states_per_second": manifest["states_per_second"],
        "validation": report,
        "norm_stats": stats_summary,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
