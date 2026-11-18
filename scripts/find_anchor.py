"""Search for a compliant anchor state with CEM + local refinement.

Usage:
    .venv/bin/python scripts/find_anchor.py --workers 12 --iterations 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.anchor_search import CEMAnchorSearch  # noqa: E402
from solution.config import ANCHOR_SEARCH_DEFAULTS, deep_merge, load_config  # noqa: E402
from solution.utils import load_json, resolve, save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional JSON with search settings")
    parser.add_argument("--route-config", default=None, help="optional route config (data only)")
    parser.add_argument("--output", default="artifacts/common/anchor.npz")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--init-from", default=None, help="resume from an existing anchor npz")
    parser.add_argument("--objective", default=None)
    parser.add_argument("--early-stop-margin", type=float, default=None)
    parser.add_argument("--local-refine-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = dict(ANCHOR_SEARCH_DEFAULTS)
    if args.route_config:
        route_cfg = load_config(args.route_config)
        data_cfg = route_cfg.get("data", {})
        if "train" in data_cfg:
            cfg["data"] = data_cfg
    if args.config:
        cfg = deep_merge(cfg, load_json(resolve(args.config)))
    for name in (
        "seed",
        "population",
        "iterations",
        "objective",
        "early_stop_margin",
        "local_refine_steps",
    ):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    cfg["workers"] = int(args.workers)
    cfg["output"] = args.output
    save_json(resolve(args.output).with_suffix(".search_config.json"), cfg)

    init_u = None
    if args.init_from:
        import numpy as np

        with np.load(resolve(args.init_from)) as archive:
            init_u = np.asarray(archive["u_star"], dtype=np.float64)
        print(f"[anchor] warm-starting from {args.init_from}")

    search = CEMAnchorSearch(cfg, output_path=args.output)
    result = search.run(init_u=init_u)
    print(json.dumps(result, indent=2))
    print(f"[anchor] saved to {resolve(args.output)}")
    return 0 if result["compliant"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
