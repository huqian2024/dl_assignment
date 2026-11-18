"""Collect residual dataset and failure analysis for a trained model.

Usage:
    .venv/bin/python scripts/analyze_model.py \
        --config configs/route_delay_fno_r3_raw_tukey.json \
        --checkpoint runs/route3/r3/best_pass.pt --split validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.analysis import collect_residuals, save_residual_analysis  # noqa: E402
from solution.config import load_config  # noqa: E402
from solution.data import CM19ResponseDataset  # noqa: E402
from solution.evaluator import evaluate_identification  # noqa: E402
from solution.models import build_model  # noqa: E402
from solution.simulator import load_simulator  # noqa: E402
from solution.trainer import load_checkpoint_into  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", choices=("validation", "internal_test"), default="validation"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", default="artifacts/analysis")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--identification-limit", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    simulator = load_simulator()
    model = build_model(cfg, simulator)
    checkpoint_info = load_checkpoint_into(model, args.checkpoint)

    dataset = CM19ResponseDataset(
        cfg["data"]["dataset_dir"], split=args.split, limit=args.limit
    )
    seeds = dataset.seeds_slice()
    arrays = collect_residuals(
        model,
        simulator,
        seeds,
        device=args.device,
        batch_size=args.batch_size,
    )
    tag = args.tag or f"{Path(args.checkpoint).parent.name}_{args.split}"
    identification = evaluate_identification(
        model,
        CM19ResponseDataset(
            cfg["data"]["dataset_dir"],
            split=args.split,
            limit=args.identification_limit,
        ),
        device=args.device,
        batch_size=64,
    )
    summary = save_residual_analysis(
        arrays,
        simulator,
        output_dir=args.output_root,
        tag=tag,
        extra={
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint_info["epoch"],
            "split": args.split,
            "identification": identification,
        },
    )
    print(
        json.dumps(
            {
                "tag": tag,
                "states": summary["num_states"],
                "pass_count": summary["pass_count"],
                "pass_rate": summary["pass_rate"],
                "max_miss_db": summary["max_miss_db"],
                "error_metrics": {
                    key: summary["error_metrics"][key]
                    for key in (
                        "mae",
                        "rmse",
                        "median_abs",
                        "p90_abs",
                        "p95_abs",
                        "p99_abs",
                        "max_abs",
                        "l2_mean",
                        "linf_mean",
                    )
                },
                "hardest": summary["hardest_parameters_by_mae"][:5],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
