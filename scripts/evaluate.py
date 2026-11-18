"""Evaluate one route on validation / internal-test seeds.

Usage:
    .venv/bin/python scripts/evaluate.py --config configs/route_delay_fno.json \
        --checkpoint runs/xxx/best.pt --split internal_test --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.config import load_config  # noqa: E402
from solution.data import CM19ResponseDataset  # noqa: E402
from solution.evaluator import evaluate_identification, evaluate_one_shot  # noqa: E402
from solution.models import build_model  # noqa: E402
from solution.simulator import load_simulator  # noqa: E402
from solution.trainer import load_checkpoint_into  # noqa: E402
from solution.utils import load_json, save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--split",
        choices=("validation", "internal_test"),
        default="internal_test",
        help="evaluation is restricted to validation/internal-test seeds",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--identification-limit", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    simulator = load_simulator()
    model = build_model(cfg, simulator)
    if args.checkpoint:
        info = load_checkpoint_into(model, args.checkpoint)
        print(f"[eval] loaded checkpoint {args.checkpoint} ({info})")

    dataset = CM19ResponseDataset(
        cfg["data"]["dataset_dir"], split=args.split, limit=args.identification_limit
    )
    identification = evaluate_identification(
        model, dataset, device=args.device, batch_size=args.batch_size
    )
    seeds = dataset.seeds_slice()[: args.limit]
    summary, rows = evaluate_one_shot(
        model,
        simulator,
        seeds,
        device=args.device,
        batch_size=args.batch_size,
        progress=True,
    )
    payload = {
        "route": cfg.get("route"),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "identification": identification,
        "one_shot": summary,
        "rows": rows,
    }
    if args.output:
        save_json(args.output, payload)
        print(f"[eval] saved {args.output}")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
