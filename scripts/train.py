"""Train one route (supports a tiny-overfit sanity mode).

Examples:
    .venv/bin/python scripts/train.py --config configs/route_freq_resnet.json --epochs 1
    .venv/bin/python scripts/train.py --config configs/route_freq_resnet.json --tiny-overfit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from solution.config import TRAIN_DEFAULTS, deep_merge, load_config  # noqa: E402
from solution.data import CM19ResponseDataset  # noqa: E402
from solution.models import build_model, model_config_summary  # noqa: E402
from solution.simulator import load_simulator  # noqa: E402
from solution.trainer import Trainer, load_checkpoint_into  # noqa: E402
from solution.utils import count_parameters  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--tiny-steps", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--one-shot-every", type=int, default=None)
    parser.add_argument("--one-shot-states", type=int, default=None)
    parser.add_argument("--one-shot-start-epoch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--stop-metric", choices=("mae", "loss"), default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    simulator = load_simulator()
    model = build_model(cfg, simulator)
    if args.init_checkpoint:
        info = load_checkpoint_into(model, args.init_checkpoint)
        print(f"[train] warm start from {args.init_checkpoint}: {info}")

    train_cfg = deep_merge(TRAIN_DEFAULTS, cfg.get("train", {}))
    train_cfg["epochs"] = int(args.epochs or train_cfg.get("epochs", 1))
    if args.batch_size:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.lr:
        train_cfg["learning_rate"] = float(args.lr)
    if args.device:
        train_cfg["device"] = args.device
    if args.no_amp:
        train_cfg["amp"] = False
    if args.run_name:
        train_cfg["run_name"] = args.run_name
    if args.one_shot_every is not None:
        train_cfg["one_shot_every"] = int(args.one_shot_every)
    if args.one_shot_states is not None:
        train_cfg["one_shot_states"] = int(args.one_shot_states)
    if args.one_shot_start_epoch is not None:
        train_cfg["one_shot_start_epoch"] = int(args.one_shot_start_epoch)
    if args.patience is not None:
        train_cfg["early_stopping_patience"] = int(args.patience)
    if args.stop_metric is not None:
        train_cfg["early_stopping_metric"] = args.stop_metric
    if args.warmup_steps is not None:
        train_cfg["warmup_steps"] = int(args.warmup_steps)

    data_dir = cfg["data"]["dataset_dir"]
    limit_train = args.limit_train
    limit_val = args.limit_val
    if args.tiny_overfit:
        limit_train = limit_train or 64
        limit_val = limit_val or 32
        steps_per_epoch = max(1, math.ceil((limit_train or 64) / int(train_cfg["batch_size"])))
        train_cfg["epochs"] = max(1, math.ceil(args.tiny_steps / steps_per_epoch))
        train_cfg["scheduler"] = "cosine"
        print(
            f"[train] tiny-overfit: {limit_train} train / {limit_val} val samples, "
            f"{train_cfg['epochs']} epoch(s)"
        )

    train_ds = CM19ResponseDataset(data_dir, split="train", limit=limit_train)
    val_ds = CM19ResponseDataset(data_dir, split="validation", limit=limit_val)
    one_shot_seeds = None
    if simulator is not None and int(train_cfg.get("one_shot_every", 0) or 0) > 0:
        count = int(train_cfg.get("one_shot_states", 200))
        one_shot_seeds = val_ds.seeds_slice()[:count]
        print(
            f"[train] periodic one-shot evaluation: {len(one_shot_seeds)} validation states "
            f"every {train_cfg['one_shot_every']} epochs "
            f"(start epoch {train_cfg.get('one_shot_start_epoch', 0)})"
        )
    trainer = Trainer(
        model,
        model_cfg=cfg,
        train_cfg=train_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=args.output_dir,
        run_name=train_cfg.get("run_name"),
        device=str(train_cfg.get("device", "auto")),
        simulator=simulator,
        one_shot_seeds=one_shot_seeds,
    )
    print(json.dumps(
        {
            "route": cfg.get("route"),
            "trainable_parameters": count_parameters(model),
            "architecture": model_config_summary(model),
            "device": str(trainer.device),
            "amp": bool(trainer.use_amp),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        },
        indent=2,
    ))
    summary = trainer.fit()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
