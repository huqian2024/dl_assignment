"""End-to-end smoke test for the stage: environment, data, four routes, export.

Usage:
    .venv/bin/python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from solution.config import TRAIN_DEFAULTS, deep_merge, load_config  # noqa: E402
from solution.data import CM19ResponseDataset, validate_cache  # noqa: E402
from solution.dsp import DelayRepresentation, complex_fft, complex_ifft  # noqa: E402
from solution.evaluator import evaluate_one_shot  # noqa: E402
from solution.export import export_torchscript  # noqa: E402
from solution.models import build_model, model_config_summary  # noqa: E402
from solution.simulator import load_simulator  # noqa: E402
from solution.trainer import Trainer  # noqa: E402
from solution.utils import (  # noqa: E402
    CONFIGS_DIR,
    count_parameters,
    cuda_sanity_check,
    describe_environment,
    get_device,
    save_json,
)

ROUTES = [
    "configs/route_freq_resnet.json",
    "configs/route_delay_resnet.json",
    "configs/route_delay_fno.json",
    "configs/route_dual_domain.json",
]


def _assert_no_simulator(model) -> None:
    for module in model.modules():
        name = type(module).__module__
        if "cm19_simulator" in name:
            raise AssertionError("Model must not embed the simulator")


def _first_last_train_loss(log_path: Path) -> tuple[float, float]:
    losses = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            record = json.loads(line)
            losses.append(float(record["train"]["loss"]))
    if not losses:
        return float("nan"), float("nan")
    return losses[0], losses[-1]


def _tiny_overfit(
    cfg: dict,
    simulator,
    data_dir: str,
    steps: int,
    output_dir: Path,
    device: str,
) -> dict:
    model = build_model(cfg, simulator)
    train_cfg = deep_merge(TRAIN_DEFAULTS, cfg.get("train", {}))
    train_cfg.update(
        {
            "epochs": 1,
            "batch_size": 16,
            "amp": True,
            "device": device,
            "output_dir": str(output_dir),
            "run_name": f"tiny_{cfg.get('route')}",
        }
    )
    train_ds = CM19ResponseDataset(data_dir, split="train", limit=48)
    val_ds = CM19ResponseDataset(data_dir, split="validation", limit=32)
    steps_per_epoch = math.ceil(len(train_ds) / 16)
    train_cfg["epochs"] = max(1, math.ceil(steps / steps_per_epoch))
    start = time.perf_counter()
    trainer = Trainer(
        model,
        model_cfg=cfg,
        train_cfg=train_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=str(output_dir),
        run_name=f"tiny_{cfg.get('route')}",
        device=device,
    )
    summary = trainer.fit()
    first, last = _first_last_train_loss(Path(summary["output_dir"]) / "log.jsonl")
    elapsed = time.perf_counter() - start
    return {
        "steps_budget": int(steps),
        "epochs": int(summary["epochs"]),
        "first_train_loss": first,
        "last_train_loss": last,
        "loss_ratio": (last / first) if first else float("nan"),
        "loss_decreased": bool(last < 0.8 * first) if first == first else False,
        "wall_time_seconds": elapsed,
        "best_checkpoint": summary["best_checkpoint"],
        "trainable_parameters": summary["trainable_parameters"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", nargs="*", default=ROUTES)
    parser.add_argument("--output-dir", default="runs/smoke")
    parser.add_argument("--tiny-steps", type=int, default=30)
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-tiny", action="store_true")
    parser.add_argument("--eval-states", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(ROOT) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = str(get_device(args.device))
    report: dict = {"env": describe_environment(), "device": device}

    print("[smoke] environment:", json.dumps(report["env"]))
    report["cuda_sanity"] = cuda_sanity_check()
    print("[smoke] cuda sanity:", report["cuda_sanity"])

    simulator = load_simulator()

    # --- DSP round trip -------------------------------------------------
    torch.manual_seed(0)
    re = torch.randn(2, 2001)
    im = torch.randn(2, 2001)
    h_re, h_im = complex_ifft(re, im)
    r_re, r_im = complex_fft(h_re, h_im)
    delay = DelayRepresentation(2001, window="tukey", concat_raw=True)
    delay_out = delay(torch.randn(2, 4, 2001))
    report["dsp"] = {
        "fft_ifft_max_abs_error": max(
            float((r_re - re).abs().max()), float((r_im - im).abs().max())
        ),
        "delay_shape": list(delay_out.shape),
        "delay_finite": bool(torch.isfinite(delay_out).all()),
    }
    print("[smoke] dsp:", report["dsp"])

    # --- dataset / anchor ----------------------------------------------
    data_dir = "cache/dataset"
    if not args.skip_dataset:
        manifest_path = Path(ROOT) / data_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"Missing dataset cache at {manifest_path}. "
                "Run scripts/build_dataset.py first."
            )
        validation = validate_cache(data_dir)
        if not validation["ok"]:
            raise SystemExit(f"Dataset validation failed: {json.dumps(validation, indent=2)}")
        report["dataset"] = validation
        print("[smoke] dataset ok:", {
            name: split["count"] for name, split in validation["splits"].items()
        })

    anchor_path = Path(ROOT) / "artifacts" / "common" / "anchor.npz"
    if not anchor_path.exists():
        raise SystemExit(f"Missing anchor at {anchor_path}. Run scripts/find_anchor.py first.")
    with np.load(anchor_path) as archive:
        anchor_min_margin = float(archive["min_margin"])
        anchor_margins = archive["margins"].astype(float).tolist()
    report["anchor"] = {
        "path": str(anchor_path),
        "compliant": bool(anchor_min_margin > 0.0),
        "min_margin_db": anchor_min_margin,
        "margins_db": anchor_margins,
    }
    print("[smoke] anchor:", {k: v for k, v in report["anchor"].items() if k != "margins_db"})

    guard_ok = False
    try:
        evaluate_one_shot(
            build_model(load_config(args.routes[0]), simulator),
            simulator,
            [10000],
            device="cpu",
        )
    except ValueError:
        guard_ok = True
    report["public_seed_guard"] = {"rejects_public_eval_seeds": guard_ok}
    if not guard_ok:
        raise SystemExit("Evaluator did not reject public evaluation seeds")

    # --- routes ---------------------------------------------------------
    report["routes"] = {}
    for route_path in args.routes:
        name = Path(route_path).stem
        print(f"\n[smoke] === {name} ===")
        cfg = load_config(route_path)
        model = build_model(cfg, simulator)
        _assert_no_simulator(model)
        params = count_parameters(model)
        x1 = torch.randn(1, 4, 2001)
        x4 = torch.randn(4, 4, 2001)
        entry: dict = {
            "config": route_path,
            "architecture": model_config_summary(model),
            "trainable_parameters": params,
        }
        with torch.no_grad():
            y1 = model(x1)
            y4 = model(x4)
        entry["cpu_forward"] = {
            "B1": list(y1.shape),
            "B4": list(y4.shape),
            "finite": bool(torch.isfinite(y1).all() and torch.isfinite(y4).all()),
        }
        gpu_model = model.to(device)
        gpu_model.train()
        gpu_out = gpu_model(x1.to(device))
        gpu_out.square().mean().backward()
        grads_finite = all(
            (parameter.grad is None) or bool(torch.isfinite(parameter.grad).all())
            for parameter in gpu_model.parameters()
        )
        entry["gpu_forward_backward"] = {
            "device": device,
            "finite_output": bool(torch.isfinite(gpu_out).all()),
            "finite_grads": bool(grads_finite),
        }
        gpu_model.zero_grad(set_to_none=True)
        gpu_model.to("cpu").eval()

        export_result = export_torchscript(
            model, simulator, output_dir=str(output_dir / "exports" / name), seed=99999
        )
        entry["export"] = {
            "method": export_result["export_method"],
            "verification": export_result["verification"],
            "parameter_count": export_result["parameter_count"],
        }

        eval_seeds = CM19ResponseDataset(
            data_dir, split="validation", limit=args.eval_states
        ).seeds_slice()
        eval_summary, _ = evaluate_one_shot(
            model, simulator, eval_seeds, device=device, batch_size=args.eval_states
        )
        entry["one_shot_untrained"] = {
            "num_states": eval_summary["num_states"],
            "pass_rate": eval_summary["pass_rate"],
            "mean_max_miss_db": eval_summary["mean_max_miss_db"],
            "clipped_entry_rate_mean": eval_summary["clipped_entry_rate_mean"],
        }
        del gpu_model

        if not args.skip_tiny and (Path(ROOT) / data_dir / "manifest.json").exists():
            entry["tiny_overfit"] = _tiny_overfit(
                cfg, simulator, data_dir, args.tiny_steps, output_dir / "tiny", device
            )
            print("[smoke] tiny-overfit:", entry["tiny_overfit"])

        report["routes"][name] = entry
        print(
            f"[smoke] {name}: params={params:,} export={entry['export']['method']} "
            f"tiny_loss_ratio={entry.get('tiny_overfit', {}).get('loss_ratio')}"
        )

    checks = {
        "all_routes_forward": all(
            entry["cpu_forward"]["finite"] for entry in report["routes"].values()
        ),
        "all_routes_backward": all(
            entry["gpu_forward_backward"]["finite_grads"]
            for entry in report["routes"].values()
        ),
        "all_exports_ok": all(
            all(v["max_abs_diff"] < 1e-4 for v in entry["export"]["verification"].values())
            for entry in report["routes"].values()
        ),
        "parameter_budget": all(
            entry["trainable_parameters"] < 20_000_000
            for entry in report["routes"].values()
        ),
        "tiny_overfit_loss_decreased": all(
            entry.get("tiny_overfit", {}).get("loss_decreased", True)
            for entry in report["routes"].values()
        ),
        "public_eval_seeds_rejected": bool(report["public_seed_guard"]["rejects_public_eval_seeds"]),
    }
    report["checks"] = checks
    summary_path = output_dir / "stage_summary.json"
    save_json(summary_path, report)
    print(f"\n[smoke] summary written to {summary_path}")
    print("[smoke] checks:", json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
