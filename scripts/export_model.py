"""Export a trained route to standalone TorchScript and verify it on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.config import load_config  # noqa: E402
from solution.export import check_exported_model, export_torchscript  # noqa: E402
from solution.models import build_model  # noqa: E402
from solution.simulator import load_simulator  # noqa: E402
from solution.trainer import load_checkpoint_into  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/export")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--check-seed", type=int, default=777777)
    args = parser.parse_args()

    simulator = load_simulator()
    cfg = load_config(args.config)
    model = build_model(cfg, simulator)
    info = load_checkpoint_into(model, args.checkpoint)
    print(f"[export] loaded {args.checkpoint} ({info})")

    result = export_torchscript(
        model,
        simulator,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    check = check_exported_model(result["model_path"], simulator, seeds=(args.check_seed,))
    result["apply_check"] = check
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
