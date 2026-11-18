"""TorchScript export and verification for the one-shot mapping.

The exported module must be a standalone CPU-loadable artifact: all
normalization statistics, windows and the anchor live in registered buffers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .simulator import load_simulator
from .utils import count_parameters, resolve, save_json


def _check_output(tensor: torch.Tensor, batch: int) -> dict:
    if tuple(tensor.shape) != (batch, 41):
        raise ValueError(f"Expected output ({batch},41), got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"Expected float32 output, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Output contains NaN/Inf")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "abs_max": float(tensor.abs().max()),
        "abs_mean": float(tensor.abs().mean()),
    }


def export_torchscript(
    model,
    simulator,
    output_dir: str,
    filename: str = "s_to_delta_m.pt",
    metadata_name: str = "metadata.json",
    seed: int = 99999,
    batch_sizes: tuple[int, ...] = (1, 4),
) -> dict:
    """Trace, save, reload and numerically verify the model on CPU."""
    output_dir = Path(resolve(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    _, example_response = simulator.sample_state(int(seed))
    example = torch.from_numpy(np.asarray(example_response, dtype=np.float32)[None])

    with torch.no_grad():
        eager = {1: model(example)}
        eager[4] = model(example.repeat(4, 1, 1))
    checks = {batch: _check_output(value, batch) for batch, value in eager.items()}

    try:
        traced = torch.jit.trace(model, example, check_trace=True)
        export_method = "trace"
    except Exception as trace_error:  # pragma: no cover - fallback path
        try:
            traced = torch.jit.script(model)
            export_method = "script"
        except Exception as script_error:
            raise RuntimeError(
                f"Both torch.jit.trace and torch.jit.script failed. "
                f"trace: {trace_error}; script: {script_error}"
            ) from script_error

    model_path = output_dir / filename
    traced.save(str(model_path))
    loaded = torch.jit.load(str(model_path), map_location="cpu").eval()
    verifications = {}
    with torch.no_grad():
        for batch in batch_sizes:
            x = example.repeat(batch, 1, 1)
            reference = eager.get(batch, model(x))
            exported = loaded(x)
            verifications[int(batch)] = {
                "max_abs_diff": float((reference - exported).abs().max()),
                "finite": bool(torch.isfinite(exported).all()),
                "shape": list(exported.shape),
            }

    metadata = {
        "input_shape": [None, 4, 2001],
        "output_shape": [None, 41],
        "channel_order": list(simulator.response_channels),
        "route": getattr(model, "route_name", type(model).__name__),
        "seed": int(seed),
        "parameter_count": int(count_parameters(model)),
        "torch_version": torch.__version__,
        "export_method": export_method,
        "batch_sizes_checked": [int(b) for b in batch_sizes],
        "verification": verifications,
        "forward_checks": checks,
        "buffers": sorted(name for name, _ in model.named_buffers()),
        "uses_simulator_in_forward": False,
        "normalization_embedded": True,
    }
    metadata_path = output_dir / metadata_name
    save_json(metadata_path, metadata)
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "export_method": export_method,
        "verification": verifications,
        "parameter_count": int(count_parameters(model)),
    }


def check_exported_model(
    model_path: str,
    simulator=None,
    seeds: tuple[int, ...] = (777777,),
) -> dict:
    """Load an exported artifact on CPU and apply one correction per seed."""
    simulator = simulator or load_simulator()
    loaded = torch.jit.load(str(resolve(model_path)), map_location="cpu").eval()
    rows = []
    for seed in seeds:
        current_m, response = simulator.sample_state(int(seed))
        with torch.no_grad():
            delta = loaded(torch.from_numpy(np.asarray(response, dtype=np.float32)[None]))
        delta_row = delta.numpy()[0]
        result = simulator.apply_correction(current_m, delta_row)
        rows.append(
            {
                "seed": int(seed),
                "output_shape": list(delta.shape),
                "finite": bool(torch.isfinite(delta).all()),
                "all_specs_pass": bool(result.all_specs_pass),
                "max_spec_miss_db": float(result.max_spec_miss_db),
            }
        )
    return {"loaded_from": str(model_path), "rows": rows}
