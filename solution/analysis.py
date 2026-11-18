"""Residual collection and failure analysis for a trained one-shot model.

The saved residual dataset is the key artifact for later parameter-sensitivity
studies, weighted losses and robust anchor refinement.  Nothing here is used
during training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .losses import state_error_metrics
from .simulator import requirement_infos
from .utils import get_device, resolve, save_json


@torch.no_grad()
def collect_residuals(
    model,
    simulator,
    seeds,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Protocol-faithful one-shot rollout that also stores state residuals."""
    device = get_device(device) if isinstance(device, str) else device
    model = model.to(device).eval()
    seeds = np.asarray(list(seeds), dtype=np.int64)
    anchor = model.anchor.detach().float().cpu().numpy()[0].astype(np.float64)
    lower = np.asarray(simulator.lower_bounds, dtype=np.float64)
    upper = np.asarray(simulator.upper_bounds, dtype=np.float64)
    requirements = requirement_infos(simulator)
    num = len(seeds)
    dim = int(anchor.size)

    m_true = np.zeros((num, dim), dtype=np.float64)
    m_pred = np.zeros_like(m_true)
    error = np.zeros_like(m_true)
    delta = np.zeros_like(m_true)
    corrected = np.zeros_like(m_true)
    margins = np.zeros((num, len(requirements)), dtype=np.float64)
    passed = np.zeros(num, dtype=bool)
    max_miss = np.zeros(num, dtype=np.float64)
    clipped_rate = np.zeros(num, dtype=np.float64)

    for begin in range(0, num, int(batch_size)):
        chunk = seeds[begin:begin + int(batch_size)]
        states = [simulator.sample_state(int(seed)) for seed in chunk]
        current = np.stack([state[0] for state in states], axis=0).astype(np.float64)
        responses = np.stack([state[1] for state in states], axis=0)
        delta_chunk = (
            model(torch.from_numpy(responses).to(device)).float().cpu().numpy().astype(np.float64)
        )
        pred_chunk = anchor[None, :] - delta_chunk
        for idx, seed in enumerate(chunk):
            row = begin + idx
            current_row = current[idx]
            delta_row = delta_chunk[idx]
            result = simulator.apply_correction(current_row, delta_row)
            raw = current_row + delta_row
            clipped = np.clip(raw, lower, upper)
            details = result.details
            for req_idx, req in enumerate(requirements):
                value = float(details[f"{req.name}_value_db"])
                threshold = float(req.threshold_db)
                margins[row, req_idx] = (
                    threshold - value if req.operator == "<=" else value - threshold
                )
            m_true[row] = current_row
            m_pred[row] = pred_chunk[idx]
            error[row] = pred_chunk[idx] - current_row
            delta[row] = delta_row
            corrected[row] = clipped
            passed[row] = bool(result.all_specs_pass)
            max_miss[row] = float(result.max_spec_miss_db)
            clipped_rate[row] = float(np.mean(clipped != raw))

    return {
        "seeds": seeds,
        "m_true": m_true,
        "m_pred": m_pred,
        "error": error,
        "delta": delta,
        "corrected": corrected,
        "margins": margins,
        "requirement_names": np.asarray([req.name for req in requirements]),
        "passed": passed,
        "max_miss_db": max_miss,
        "clipped_entry_rate": clipped_rate,
    }


def summarize_residuals(arrays: dict[str, np.ndarray], simulator) -> dict:
    """Aggregate error, difficulty and failure statistics from residual arrays."""
    m_true = arrays["m_true"]
    m_pred = arrays["m_pred"]
    error = arrays["error"]
    passed = arrays["passed"].astype(bool)
    max_miss = arrays["max_miss_db"]
    names = [str(name) for name in arrays["requirement_names"]]

    metrics = state_error_metrics(m_pred, m_true)
    lower = np.asarray(simulator.lower_bounds, dtype=np.float64)
    upper = np.asarray(simulator.upper_bounds, dtype=np.float64)
    half_range = np.maximum(0.5 * (upper - lower), 1e-6)

    per_param_mae = np.asarray(metrics["per_parameter_mae"], dtype=np.float64)
    per_param_p95 = np.asarray(metrics["per_parameter_p95"], dtype=np.float64)
    per_param_mae_u = per_param_mae / half_range
    per_param_p95_u = per_param_p95 / half_range

    failing = ~passed
    if failing.any() and passed.any():
        mae_fail = np.abs(error[failing]).mean(axis=0)
        mae_pass = np.abs(error[passed]).mean(axis=0)
        failure_ratio = mae_fail / np.maximum(mae_pass, 1e-12)
    else:
        mae_fail = np.zeros_like(per_param_mae)
        mae_pass = np.zeros_like(per_param_mae)
        failure_ratio = np.zeros_like(per_param_mae)

    per_state_l2 = np.linalg.norm(error, axis=1)
    per_state_linf = np.abs(error).max(axis=1)
    if len(per_state_linf) > 2 and np.std(per_state_linf) > 0 and np.std(max_miss) > 0:
        miss_vs_linf_corr = float(np.corrcoef(per_state_linf, max_miss)[0, 1])
    else:
        miss_vs_linf_corr = float("nan")

    hardest_by_mae = np.argsort(-per_param_mae)[:10]
    hardest_by_failure_ratio = np.argsort(-failure_ratio)[:10]

    worst_rows = []
    for row in np.argsort(-max_miss)[:20]:
        top_params = np.argsort(-np.abs(error[row]))[:5]
        worst_rows.append(
            {
                "seed": int(arrays["seeds"][row]),
                "passed": bool(passed[row]),
                "max_miss_db": float(max_miss[row]),
                "l2": float(per_state_l2[row]),
                "linf": float(per_state_linf[row]),
                "largest_error_parameters": [
                    {
                        "index": int(idx),
                        "error": float(error[row, idx]),
                        "abs_error": float(abs(error[row, idx])),
                    }
                    for idx in top_params
                ],
                "margins_db": {
                    name: float(arrays["margins"][row, idx])
                    for idx, name in enumerate(names)
                },
                "clipped_entry_rate": float(arrays["clipped_entry_rate"][row]),
            }
        )

    per_requirement = {}
    for idx, name in enumerate(names):
        per_requirement[name] = {
            "pass_rate": float(np.mean(arrays["margins"][:, idx] > 0.0)),
            "mean_margin_db": float(np.mean(arrays["margins"][:, idx])),
            "min_margin_db": float(np.min(arrays["margins"][:, idx])),
        }

    return {
        "num_states": int(len(m_true)),
        "pass_count": int(passed.sum()),
        "pass_rate": float(passed.mean()),
        "max_miss_db": {
            "mean": float(np.mean(max_miss)),
            "median": float(np.median(max_miss)),
            "p90": float(np.quantile(max_miss, 0.90)),
            "worst": float(np.max(max_miss)),
        },
        "error_metrics": {
            key: float(value)
            for key, value in metrics.items()
            if key not in ("per_parameter_mae", "per_parameter_p95")
        },
        "per_parameter_mae": per_param_mae.tolist(),
        "per_parameter_p95": per_param_p95.tolist(),
        "per_parameter_mae_over_half_range": per_param_mae_u.tolist(),
        "per_parameter_p95_over_half_range": per_param_p95_u.tolist(),
        "hardest_parameters_by_mae": [
            {
                "index": int(idx),
                "mae": float(per_param_mae[idx]),
                "p95": float(per_param_p95[idx]),
                "mae_over_half_range": float(per_param_mae_u[idx]),
                "half_range": float(half_range[idx]),
            }
            for idx in hardest_by_mae
        ],
        "failure_enrichment": {
            "per_parameter_mae_fail": mae_fail.tolist(),
            "per_parameter_mae_pass": mae_pass.tolist(),
            "ratio_fail_over_pass": failure_ratio.tolist(),
            "top_parameters": [
                {
                    "index": int(idx),
                    "mae_fail": float(mae_fail[idx]),
                    "mae_pass": float(mae_pass[idx]),
                    "ratio": float(failure_ratio[idx]),
                }
                for idx in hardest_by_failure_ratio
            ],
        },
        "max_miss_vs_linf_correlation": miss_vs_linf_corr,
        "per_requirement": per_requirement,
        "worst_states": worst_rows,
    }


def _plot_diagnostics(arrays: dict[str, np.ndarray], summary: dict, target_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    per_mae = np.asarray(summary["per_parameter_mae"])
    per_p95 = np.asarray(summary["per_parameter_p95"])
    fig, ax = plt.subplots(figsize=(12, 4.0))
    indices = np.arange(per_mae.size)
    ax.bar(indices - 0.2, per_mae, width=0.4, label="MAE")
    ax.bar(indices + 0.2, per_p95, width=0.4, label="P95 |error|")
    ax.set(xlabel="parameter index", ylabel="physical absolute error", title="Per-parameter error")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    path = target_dir / "per_parameter_error.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(str(path))

    error = arrays["error"]
    l2 = np.linalg.norm(error, axis=1)
    max_miss = arrays["max_miss_db"]
    passed = arrays["passed"].astype(bool)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(l2[passed], max_miss[passed], s=9, alpha=0.55, label="pass")
    ax.scatter(l2[~passed], max_miss[~passed], s=9, alpha=0.55, label="fail")
    ax.set(xlabel="||error||_2", ylabel="max specification miss (dB)", title="Residual vs one-shot miss")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = target_dir / "error_vs_miss.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(max_miss, bins=40)
    ax.set(xlabel="max specification miss (dB)", ylabel="states", title="One-shot miss distribution")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    path = target_dir / "max_miss_histogram.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(str(path))
    return paths


def save_residual_analysis(
    arrays: dict[str, np.ndarray],
    simulator,
    output_dir: str,
    tag: str,
    extra: dict | None = None,
) -> dict:
    """Persist residual arrays, aggregate statistics and diagnostic plots."""
    output = Path(resolve(output_dir)) / tag
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "residuals.npz", **arrays)
    summary = summarize_residuals(arrays, simulator)
    summary["tag"] = tag
    summary["residual_file"] = str(output / "residuals.npz")
    summary["plots"] = _plot_diagnostics(arrays, summary, output)
    if extra:
        summary["extra"] = extra
    save_json(output / "summary.json", summary)
    return summary
