"""Loss functions and physical-parameter error metrics."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def state_loss(
    pred_u: torch.Tensor,
    target_u: torch.Tensor,
    kind: str = "huber",
    delta: float = 1.0,
) -> torch.Tensor:
    """Supervised loss on the normalized state ``u in [-1, 1]``."""
    key = str(kind).lower()
    if key == "huber":
        return F.huber_loss(pred_u, target_u, delta=float(delta))
    if key == "mse":
        return F.mse_loss(pred_u, target_u)
    if key == "l1":
        return F.l1_loss(pred_u, target_u)
    raise ValueError(f"Unknown loss '{kind}'")


def state_error_metrics(
    pred_m: torch.Tensor | np.ndarray,
    target_m: torch.Tensor | np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Physical-space errors in the original matrix-parameter scale.

    Returns aggregate absolute-error quantiles plus per-parameter MAE/P95 and
    the per-state L2/L-infinity norms of the residual vector.
    """
    if torch.is_tensor(pred_m):
        pred = pred_m.detach().float().cpu()
        target = (
            target_m.detach().float().cpu()
            if torch.is_tensor(target_m)
            else torch.as_tensor(np.asarray(target_m), dtype=torch.float32)
        )
    else:
        pred = torch.as_tensor(np.asarray(pred_m), dtype=torch.float32)
        target = torch.as_tensor(np.asarray(target_m), dtype=torch.float32)

    difference = pred - target
    error = difference.abs()
    flat = error.flatten()
    per_state_l2 = torch.linalg.vector_norm(difference, dim=1)
    per_state_linf = error.max(dim=1).values
    return {
        "mae": float(error.mean()),
        "rmse": float(torch.sqrt((difference**2).mean())),
        "median_abs": float(torch.quantile(flat, 0.50)),
        "p90_abs": float(torch.quantile(flat, 0.90)),
        "p95_abs": float(torch.quantile(flat, 0.95)),
        "p99_abs": float(torch.quantile(flat, 0.99)),
        "max_abs": float(error.max()),
        "l2_mean": float(per_state_l2.mean()),
        "l2_p95": float(torch.quantile(per_state_l2, 0.95)),
        "l2_max": float(per_state_l2.max()),
        "linf_mean": float(per_state_linf.mean()),
        "linf_p95": float(torch.quantile(per_state_linf, 0.95)),
        "linf_max": float(per_state_linf.max()),
        "per_parameter_mae": error.mean(dim=0).numpy(),
        "per_parameter_p95": torch.quantile(error, 0.95, dim=0).numpy(),
    }


def aggregate_metric_dicts(metrics: list[dict]) -> dict:
    keys = [
        key
        for key in metrics[0]
        if key not in ("per_parameter_mae", "per_parameter_p95")
    ]
    aggregate = {
        key: float(np.mean([row[key] for row in metrics])) for key in keys
    }
    aggregate["per_parameter_mae"] = np.mean(
        np.stack([row["per_parameter_mae"] for row in metrics], axis=0), axis=0
    )
    aggregate["per_parameter_p95"] = np.mean(
        np.stack([row["per_parameter_p95"] for row in metrics], axis=0), axis=0
    )
    return aggregate

