"""Differential (pairwise) control formulation for CM19.

Core idea: instead of inverting the full 41-D state, learn the observable
parameter changes that explain a response difference:

    Delta z = V_K^T (S_j - S_i)        (response latent difference)
    Delta u = u_j - u_i                (bounds-normalized parameter difference)
    Delta z ~= A Delta u               (data-driven local Jacobian)
    A = U S V^T                        (observable parameter subspace V_r)

No new states are generated for training; only the cached train split is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .simulator import requirement_infos
from .utils import resolve, save_json


# ---------------------------------------------------------------------------
# Response PCA (train split only)
# ---------------------------------------------------------------------------

def flatten_responses(responses: np.ndarray) -> np.ndarray:
    return np.asarray(responses, dtype=np.float32).reshape(responses.shape[0], -1)


def pca_response(
    responses: np.ndarray,
    k_max: int = 512,
    device: str | torch.device = "cuda",
    seed: int = 2026,
    niter: int = 6,
) -> dict[str, np.ndarray]:
    """Randomized PCA of flattened responses; returns mean/basis/spectrum.

    The mean and basis are estimated from the provided array only (callers pass
    the train split).  Component signs are fixed deterministically.
    """
    torch.manual_seed(int(seed))
    device = torch.device(device)
    X = torch.as_tensor(flatten_responses(responses), device=device)
    mean = X.mean(dim=0)
    centered = X - mean
    k_max = int(min(k_max, min(centered.shape) - 1))
    _, singular, basis = torch.pca_lowrank(
        centered, q=k_max, center=False, niter=int(niter)
    )
    # deterministic sign: largest-|.| entry of each component is positive
    pivot = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivot, torch.arange(k_max, device=device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    basis = basis * signs.unsqueeze(0)
    total_energy = float((centered * centered).sum())
    explained = (singular**2 / max(total_energy, 1e-30)).cpu().numpy()
    return {
        "mean": mean.float().cpu().numpy(),
        "basis": basis.float().cpu().numpy(),
        "singular_values": singular.float().cpu().numpy(),
        "explained_ratio": explained.astype(np.float64),
        "cumulative_ratio": np.cumsum(explained).astype(np.float64),
    }


def project_latent(responses: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    X = flatten_responses(responses)
    return ((X - mean[None, :]) @ basis).astype(np.float32)


# ---------------------------------------------------------------------------
# Pair sampling (indices only, no new states)
# ---------------------------------------------------------------------------

PAIR_TYPES = {0: "local", 1: "response_local", 2: "medium", 3: "random"}


def sample_pairs(
    u: np.ndarray,
    z: np.ndarray | None,
    k_local: int = 16,
    k_response: int = 16,
    n_medium: int = 4,
    n_random: int = 8,
    chunk: int = 500,
    seed: int = 2026,
    device: str | torch.device = "cuda",
) -> dict[str, np.ndarray]:
    """Construct pairs by parameter / response locality, plus medium/random pairs."""
    device = torch.device(device)
    n = len(u)
    ut = torch.as_tensor(u, dtype=torch.float32, device=device)
    zt = torch.as_tensor(z, dtype=torch.float32, device=device) if z is not None else None

    src, dst, kinds = [], [], []
    for begin in range(0, n, int(chunk)):
        end = min(n, begin + int(chunk))
        rows = torch.arange(end - begin, device=device)
        global_rows = torch.arange(begin, end, device=device)
        du = torch.cdist(ut[begin:end], ut)
        du[rows, global_rows] = float("inf")

        local = torch.topk(du, k_local, largest=False).indices
        for row in range(end - begin):
            for target in local[row].tolist():
                src.append(begin + row)
                dst.append(int(target))
                kinds.append(0)

        if zt is not None:
            dz = torch.cdist(zt[begin:end], zt)
            dz[rows, global_rows] = float("inf")
            resp = torch.topk(dz, k_response, largest=False).indices
            for row in range(end - begin):
                for target in resp[row].tolist():
                    src.append(begin + row)
                    dst.append(int(target))
                    kinds.append(1)

        for row in range(end - begin):
            row_dist = du[row]
            valid = torch.isfinite(row_dist)
            if int(valid.sum()) == 0:
                continue
            adjusted = row_dist[valid]
            lo = torch.quantile(adjusted, 0.5).item()
            hi = torch.quantile(adjusted, 0.9).item()
            candidates = torch.nonzero((row_dist >= lo) & (row_dist <= hi)).flatten()
            if candidates.numel() > 0:
                picks = candidates[
                    torch.randint(0, candidates.numel(), (n_medium,), device=device)
                ]
                for target in picks.tolist():
                    src.append(begin + row)
                    dst.append(int(target))
                    kinds.append(2)
            randoms = torch.randint(0, n, (n_random,), device=device)
            for target in randoms.tolist():
                if target != begin + row:
                    src.append(begin + row)
                    dst.append(int(target))
                    kinds.append(3)

    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)
    kinds = np.asarray(kinds, dtype=np.uint8)
    du_all = np.linalg.norm(u[src] - u[dst], axis=1).astype(np.float32)
    dz_all = (
        np.linalg.norm(z[src] - z[dst], axis=1).astype(np.float32)
        if z is not None
        else np.zeros_like(du_all)
    )
    return {
        "src": src,
        "dst": dst,
        "type": kinds,
        "u_distance": du_all,
        "z_distance": dz_all,
    }


def normalized_state(m: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    center = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-3)
    return (m - center) / half, center, half


def ridge_jacobian(delta_u: np.ndarray, delta_z: np.ndarray, lam: float) -> np.ndarray:
    """Ridge fit of ``Delta z ~= A Delta u``; returns A with shape (K, 41)."""
    du = torch.as_tensor(delta_u, dtype=torch.float64)
    dz = torch.as_tensor(delta_z, dtype=torch.float64)
    gram = du.T @ du + float(lam) * torch.eye(du.shape[1], dtype=torch.float64)
    rhs = du.T @ dz
    a_t = torch.linalg.solve(gram, rhs)
    return a_t.T.numpy()


def ridge_controller(delta_z: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    """Ridge fit of ``c ~= W Delta z``; returns W with shape (r, K)."""
    z = torch.as_tensor(delta_z, dtype=torch.float64)
    c = torch.as_tensor(target, dtype=torch.float64)
    gram = z.T @ z + float(lam) * torch.eye(z.shape[1], dtype=torch.float64)
    rhs = z.T @ c
    w_t = torch.linalg.solve(gram, rhs)
    return w_t.T.numpy()


def normalized_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denom = float(np.sqrt(np.mean(target**2))) + 1e-12
    return float(np.sqrt(np.mean((pred - target) ** 2)) / denom)


def evaluate_delta(
    simulator,
    m_current: np.ndarray,
    delta_m: np.ndarray,
) -> tuple[dict, list[dict]]:
    """Official one-shot evaluation for precomputed corrections (cached states)."""
    requirements = requirement_infos(simulator)
    lower = np.asarray(simulator.lower_bounds)
    upper = np.asarray(simulator.upper_bounds)
    rows = []
    for idx in range(len(m_current)):
        delta = np.asarray(delta_m[idx], dtype=np.float64)
        result = simulator.apply_correction(np.asarray(m_current[idx], dtype=np.float64), delta)
        raw = np.asarray(m_current[idx], dtype=np.float64) + delta
        clipped = np.clip(raw, lower, upper)
        details = result.details
        per_requirement = {}
        margins = []
        for req in requirements:
            value = float(details[f"{req.name}_value_db"])
            threshold = float(req.threshold_db)
            margin = threshold - value if req.operator == "<=" else value - threshold
            margins.append(margin)
            per_requirement[req.name] = {
                "value_db": value,
                "pass": bool(details[f"{req.name}_pass"]),
                "miss_db": float(details[f"{req.name}_miss_db"]),
                "margin_db": float(margin),
            }
        rows.append(
            {
                "passed": bool(result.all_specs_pass),
                "max_miss_db": float(result.max_spec_miss_db),
                "min_margin_db": float(min(margins)),
                "margins": margins,
                "requirements": per_requirement,
                "clipped_entry_rate": float(np.mean(clipped != raw)),
                "correction_l2": float(np.linalg.norm(delta)),
            }
        )
    misses = np.asarray([row["max_miss_db"] for row in rows], dtype=np.float64)
    summary = {
        "num_states": len(rows),
        "pass_count": int(sum(row["passed"] for row in rows)),
        "pass_rate": float(np.mean([row["passed"] for row in rows])) if rows else 0.0,
        "mean_max_miss_db": float(np.mean(misses)) if rows else float("inf"),
        "median_max_miss_db": float(np.median(misses)) if rows else float("inf"),
        "p90_max_miss_db": float(np.quantile(misses, 0.90)) if rows else float("inf"),
        "worst_max_miss_db": float(np.max(misses)) if rows else float("inf"),
        "correction_l2_mean": float(np.mean([row["correction_l2"] for row in rows])) if rows else 0.0,
        "clipped_entry_rate_mean": float(np.mean([row["clipped_entry_rate"] for row in rows])) if rows else 0.0,
        "per_requirement": {
            req.name: {
                "pass_rate": float(
                    np.mean([row["requirements"][req.name]["pass"] for row in rows])
                ),
                "mean_miss_db": float(
                    np.mean([row["requirements"][req.name]["miss_db"] for row in rows])
                ),
            }
            for req in requirements
        },
    }
    return summary, rows
