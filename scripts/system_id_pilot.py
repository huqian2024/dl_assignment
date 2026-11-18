"""Stage 5 pilot: system-identification features for CM19.

Stages:
    sweep      rational order sweep on 512 train states + frequency holdout
    rational   fit selected orders on 2000 train states, save features
    algebra    complex-algebra feature screening (1/S21, S11/S21, log-deriv)
    regress    PCA/Ridge on raw vs rational vs algebra features
    oneshot    official validation one-shot for the best cheap model
    all        everything

No new states are sampled; only cached train/validation data are used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import multiprocessing as mp

import numpy as np
import torch

from solution.data import CM19ResponseDataset
from solution.pairwise import evaluate_delta, normalized_rmse
from solution.simulator import load_simulator
from solution.system_id import (
    cheb_basis,
    coefficient_features,
    complex_algebra_features,
    evaluate_rational,
    fit_shared_rational,
    log_derivative_features,
    pole_zero_features,
    reconstruction_metrics,
    spec_reconstruction,
)
from solution.utils import save_json

OUT = ROOT / "artifacts/system_id"
OUT.mkdir(parents=True, exist_ok=True)
ORDERS = (8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
EPS_LIST = (1e-8, 1e-6, 1e-4)
K_LIST = (64, 128, 256, 512)
WEAK_PARAMS = [1, 21, 23, 26, 29, 31, 33, 34, 37, 38, 39, 40]
DANGEROUS = [8, 32, 37, 35, 22, 23, 10, 24, 31, 5]

_SIM = None


def log(message: str) -> None:
    print(f"[system-id] {message}", flush=True)


def _init_worker() -> None:
    global _SIM
    _SIM = load_simulator()


def load_split(split: str, n: int | None = None):
    dataset = CM19ResponseDataset("cache/dataset", split=split, limit=n)
    rows = np.asarray(dataset._indices)
    responses = np.array(dataset.responses[rows]).astype(np.float64)
    m_current = np.array(dataset.m_current[rows]).astype(np.float64)
    seeds = np.asarray(dataset.seeds_slice())
    return responses, m_current, seeds


def to_complex(responses: np.ndarray):
    s11 = responses[:, 0] + 1j * responses[:, 1]
    s21 = responses[:, 2] + 1j * responses[:, 3]
    return s11, s21


def frequency_grid():
    simulator = load_simulator()
    freqs = np.asarray(simulator.frequency_hz, dtype=np.float64)
    fmin, fmax = freqs.min(), freqs.max()
    x = 2.0 * (freqs - fmin) / (fmax - fmin) - 1.0
    return freqs, x


FIT_MASK = (np.arange(2001) % 4) != 0
HOLDOUT_MASK = ~FIT_MASK


def _sweep_task(task):
    idx, s11, s21, x = task
    out = {}
    for order in ORDERS:
        coeffs = fit_shared_rational(s11, s21, x, order, fit_mask=FIT_MASK)
        s11_hat, s21_hat = evaluate_rational(coeffs, x)
        hold = reconstruction_metrics(s11, s21, s11_hat, s21_hat, HOLDOUT_MASK)
        fitm = reconstruction_metrics(s11, s21, s11_hat, s21_hat, FIT_MASK)
        spec = spec_reconstruction(_SIM, s11, s21, s11_hat, s21_hat, HOLDOUT_MASK)
        out[str(order)] = {
            "order": order,
            "condition": coeffs["condition"],
            "holdout": hold,
            "fit": fitm,
            "spec": spec,
        }
    return idx, out


def _rational_task(task):
    idx, s11, s21, x, orders = task
    out = {}
    for order in orders:
        full = fit_shared_rational(s11, s21, x, order, fit_mask=None)
        hold = fit_shared_rational(s11, s21, x, order, fit_mask=FIT_MASK)
        s11_hat, s21_hat = evaluate_rational(hold, x)
        metrics = reconstruction_metrics(s11, s21, s11_hat, s21_hat, HOLDOUT_MASK)
        spec = spec_reconstruction(_SIM, s11, s21, s11_hat, s21_hat, HOLDOUT_MASK)
        out[str(order)] = {
            "full": full,
            "holdout_fit": hold,
            "metrics": metrics,
            "spec": spec,
        }
    return idx, out


def choose_orders(sweep_path: Path) -> list[int]:
    data = json.loads(sweep_path.read_text())
    rows = data["summary"]
    best_rel = min(row["holdout_relative_l2_median"] for row in rows.values())
    best_spec = min(row["spec_max_error_db_median"] for row in rows.values())
    selected = [
        row["order"]
        for row in rows.values()
        if row["holdout_relative_l2_median"] <= best_rel * 1.5
        and row["spec_max_error_db_median"] <= max(best_spec * 2.0, 0.05)
    ]
    return sorted(set(selected))[:2] if selected else [int(min(rows, key=lambda k: rows[k]["holdout_relative_l2_median"]))]


def stage_rational(n_states: int, workers: int, orders: list[int]):
    responses, m_current, seeds = load_split("train", n_states)
    s11, s21 = to_complex(responses)
    _, x = frequency_grid()
    tasks = [(i, s11[i], s21[i], x, orders) for i in range(n_states)]
    start = time.perf_counter()
    with mp.get_context("fork").Pool(
        processes=int(workers), initializer=_init_worker
    ) as pool:
        results = pool.map(_rational_task, tasks, chunksize=2)
    elapsed = time.perf_counter() - start
    order_idx = np.argsort([idx for idx, _ in results])
    results = [results[i] for i in order_idx]

    payload = {
        "n_states": int(n_states),
        "seeds": seeds,
        "orders": list(orders),
        "m_current": m_current,
        "wall_seconds": float(elapsed),
    }
    summary = {}
    for order in orders:
        key = str(order)
        coeffs_full = [result[key]["full"] for _, result in results]
        coeffs_hold = [result[key]["holdout_fit"] for _, result in results]
        d = np.stack([c["d"] for c in coeffs_full])
        a = np.stack([c["a"] for c in coeffs_full])
        b = np.stack([c["b"] for c in coeffs_full])
        d_h = np.stack([c["d"] for c in coeffs_hold])
        a_h = np.stack([c["a"] for c in coeffs_hold])
        b_h = np.stack([c["b"] for c in coeffs_hold])
        features_r1 = np.stack(
            [coefficient_features(c) for c in coeffs_full]
        )
        try:
            features_r2 = np.stack(
                [pole_zero_features(c) for c in coeffs_full]
            )
        except Exception:
            features_r2 = np.full((n_states, 2 * (3 * order + 2)), np.nan)
        metrics = [result[key]["metrics"] for _, result in results]
        spec_errors = np.array(
            [result[key]["spec"]["max_abs_value_error_db"] for _, result in results]
        )
        rel = np.array([row["relative_l2"] for row in metrics])
        payload[f"d_{order}"] = d
        payload[f"a_{order}"] = a
        payload[f"b_{order}"] = b
        payload[f"d_hold_{order}"] = d_h
        payload[f"a_hold_{order}"] = a_h
        payload[f"b_hold_{order}"] = b_h
        payload[f"features_r1_{order}"] = features_r1
        payload[f"features_r2_{order}"] = features_r2
        payload[f"holdout_rel_l2_{order}"] = rel
        payload[f"spec_error_{order}"] = spec_errors
        summary[str(order)] = {
            "order": int(order),
            "holdout_relative_l2_median": float(np.median(rel)),
            "holdout_relative_l2_p90": float(np.quantile(rel, 0.90)),
            "spec_max_error_db_median": float(np.median(spec_errors)),
            "spec_max_error_db_p90": float(np.quantile(spec_errors, 0.90)),
            "condition_median": float(
                np.median([c["condition"] for c in coeffs_full])
            ),
            "feature_dim_r1": int(features_r1.shape[1]),
            "feature_dim_r2": int(features_r2.shape[1]),
            "r2_nan_fraction": float(np.mean(~np.isfinite(features_r2))),
        }
        log(
            f"order {order}: holdout relL2 median {np.median(rel):.4g}, "
            f"spec err {np.median(spec_errors):.3g} dB, "
            f"R1 dim {features_r1.shape[1]}, R2 nan {summary[str(order)]['r2_nan_fraction']:.3f}"
        )
    payload["summary"] = summary
    np.savez_compressed(OUT / f"rational_features_{n_states}.npz", **payload)
    save_json(OUT / f"rational_features_{n_states}.json", summary)
    return summary


def stage_sweep(n_states: int, workers: int):
    responses, m_current, seeds = load_split("train", n_states)
    s11, s21 = to_complex(responses)
    _, x = frequency_grid()
    tasks = [(i, s11[i], s21[i], x) for i in range(n_states)]
    start = time.perf_counter()
    with mp.get_context("fork").Pool(
        processes=int(workers), initializer=_init_worker
    ) as pool:
        results = pool.map(_sweep_task, tasks, chunksize=2)
    elapsed = time.perf_counter() - start
    order = np.argsort([idx for idx, _ in results])
    results = [results[i] for i in order]

    summary = {}
    for ord_str in map(str, ORDERS):
        rows = [result[ord_str] for _, result in results]
        rel = np.array([row["holdout"]["relative_l2"] for row in rows])
        maxabs = np.array([row["holdout"]["max_abs"] for row in rows])
        spec_err = np.array([row["spec"]["max_abs_value_error_db"] for row in rows])
        pass_match = np.array([row["spec"]["pass_match_rate"] for row in rows])
        cond = np.array([row["condition"] for row in rows])
        summary[ord_str] = {
            "order": int(ord_str),
            "holdout_relative_l2_median": float(np.median(rel)),
            "holdout_relative_l2_p90": float(np.quantile(rel, 0.90)),
            "holdout_max_abs_median": float(np.median(maxabs)),
            "spec_max_error_db_median": float(np.median(spec_err)),
            "spec_max_error_db_p90": float(np.quantile(spec_err, 0.90)),
            "spec_pass_match_rate_mean": float(np.mean(pass_match)),
            "condition_median": float(np.median(cond)),
            "condition_p90": float(np.quantile(cond, 0.90)),
        }
        log(
            f"order {ord_str}: holdout relL2 median {summary[ord_str]['holdout_relative_l2_median']:.4g}, "
            f"spec err {summary[ord_str]['spec_max_error_db_median']:.3g} dB, "
            f"pass match {summary[ord_str]['spec_pass_match_rate_mean']:.3f}"
        )
    payload = {
        "n_states": int(n_states),
        "orders": list(ORDERS),
        "fit_mask": FIT_MASK,
        "holdout_mask": HOLDOUT_MASK,
        "summary": summary,
        "wall_seconds": float(elapsed),
        "per_state": [
            {"seed": int(seeds[i]), **result} for i, result in results
        ],
    }
    save_json(OUT / f"sweep_{n_states}.json", payload)
    return summary


def pca_ridge_eval(
    X_train,
    X_val,
    Y_train,
    Y_val,
    feature_name: str,
    target_name: str,
    half: np.ndarray | None = None,
    K_list=K_LIST,
    lambdas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4),
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    Y_val = np.asarray(Y_val, dtype=np.float64)
    max_q = int(min(max(K_list), min(X_train.shape) - 1))
    K_list = [k for k in K_list if k <= max_q]
    if not K_list:
        return {}
    Xt = torch.as_tensor(X_train, device=device)
    mean = Xt.mean(dim=0)
    Xc = Xt - mean
    _, _, basis = torch.pca_lowrank(Xc, q=max_q, center=False, niter=6)
    mean_np = mean.cpu().numpy()
    basis_np = basis.cpu().numpy()
    results = {}
    n_fit = int(round(0.8 * len(X_train)))
    for K in K_list:
        B = basis_np[:, :K]
        Z_train = (X_train - mean_np) @ B
        Z_val = (X_val - mean_np) @ B
        mean_x = Z_train[:n_fit].mean(axis=0, keepdims=True)
        std_x = Z_train[:n_fit].std(axis=0, keepdims=True) + 1e-8
        Zf = (Z_train[:n_fit] - mean_x) / std_x
        Yf = Y_train[:n_fit]
        mean_y = Yf.mean(axis=0, keepdims=True)
        Yf_c = Yf - mean_y
        best = None
        for lam in lambdas:
            gram = Zf.T @ Zf + lam * np.eye(K)
            W = np.linalg.solve(gram, Zf.T @ Yf_c)
            pred = ((Z_train[n_fit:] - mean_x) / std_x) @ W + mean_y
            err = np.sqrt(np.mean((pred - Y_train[n_fit:]) ** 2)) / (
                np.sqrt(np.mean(Y_train[n_fit:] ** 2)) + 1e-12
            )
            if best is None or err < best[0]:
                best = (err, lam)
        _, lam = best
        Zall = (Z_train - mean_x) / std_x
        W = np.linalg.solve(
            Zall.T @ Zall + lam * np.eye(K), Zall.T @ (Y_train - mean_y)
        )
        pred = ((Z_val - mean_x) / std_x) @ W + mean_y
        diff = pred - Y_val
        abs_err = np.abs(diff)
        var = Y_val.var(axis=0)
        r2 = 1.0 - (diff**2).mean(axis=0) / np.maximum(var, 1e-12)
        physical = half[None, :] * diff if half is not None else diff
        entry = {
            "feature": feature_name,
            "target": target_name,
            "K": int(K),
            "lambda": float(lam),
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt((diff**2).mean())),
            "p95_abs": float(np.quantile(abs_err, 0.95)),
            "p99_abs": float(np.quantile(abs_err, 0.99)),
            "max_abs": float(abs_err.max()),
            "physical_mae": float(np.abs(physical).mean()),
            "physical_p95": float(np.quantile(np.abs(physical), 0.95)),
            "mean_r2": float(r2.mean()),
            "median_r2": float(np.median(r2)),
            "num_r2_below_0.1": int(np.sum(r2 < 0.1)),
            "per_parameter_r2": r2.tolist(),
            "weak_parameter_r2": {str(i): float(r2[i]) for i in WEAK_PARAMS},
            "dangerous_parameter_r2": {str(i): float(r2[i]) for i in DANGEROUS},
        }
        results[str(K)] = entry
        log(
            f"{feature_name}->{target_name} K={K}: physical MAE {entry['physical_mae']:.5f}, "
            f"mean R2 {entry['mean_r2']:.3f}, #R2<0.1 {entry['num_r2_below_0.1']}"
        )
    return results



def _targets(n_train: int | None):
    simulator = load_simulator()
    lower = np.asarray(simulator.lower_bounds)
    upper = np.asarray(simulator.upper_bounds)
    center = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-6)
    anchor = np.load(ROOT / "artifacts/common/anchor.npz")["m_star"]
    responses_tr, m_tr, seeds_tr = load_split("train", n_train)
    responses_va, m_va, seeds_va = load_split("validation", None)
    u_tr = (m_tr - center) / half
    u_va = (m_va - center) / half
    du_tr = (anchor[None, :] - center) / half - u_tr
    du_va = (anchor[None, :] - center) / half - u_va
    return {
        "responses_tr": responses_tr,
        "responses_va": responses_va,
        "m_tr": m_tr,
        "m_va": m_va,
        "u_tr": u_tr,
        "u_va": u_va,
        "du_tr": du_tr,
        "du_va": du_va,
        "half": half,
        "center": center,
        "anchor": anchor,
        "seeds_tr": seeds_tr,
        "seeds_va": seeds_va,
    }


def stage_algebra(n_train: int, device: str):
    data = _targets(n_train)
    s11_tr, s21_tr = to_complex(data["responses_tr"])
    s11_va, s21_va = to_complex(data["responses_va"])
    feature_sets = {
        "raw": (
            data["responses_tr"].reshape(len(s11_tr), -1).astype(np.float32),
            data["responses_va"].reshape(len(s11_va), -1).astype(np.float32),
        )
    }
    for eps in EPS_LIST:
        feature_sets[f"inverse_eps{eps:g}"] = (
            np.stack(
                [complex_algebra_features(s11_tr[i], s21_tr[i], eps, "inverse") for i in range(len(s11_tr))]
            ).astype(np.float32),
            np.stack(
                [complex_algebra_features(s11_va[i], s21_va[i], eps, "inverse") for i in range(len(s11_va))]
            ).astype(np.float32),
        )
    for eps in (1e-8, 1e-6):
        feature_sets[f"ratio_eps{eps:g}"] = (
            np.stack(
                [complex_algebra_features(s11_tr[i], s21_tr[i], eps, "ratio") for i in range(len(s11_tr))]
            ).astype(np.float32),
            np.stack(
                [complex_algebra_features(s11_va[i], s21_va[i], eps, "ratio") for i in range(len(s11_va))]
            ).astype(np.float32),
        )
        feature_sets[f"combined_eps{eps:g}"] = (
            np.stack(
                [complex_algebra_features(s11_tr[i], s21_tr[i], eps, "combined") for i in range(len(s11_tr))]
            ).astype(np.float32),
            np.stack(
                [complex_algebra_features(s11_va[i], s21_va[i], eps, "combined") for i in range(len(s11_va))]
            ).astype(np.float32),
        )
        feature_sets[f"logderiv_eps{eps:g}"] = (
            np.stack(
                [log_derivative_features(s11_tr[i], s21_tr[i], eps) for i in range(len(s11_tr))]
            ).astype(np.float32),
            np.stack(
                [log_derivative_features(s11_va[i], s21_va[i], eps) for i in range(len(s11_va))]
            ).astype(np.float32),
        )

    results = {}
    for name, (X_tr, X_va) in feature_sets.items():
        results[name] = {}
        for target_name, Y_tr, Y_va in (
            ("absolute_u", data["u_tr"], data["u_va"]),
            ("anchor_delta_u", data["du_tr"], data["du_va"]),
        ):
            results[name][target_name] = pca_ridge_eval(
                X_tr,
                X_va,
                Y_tr,
                Y_va,
                feature_name=name,
                target_name=target_name,
                half=data["half"],
                device=device,
            )
    save_json(OUT / f"algebra_regression_{n_train}.json", results)
    return results



def stage_rational_apply(split: str, n_states: int | None, orders: list[int], workers: int):
    responses, m_current, seeds = load_split(split, n_states)
    s11, s21 = to_complex(responses)
    _, x = frequency_grid()
    tasks = [(i, s11[i], s21[i], x, orders) for i in range(len(responses))]
    start = time.perf_counter()
    with mp.get_context("fork").Pool(
        processes=int(workers), initializer=_init_worker
    ) as pool:
        results = pool.map(_rational_task, tasks, chunksize=2)
    elapsed = time.perf_counter() - start
    order_idx = np.argsort([idx for idx, _ in results])
    results = [results[i] for i in order_idx]
    log(f"{split}: {len(results)} rational fits done in {elapsed:.1f}s")
    payload = {
        "split": split,
        "n_states": int(len(responses)),
        "seeds": seeds,
        "orders": list(orders),
        "m_current": m_current,
        "wall_seconds": float(elapsed),
    }
    summary = {}
    for order in orders:
        key = str(order)
        coeffs = [result[key]["full"] for _, result in results]
        r1 = np.stack([coefficient_features(c) for c in coeffs]).astype(np.float32)
        try:
            r2 = np.stack([pole_zero_features(c) for c in coeffs]).astype(np.float32)
        except Exception:
            r2 = np.full((len(coeffs), 2 * (3 * order + 2)), np.nan, dtype=np.float32)
        rel = np.array([result[key]["metrics"]["relative_l2"] for _, result in results])
        spec = np.array(
            [result[key]["spec"]["max_abs_value_error_db"] for _, result in results]
        )
        payload[f"features_r1_{order}"] = r1
        payload[f"features_r2_{order}"] = r2
        payload[f"holdout_rel_l2_{order}"] = rel
        payload[f"spec_error_{order}"] = spec
        summary[str(order)] = {
            "order": int(order),
            "holdout_relative_l2_median": float(np.median(rel)),
            "spec_max_error_db_median": float(np.median(spec)),
            "feature_dim_r1": int(r1.shape[1]),
            "r2_nan_fraction": float(np.mean(~np.isfinite(r2))),
        }
    payload["summary"] = summary
    np.savez_compressed(OUT / f"rational_{split}.npz", **payload)
    save_json(OUT / f"rational_{split}.json", summary)
    return summary


def stage_rational_regression(n_train: int, device: str, orders: list[int]):
    data = _targets(n_train)
    train_archive = np.load(OUT / "rational_train.npz")
    val_archive = np.load(OUT / "rational_validation.npz")
    raw_tr = data["responses_tr"].reshape(len(data["responses_tr"]), -1).astype(np.float32)
    raw_va = data["responses_va"].reshape(len(data["responses_va"]), -1).astype(np.float32)

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    xt = torch.as_tensor(raw_tr, device=device_t)
    mean = xt.mean(dim=0)
    _, _, basis = torch.pca_lowrank(xt - mean, q=64, center=False, niter=6)
    mean_np = mean.cpu().numpy()
    basis_np = basis.cpu().numpy()[:, :64]
    z_tr = ((raw_tr - mean_np) @ basis_np).astype(np.float32)
    z_va = ((raw_va - mean_np) @ basis_np).astype(np.float32)

    feature_sets = {"raw": (raw_tr, raw_va)}
    for order in orders:
        r1_tr = train_archive[f"features_r1_{order}"].astype(np.float32)
        r1_va = val_archive[f"features_r1_{order}"].astype(np.float32)
        feature_sets[f"r1_order{order}"] = (r1_tr, r1_va)
        feature_sets[f"r3_order{order}_plus_raw64"] = (
            np.concatenate([r1_tr, z_tr], axis=1),
            np.concatenate([r1_va, z_va], axis=1),
        )
        r2_tr = train_archive.get(f"features_r2_{order}")
        r2_va = val_archive.get(f"features_r2_{order}")
        if (
            r2_tr is not None
            and r2_va is not None
            and np.isfinite(r2_tr).all()
            and np.isfinite(r2_va).all()
        ):
            feature_sets[f"r2_order{order}"] = (
                r2_tr.astype(np.float32),
                r2_va.astype(np.float32),
            )

    results = {}
    for name, (X_tr, X_va) in feature_sets.items():
        results[name] = {}
        for target_name, Y_tr, Y_va in (
            ("absolute_u", data["u_tr"], data["u_va"]),
            ("anchor_delta_u", data["du_tr"], data["du_va"]),
        ):
            results[name][target_name] = pca_ridge_eval(
                X_tr,
                X_va,
                Y_tr,
                Y_va,
                feature_name=name,
                target_name=target_name,
                half=data["half"],
                device=device,
            )
    save_json(OUT / f"rational_regression_{n_train}.json", results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("sweep", "rational", "algebra", "regress", "all"),
        default="all",
    )
    parser.add_argument("--n-sweep", type=int, default=512)
    parser.add_argument("--n-rational", type=int, default=2000)
    parser.add_argument("--n-algebra", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--orders", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.stage in ("sweep", "all"):
        stage_sweep(args.n_sweep, args.workers)
        if args.stage == "sweep":
            return 0

    orders = args.orders
    if orders is None:
        sweep_path = OUT / f"sweep_{args.n_sweep}.json"
        if not sweep_path.exists():
            raise SystemExit("run --stage sweep first (or pass --orders)")
        orders = choose_orders(sweep_path)
    log(f"selected orders: {orders}")

    if args.stage in ("rational", "all"):
        stage_rational_apply("train", args.n_rational, orders, args.workers)
        stage_rational_apply("validation", None, orders, args.workers)
        if args.stage == "rational":
            return 0

    if args.stage in ("algebra", "all"):
        stage_algebra(args.n_algebra, args.device)
        if args.stage == "algebra":
            return 0

    if args.stage in ("regress", "all"):
        stage_rational_regression(args.n_rational, args.device, orders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

