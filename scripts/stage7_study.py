"""Stage 7: rational-coordinate stability, full-rank regression, probes.

Only cached train 8000 / validation 1000 are used.  No simulator states are
sampled and no anchor is re-optimized.
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

from solution.data import CM19ResponseDataset
from solution.pairwise import evaluate_delta, normalized_state
from solution.simulator import load_simulator
from solution.system_id import (
    evaluate_rational,
    fit_shared_rational,
    reconstruction_metrics,
    spec_reconstruction,
)
from solution.utils import save_json

OUT = ROOT / "artifacts/stage7"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = 64
DANGEROUS = [5, 8, 10, 22, 23, 24, 31, 32, 35, 37]
WEAK = [21, 23, 26, 31]
LAMBDAS = (3e-11, 1e-10, 3e-10)
RIDGE_LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def log(message: str) -> None:
    print(f"[stage7] {message}", flush=True)


def load_rational(split: str) -> dict:
    path = ROOT / f"artifacts/system_id/rational_{split}.npz"
    data = np.load(path, allow_pickle=True)
    features = data["features_r1_64"].astype(np.float64)
    size = ORDER + (ORDER + 1) * 2
    d = features[:, 0:ORDER] + 1j * features[:, size:size + ORDER]
    a = (
        features[:, ORDER:ORDER + ORDER + 1]
        + 1j * features[:, size + ORDER:size + ORDER + ORDER + 1]
    )
    b = (
        features[:, ORDER + ORDER + 1:size]
        + 1j * features[:, size + ORDER + ORDER + 1:2 * size]
    )
    return {
        "d": d,
        "a": a,
        "b": b,
        "features": features,
        "m_current": data["m_current"].astype(np.float64),
        "seeds": data["seeds"].astype(np.int64),
    }


def coefficient_vector(d, a, b) -> np.ndarray:
    vec = np.concatenate([d, a, b])
    return np.concatenate([vec.real, vec.imag])


def group_indices(order: int = ORDER) -> dict[str, np.ndarray]:
    size = order + (order + 1) * 2
    d = np.arange(0, order)
    a = np.arange(order, order + order + 1)
    b = np.arange(order + order + 1, size)
    return {
        "D": np.concatenate([d, d + size]),
        "A": np.concatenate([a, a + size]),
        "B": np.concatenate([b, b + size]),
    }


def metrics(pred: np.ndarray, target: np.ndarray, half: np.ndarray | None = None) -> dict:
    error = pred - target
    abs_error = np.abs(error)
    ss_res = (error ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    out = {
        "mae": float(abs_error.mean()),
        "p95_abs": float(np.quantile(abs_error, 0.95)),
        "mean_r2": float(np.mean(r2)),
        "median_r2": float(np.median(r2)),
        "count_r2_below_0_1": int((r2 < 0.1).sum()),
        "per_parameter_r2": r2.tolist(),
        "weak_r2": {str(p): float(r2[p]) for p in WEAK},
        "dangerous_r2": {str(p): float(r2[p]) for p in DANGEROUS},
    }
    if half is not None:
        physical = np.abs(half[None, :] * error)
        out["physical_mae"] = float(physical.mean())
        out["physical_p95"] = float(np.quantile(physical, 0.95))
        out["dangerous_physical_mae"] = float(physical[:, DANGEROUS].mean())
        out["dangerous_physical_p95"] = float(np.quantile(physical[:, DANGEROUS], 0.95))
    return out


def standardize(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X_train - mean) / std, (X_other - mean) / std, mean, std


def per_output_ridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    lams=RIDGE_LAMBDAS,
    fit_end: int = 7000,
    select_end: int = 8000,
):
    """Select lambda_j on train 0:7000 -> 7000:8000, then refit on all train."""
    Xf, Yf = X_train[:fit_end], Y_train[:fit_end]
    Xs, Ys = X_train[fit_end:select_end], Y_train[fit_end:select_end]
    xm, ym = Xf.mean(axis=0), Yf.mean(axis=0)
    Xc, Yc = Xf - xm, Yf - ym
    gram = Xc.T @ Xc
    rhs = Xc.T @ Yc
    best_lam = np.zeros(Y_train.shape[1], dtype=np.float64)
    best_mae = np.full(Y_train.shape[1], np.inf)
    predictions = {}
    for lam in lams:
        coef = np.linalg.solve(gram + float(lam) * np.eye(gram.shape[0]), rhs)
        pred = (Xs - xm) @ coef + ym
        mae = np.abs(pred - Ys).mean(axis=0)
        better = mae < best_mae
        best_mae[better] = mae[better]
        best_lam[better] = lam
        predictions[lam] = pred
    # refit on all train with per-output lambda
    xm_all, ym_all = X_train.mean(axis=0), Y_train.mean(axis=0)
    Xc_all, Yc_all = X_train - xm_all, Y_train - ym_all
    gram_all = Xc_all.T @ Xc_all
    rhs_all = Xc_all.T @ Yc_all
    coef_all = np.zeros((X_train.shape[1], Y_train.shape[1]), dtype=np.float64)
    for lam in np.unique(best_lam):
        cols = np.where(best_lam == lam)[0]
        coef_all[:, cols] = np.linalg.solve(
            gram_all + float(lam) * np.eye(gram_all.shape[0]), rhs_all[:, cols]
        )
    val_pred = (X_val - xm_all) @ coef_all + ym_all
    return {
        "pred": val_pred,
        "best_lambda": best_lam,
        "select_mae": best_mae,
        "val_pred_by_lambda": predictions,
        "coef": coef_all,
        "mean": xm_all,
        "target_mean": ym_all,
    }



FIT_MASKS = {
    "A": (np.arange(2001) % 4) != 0,
    "B": ~np.isin(np.arange(2001) % 8, (0, 1)),
    "C": ~np.isin(np.arange(2001) % 8, (2, 3)),
}
PROBE_X = np.linspace(-1.0, 1.0, 32)
PROBE_ETAS = (0.02, 0.05, 0.10)

_SIM = None


def _init_worker():
    global _SIM
    _SIM = load_simulator()


def cheb_matrix(z: np.ndarray, order: int) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    out = np.empty((len(z), int(order) + 1), dtype=np.complex128)
    out[:, 0] = 1.0
    if order >= 1:
        out[:, 1] = z
    for k in range(2, int(order) + 1):
        out[:, k] = 2.0 * z * out[:, k - 1] - out[:, k - 2]
    return out


def probe_features(d, a, b, etas=PROBE_ETAS, include_denominator: bool = False) -> np.ndarray:
    """Vectorized complex-plane evaluation of S11/S21 (and optionally 1/D)."""
    parts = []
    for eta in etas:
        z = PROBE_X + 1j * float(eta)
        t = cheb_matrix(z, ORDER)
        denominator = 1.0 + t[:, 1:] @ np.asarray(d).T
        s11 = (t @ np.asarray(a).T) / denominator
        s21 = (t @ np.asarray(b).T) / denominator
        parts.extend([s11.real, s11.imag, s21.real, s21.imag])
        if include_denominator:
            inv = np.conj(denominator) / (np.abs(denominator) ** 2 + 1e-8)
            parts.extend([inv.real, inv.imag])
    return np.concatenate(parts, axis=0)


def _audit_task(task):
    idx, s11, s21, x = task
    qs, probes, rels, specs, conditions = [], [], [], [], []
    for mask in FIT_MASKS.values():
        for lam in LAMBDAS:
            coeffs = fit_shared_rational(s11, s21, x, ORDER, fit_mask=mask, ridge=lam)
            qs.append(coefficient_vector(coeffs["d"], coeffs["a"], coeffs["b"]))
            probes.append(probe_features(coeffs["d"][None, :], coeffs["a"][None, :], coeffs["b"][None, :]).ravel())
            hat11, hat21 = evaluate_rational(coeffs, x)
            hold = ~mask
            rels.append(reconstruction_metrics(s11, s21, hat11, hat21, hold)["relative_l2"])
            specs.append(spec_reconstruction(_SIM, s11, s21, hat11, hat21, hold)["max_abs_value_error_db"])
            conditions.append(coeffs["condition"])
    return idx, np.asarray(qs), np.asarray(probes), np.asarray(rels), np.asarray(specs), np.asarray(conditions)


def stage_audit(n_states: int = 256, workers: int = 8):
    dataset = CM19ResponseDataset("cache/dataset", split="train", limit=n_states)
    rows = np.asarray(dataset._indices)
    responses = np.array(dataset.responses[rows]).astype(np.float64)
    seeds = np.asarray(dataset.seeds_slice())
    s11 = responses[:, 0] + 1j * responses[:, 1]
    s21 = responses[:, 2] + 1j * responses[:, 3]
    sim = load_simulator()
    freqs = np.asarray(sim.frequency_hz, dtype=np.float64)
    x = 2.0 * (freqs - freqs.min()) / (freqs.max() - freqs.min()) - 1.0
    tasks = [(i, s11[i], s21[i], x) for i in range(n_states)]
    start = time.perf_counter()
    with mp.get_context("fork").Pool(processes=int(workers), initializer=_init_worker) as pool:
        results = pool.map(_audit_task, tasks, chunksize=2)
    results.sort(key=lambda item: item[0])
    elapsed = time.perf_counter() - start

    groups = group_indices()
    ref_index = 1  # mask A, lambda 1e-10
    group_rel = {name: [] for name in groups}
    group_cos = {name: [] for name in groups}
    all_rel, all_cos, probe_rel, probe_cos = [], [], [], []
    coordinate_cv, probe_coordinate_cv = [], []
    rels, specs, conditions = [], [], []
    for idx, qs, probes, rel, spec, cond in results:
        q_ref = qs[ref_index]
        p_ref = probes[ref_index]
        for k in range(len(qs)):
            all_rel.append(np.linalg.norm(qs[k] - q_ref) / (np.linalg.norm(q_ref) + 1e-30))
            all_cos.append(float(np.dot(qs[k], q_ref) / (np.linalg.norm(qs[k]) * np.linalg.norm(q_ref) + 1e-30)))
            probe_rel.append(np.linalg.norm(probes[k] - p_ref) / (np.linalg.norm(p_ref) + 1e-30))
            probe_cos.append(float(np.dot(probes[k], p_ref) / (np.linalg.norm(probes[k]) * np.linalg.norm(p_ref) + 1e-30)))
        for name, idxs in groups.items():
            ref = q_ref[idxs]
            for k in range(len(qs)):
                q = qs[k][idxs]
                group_rel[name].append(np.linalg.norm(q - ref) / (np.linalg.norm(ref) + 1e-30))
                group_cos[name].append(float(np.dot(q, ref) / (np.linalg.norm(q) * np.linalg.norm(ref) + 1e-30)))
        mean_q = qs.mean(axis=0)
        coordinate_cv.append(np.std(qs, axis=0) / (np.abs(mean_q) + 1e-8))
        mean_p = probes.mean(axis=0)
        probe_coordinate_cv.append(np.std(probes, axis=0) / (np.abs(mean_p) + 1e-8))
        rels.extend(rel.tolist())
        specs.extend(spec.tolist())
        conditions.extend(cond.tolist())

    def summarize(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }

    payload = {
        "n_states": int(n_states),
        "seeds": seeds.tolist(),
        "masks": {name: int(mask.sum()) for name, mask in FIT_MASKS.items()},
        "lambdas": list(LAMBDAS),
        "wall_seconds": float(elapsed),
        "coefficient_relative_variation": summarize(all_rel),
        "coefficient_cosine": summarize(all_cos),
        "group_relative_variation": {name: summarize(values) for name, values in group_rel.items()},
        "group_cosine": {name: summarize(values) for name, values in group_cos.items()},
        "per_coordinate_cv_median": float(np.median(np.concatenate(coordinate_cv))),
        "per_coordinate_cv_p90": float(np.quantile(np.concatenate(coordinate_cv), 0.90)),
        "probe_relative_variation": summarize(probe_rel),
        "probe_cosine": summarize(probe_cos),
        "probe_per_coordinate_cv_median": float(np.median(np.concatenate(probe_coordinate_cv))),
        "probe_per_coordinate_cv_p90": float(np.quantile(np.concatenate(probe_coordinate_cv), 0.90)),
        "reconstruction_relative_l2": summarize(rels),
        "spec_error_db": summarize(specs),
        "condition": summarize(conditions),
    }
    save_json(OUT / "coefficient_stability.json", payload)
    log(f"audit done: coeff rel {payload['coefficient_relative_variation']['median']:.3g}, "
        f"probe rel {payload['probe_relative_variation']['median']:.3g}")
    return payload


def load_targets():
    train = np.load(ROOT / "artifacts/system_id/rational_train.npz", allow_pickle=True)
    val = np.load(ROOT / "artifacts/system_id/rational_validation.npz", allow_pickle=True)
    sim = load_simulator()
    lower = np.asarray(sim.lower_bounds, dtype=np.float64)
    upper = np.asarray(sim.upper_bounds, dtype=np.float64)
    u_train, center, half = normalized_state(train["m_current"], lower, upper)
    u_val, _, _ = normalized_state(val["m_current"], lower, upper)
    anchor = np.load(ROOT / "artifacts/common/anchor.npz", allow_pickle=True)
    m_star = np.asarray(anchor["m_star"], dtype=np.float64)
    u_star, _, _ = normalized_state(m_star[None, :], lower, upper)
    return {
        "u_train": u_train,
        "u_val": u_val,
        "m_train": train["m_current"].astype(np.float64),
        "m_val": val["m_current"].astype(np.float64),
        "m_star": m_star,
        "u_star": u_star[0],
        "center": center,
        "half": half,
        "train": train,
        "val": val,
    }


def stage_full():
    train = load_rational("train")
    val = load_rational("validation")
    targets = load_targets()
    X_train = train["features"]
    X_val = val["features"]
    Xtr, Xva, mean, std = standardize(X_train, X_val)
    result = per_output_ridge(Xtr, targets["u_train"], Xva, targets["u_val"])
    report = metrics(result["pred"], targets["u_val"], targets["half"])
    report.update(
        {
            "feature_dim": int(Xtr.shape[1]),
            "best_lambda_counts": {
                str(lam): int(np.sum(result["best_lambda"] == lam)) for lam in np.unique(result["best_lambda"])
            },
            "best_lambda": result["best_lambda"].tolist(),
            "select_mae": result["select_mae"].tolist(),
        }
    )
    np.savez_compressed(
        OUT / "full_pred.npz",
        pred=result["pred"].astype(np.float32),
        best_lambda=result["best_lambda"],
        coef=result["coef"].astype(np.float32),
        mean=mean,
        std=std,
    )
    save_json(OUT / "full_coefficients.json", report)
    log(f"full ridge: MAE {report['mae']:.5f}, physical {report['physical_mae']:.5f}, R2 {report['mean_r2']:.3f}")
    return report


def stage_full_raw():
    train = load_rational("train")
    val = load_rational("validation")
    targets = load_targets()
    base = np.load(ROOT / "artifacts/stage6/stage6_features.npz", allow_pickle=True)
    variants = {
        "full_plus_raw64": (
            np.concatenate([train["features"], base["raw64_train"]], axis=1),
            np.concatenate([val["features"], base["raw64_val"]], axis=1),
        ),
        "pca256_plus_raw64": (
            np.concatenate([base["z_256_train"], base["raw64_train"]], axis=1),
            np.concatenate([base["z_256_val"], base["raw64_val"]], axis=1),
        ),
    }
    payload = {}
    predictions = {}
    for name, (Xtr_raw, Xva_raw) in variants.items():
        Xtr, Xva, _, _ = standardize(Xtr_raw.astype(np.float64), Xva_raw.astype(np.float64))
        result = per_output_ridge(Xtr, targets["u_train"], Xva, targets["u_val"])
        report = metrics(result["pred"], targets["u_val"], targets["half"])
        report["feature_dim"] = int(Xtr.shape[1])
        payload[name] = report
        predictions[f"pred_{name}"] = result["pred"].astype(np.float32)
        log(f"{name}: MAE {report['mae']:.5f}, physical {report['physical_mae']:.5f}")
    np.savez_compressed(OUT / "full_raw_pred.npz", **predictions)
    save_json(OUT / "full_plus_raw.json", payload)
    return payload


def stage_pca():
    from scripts.stage6_study import group_whiten

    train = load_rational("train")
    val = load_rational("validation")
    targets = load_targets()
    groups = group_indices()
    base = np.load(ROOT / "artifacts/stage6/stage6_features.npz", allow_pickle=True)
    payload = {}
    predictions = {}
    for k in (64, 128, 256, 388):
        ztr = np.asarray(base[f"z_{k}_train"], dtype=np.float64)
        zva = np.asarray(base[f"z_{k}_val"], dtype=np.float64)
        result = per_output_ridge(ztr, targets["u_train"], zva, targets["u_val"])
        report = metrics(result["pred"], targets["u_val"], targets["half"])
        report["feature_dim"] = int(k)
        payload[str(k)] = report
        predictions[f"pred_{k}"] = result["pred"].astype(np.float32)
        log(f"pca {k}: MAE {report['mae']:.5f}, R2 {report['mean_r2']:.3f}")
    whitened = group_whiten(train["features"], val["features"], groups, 320)
    result = per_output_ridge(
        whitened["train"].astype(np.float64),
        targets["u_train"],
        whitened["validation"].astype(np.float64),
        targets["u_val"],
    )
    report = metrics(result["pred"], targets["u_val"], targets["half"])
    report["feature_dim"] = 320
    payload["320"] = report
    predictions["pred_320"] = result["pred"].astype(np.float32)
    np.savez_compressed(OUT / "pca_pred.npz", **predictions)
    save_json(OUT / "pca_curve.json", payload)
    log(f"pca 320: MAE {report['mae']:.5f}, R2 {report['mean_r2']:.3f}")
    return payload


def pls2_fit(X: np.ndarray, Y: np.ndarray, max_components: int = 128):
    """Standard PLS2 with deflation; returns W* so that latent = X_centered @ W*."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    Xr = X - x_mean
    Yr = Y - y_mean
    weights, loadings = [], []
    for _ in range(int(max_components)):
        cross = Xr.T @ Yr
        u, s, _ = np.linalg.svd(cross, full_matrices=False)
        if s.size == 0 or s[0] <= 1e-12:
            break
        w = u[:, 0]
        t = Xr @ w
        tt = float(t @ t)
        if tt <= 1e-12:
            break
        p = Xr.T @ t / tt
        q = Yr.T @ t / tt
        weights.append(w)
        loadings.append(p)
        Xr = Xr - np.outer(t, p)
        Yr = Yr - np.outer(t, q)
    if not weights:
        return np.zeros((X.shape[1], 0)), x_mean, y_mean
    W = np.stack(weights, axis=1)
    P = np.stack(loadings, axis=1)
    Wstar = W @ np.linalg.inv(P.T @ W)
    return Wstar, x_mean, y_mean


def stage_pls():
    train = load_rational("train")
    val = load_rational("validation")
    targets = load_targets()
    Xtr_raw, Xva_raw, _, _ = standardize(train["features"], val["features"])
    Xtr_raw = Xtr_raw - Xtr_raw.mean(axis=0)
    Xva_raw = Xva_raw - Xtr_raw.mean(axis=0)
    Wstar, _, _ = pls2_fit(Xtr_raw, targets["u_train"], max_components=128)
    payload = {}
    predictions = {}
    for d in (16, 24, 32, 41, 64, 96, 128):
        d = min(int(d), Wstar.shape[1])
        if d <= 0:
            continue
        ztr = Xtr_raw @ Wstar[:, :d]
        zva = Xva_raw @ Wstar[:, :d]
        result = per_output_ridge(ztr, targets["u_train"], zva, targets["u_val"])
        report = metrics(result["pred"], targets["u_val"], targets["half"])
        report["latent_dim"] = int(d)
        payload[str(d)] = report
        predictions[f"pred_{d}"] = result["pred"].astype(np.float32)
        log(f"pls {d}: MAE {report['mae']:.5f}, R2 {report['mean_r2']:.3f}")
    np.savez_compressed(OUT / "pls_pred.npz", **predictions)
    save_json(OUT / "pls_curve.json", payload)
    return payload


def stage_groups():
    train = load_rational("train")
    val = load_rational("validation")
    targets = load_targets()
    groups = group_indices()
    combinations = {
        "G1_D": ["D"],
        "G2_S11": ["A"],
        "G3_S21": ["B"],
        "G4_D_S11": ["D", "A"],
        "G5_D_S21": ["D", "B"],
        "G6_all": ["D", "A", "B"],
    }
    payload = {}
    predictions = {}
    for name, keys in combinations.items():
        idx = np.concatenate([groups[key] for key in keys])
        Xtr, Xva, _, _ = standardize(train["features"][:, idx], val["features"][:, idx])
        result = per_output_ridge(Xtr, targets["u_train"], Xva, targets["u_val"])
        report = metrics(result["pred"], targets["u_val"], targets["half"])
        report["groups"] = keys
        report["feature_dim"] = int(len(idx))
        payload[name] = report
        if name == "G6_all":
            predictions["pred_G6"] = result["pred"].astype(np.float32)
        log(f"{name}: MAE {report['mae']:.5f}, R2 {report['mean_r2']:.3f}")
    np.savez_compressed(OUT / "group_pred.npz", **predictions)
    save_json(OUT / "group_ablation.json", payload)
    return payload


def stage_probes():
    train = load_rational("train")
    val = load_rational("validation")
    targets = load_targets()
    variants = {
        "P1_eta002": (0.02,),
        "P2_eta002_005": (0.02, 0.05),
        "P3_eta002_005_010": (0.02, 0.05, 0.10),
    }
    payload = {}
    predictions = {}
    for name, etas in variants.items():
        Xtr = probe_features(train["d"], train["a"], train["b"], etas=etas).T
        Xva = probe_features(val["d"], val["a"], val["b"], etas=etas).T
        finite = np.isfinite(Xtr).all() and np.isfinite(Xva).all()
        report = {"finite": bool(finite), "feature_dim": int(Xtr.shape[1])}
        if finite:
            Xtr_s, Xva_s, _, _ = standardize(Xtr, Xva)
            result = per_output_ridge(Xtr_s, targets["u_train"], Xva_s, targets["u_val"])
            report.update(metrics(result["pred"], targets["u_val"], targets["half"]))
            predictions[f"pred_{name}"] = result["pred"].astype(np.float32)
        payload[name] = report
        log(f"{name}: MAE {report.get('mae', float('nan')):.5f}, finite {finite}")
    Xtr = probe_features(train["d"], train["a"], train["b"], include_denominator=True).T
    Xva = probe_features(val["d"], val["a"], val["b"], include_denominator=True).T
    finite = np.isfinite(Xtr).all() and np.isfinite(Xva).all()
    report = {"finite": bool(finite), "feature_dim": int(Xtr.shape[1])}
    if finite:
        Xtr_s, Xva_s, _, _ = standardize(Xtr, Xva)
        result = per_output_ridge(Xtr_s, targets["u_train"], Xva_s, targets["u_val"])
        report.update(metrics(result["pred"], targets["u_val"], targets["half"]))
        predictions["pred_P4"] = result["pred"].astype(np.float32)
    payload["P4_S_plus_denominator"] = report
    np.savez_compressed(OUT / "probe_pred.npz", **predictions)
    save_json(OUT / "complex_probes.json", payload)
    return payload

    log(f"full ridge: MAE {report['mae']:.5f}, physical {report['physical_mae']:.5f}, R2 {report['mean_r2']:.3f}")
    return report



def quadratic_features(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    d = z.shape[1]
    parts = [z]
    parts.append(z ** 2)
    cross = []
    for i in range(d):
        for j in range(i + 1, d):
            cross.append(z[:, i] * z[:, j])
    if cross:
        parts.append(np.stack(cross, axis=1))
    return np.concatenate(parts, axis=1)


def stage_quadratic():
    train = load_rational("train")
    val = load_rational("validation")
    targets = load_targets()
    Xtr, Xva, _, _ = standardize(train["features"], val["features"])
    xmean = Xtr.mean(axis=0)
    Xtr = Xtr - xmean
    Xva = Xva - xmean
    Wstar, _, _ = pls2_fit(Xtr, targets["u_train"], max_components=64)
    payload = {}
    predictions = {}
    for d in (16, 24, 32):
        ztr = Xtr @ Wstar[:, :d]
        zva = Xva @ Wstar[:, :d]
        Qtr = quadratic_features(ztr)
        Qva = quadratic_features(zva)
        Qtr, Qva, _, _ = standardize(Qtr, Qva)
        result = per_output_ridge(Qtr, targets["u_train"], Qva, targets["u_val"])
        report = metrics(result["pred"], targets["u_val"], targets["half"])
        report["latent_dim"] = d
        report["feature_dim"] = int(Qtr.shape[1])
        payload[str(d)] = report
        predictions[f"pred_quad_{d}"] = result["pred"].astype(np.float32)
        log(f"quadratic {d}: MAE {report['mae']:.5f}, R2 {report['mean_r2']:.3f}")
    np.savez_compressed(OUT / "quad_pred.npz", **predictions)
    save_json(OUT / "quadratic.json", payload)
    return payload



def fit_ridge_xy(X: np.ndarray, Y: np.ndarray, lam: float = 1e-3):
    xm, ym = X.mean(axis=0), Y.mean(axis=0)
    Xc = X - xm
    coef = np.linalg.solve(
        Xc.T @ Xc + float(lam) * np.eye(X.shape[1]), Xc.T @ (Y - ym)
    )
    return lambda Z: (Z - xm) @ coef + ym


def stage_residual():
    train = load_rational("train")
    val = load_rational("validation")
    targets = load_targets()
    Xtr, _, _, _ = standardize(train["features"], val["features"])
    base_idx = np.arange(0, 5000)
    fit_idx = np.arange(5000, 6500)
    eval_idx = np.arange(6500, 8000)
    base = fit_ridge_xy(Xtr[base_idx], targets["u_train"][base_idx], lam=1e-3)
    base_fit_pred = base(Xtr[fit_idx])
    base_eval_pred = base(Xtr[eval_idx])
    residual_fit = targets["u_train"][fit_idx] - base_fit_pred
    residual_model = fit_ridge_xy(Xtr[fit_idx], residual_fit, lam=1e-3)
    combined = base_eval_pred + residual_model(Xtr[eval_idx])
    target_eval = targets["u_train"][eval_idx]
    residual_eval = target_eval - base_eval_pred
    residual_eval_pred = residual_model(Xtr[eval_idx])
    ss_res = ((residual_eval - residual_eval_pred) ** 2).sum(axis=0)
    ss_tot = ((residual_eval - residual_eval.mean(axis=0)) ** 2).sum(axis=0)
    payload = {
        "base_eval_mae": float(np.abs(base_eval_pred - target_eval).mean()),
        "combined_eval_mae": float(np.abs(combined - target_eval).mean()),
        "residual_mae": float(np.abs(residual_fit).mean()),
        "residual_eval_r2_mean": float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12))),
        "residual_eval_r2_median": float(np.median(1.0 - ss_res / np.maximum(ss_tot, 1e-12))),
        "n_base": int(len(base_idx)),
        "n_residual_fit": int(len(fit_idx)),
        "n_eval": int(len(eval_idx)),
        "lambda": 1e-3,
    }
    save_json(OUT / "residual_structure.json", payload)
    log(f"residual: base {payload['base_eval_mae']:.5f} -> combined {payload['combined_eval_mae']:.5f}")
    return payload



def _best(report: dict, metric: str = "physical_mae"):
    valid = {k: v for k, v in report.items() if isinstance(v, dict) and metric in v}
    return min(valid, key=lambda k: valid[k][metric]) if valid else None


def stage_oneshot():
    targets = load_targets()
    m_val = targets["m_val"]
    center = targets["center"]
    half = targets["half"]
    m_star = targets["m_star"]
    sim = load_simulator()
    candidates = {}

    full = json.loads((OUT / "full_coefficients.json").read_text())
    candidates["full_coefficients"] = (OUT / "full_pred.npz", "pred", full)

    pca = json.loads((OUT / "pca_curve.json").read_text())
    key = _best(pca)
    candidates[f"pca_{key}"] = (OUT / "pca_pred.npz", f"pred_{key}", pca[key])

    pls = json.loads((OUT / "pls_curve.json").read_text())
    key = _best(pls)
    candidates[f"pls_{key}"] = (OUT / "pls_pred.npz", f"pred_{key}", pls[key])

    probes = json.loads((OUT / "complex_probes.json").read_text())
    key = _best(probes)
    probe_pred_key = "pred_P4" if key == "P4_S_plus_denominator" else f"pred_{key}"
    candidates[f"probe_{key}"] = (OUT / "probe_pred.npz", probe_pred_key, probes[key])

    quad = json.loads((OUT / "quadratic.json").read_text())
    key = _best(quad)
    candidates[f"quad_{key}"] = (OUT / "quad_pred.npz", f"pred_quad_{key}", quad[key])

    fullraw = json.loads((OUT / "full_plus_raw.json").read_text())
    fullraw_key = _best(fullraw)
    candidates[fullraw_key] = (OUT / "full_raw_pred.npz", f"pred_{fullraw_key}", fullraw[fullraw_key])

    summary = {}
    for name, (path, pred_key, report) in candidates.items():
        data = np.load(path, allow_pickle=True)
        pred_u = data[pred_key].astype(np.float64)
        pred_m = center[None, :] + half[None, :] * pred_u
        physical = np.abs(pred_m - m_val)
        delta = m_star[None, :] - pred_m
        metrics_one, _ = evaluate_delta(sim, m_val, delta)
        summary[name] = {
            "pass_rate": metrics_one["pass_rate"],
            "mean_max_miss_db": metrics_one["mean_max_miss_db"],
            "median_max_miss_db": metrics_one["median_max_miss_db"],
            "p90_max_miss_db": metrics_one["p90_max_miss_db"],
            "worst_max_miss_db": metrics_one["worst_max_miss_db"],
            "physical_mae": report.get("physical_mae"),
            "dangerous_physical_mae": report.get("dangerous_physical_mae"),
            "dangerous_physical_p95": report.get("dangerous_physical_p95"),
            "mean_r2": report.get("mean_r2"),
            "count_r2_below_0_1": report.get("count_r2_below_0_1"),
        }
        log(f"oneshot {name}: pass {summary[name]['pass_rate']:.3f}, mean {summary[name]['mean_max_miss_db']:.2f} dB")
    save_json(OUT / "stage7_oneshot.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("audit", "full", "fullraw", "pca", "pls", "groups", "probes", "quadratic", "residual", "oneshot", "all"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage in ("audit", "all"):
        stage_audit(workers=args.workers)
    if args.stage in ("full", "all"):
        stage_full()
    if args.stage in ("fullraw", "all"):
        stage_full_raw()
    if args.stage in ("pca", "all"):
        stage_pca()
    if args.stage in ("pls", "all"):
        stage_pls()
    if args.stage in ("groups", "all"):
        stage_groups()
    if args.stage in ("probes", "all"):
        stage_probes()
    if args.stage in ("quadratic", "all"):
        stage_quadratic()
    if args.stage in ("residual", "all"):
        stage_residual()
    if args.stage in ("oneshot", "all"):
        stage_oneshot()


if __name__ == "__main__":
    main()

