"""Pilot study: per-state compliant teacher for CM19 (no new states).

Stages:
    line      current -> anchor line search for 256 train pilot states
    teacher   constrained local feasible-point search from the line teacher
    analyze   geometry, sparsity, low-rank, consistency, cheap learnability
    all       line + teacher + analyze

All simulator use is on existing cached states / candidate corrected vectors.
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
from solution.pairwise import normalized_rmse
from solution.simulator import load_simulator, requirement_infos
from solution.teacher import line_search_anchor, local_teacher_search
from solution.utils import save_json

OUT = ROOT / "artifacts/teacher/pilot"
OUT.mkdir(parents=True, exist_ok=True)
PAIRWISE = ROOT / "artifacts/analysis/pairwise_control"

_SIM = None
_BASIS = None
_ONEHOT = None
_MAX_ITER = 8
_POPULATION = 32


def log(message: str) -> None:
    print(f"[teacher] {message}", flush=True)


def _init_worker(
    basis_path: str | None,
    onehot: np.ndarray | None,
    max_iter: int,
    population: int,
) -> None:
    global _SIM, _BASIS, _ONEHOT, _MAX_ITER, _POPULATION
    _SIM = load_simulator()
    if basis_path is not None:
        with np.load(basis_path) as archive:
            _BASIS = archive["basis"].astype(np.float64)
    else:
        _BASIS = np.empty((0, 41))
    _ONEHOT = onehot if onehot is not None else np.empty((0, 41))
    _MAX_ITER = int(max_iter)
    _POPULATION = int(population)


def _line_task(task):
    idx, m_current, m_anchor, gamma = task
    result = line_search_anchor(_SIM, m_current, m_anchor, required_margin=gamma)
    return idx, result


def _teacher_task(task):
    idx, m_current, start_m, gamma, seed = task
    result = local_teacher_search(
        _SIM,
        m_current,
        start_m,
        required_margin=gamma,
        basis_dirs=_BASIS,
        onehot_dirs=_ONEHOT,
        max_iter=_MAX_ITER,
        population=_POPULATION,
        seed=seed,
    )
    return idx, result


def load_pilot(n: int):
    train = CM19ResponseDataset("cache/dataset", split="train", limit=n)
    rows = np.asarray(train._indices)
    responses = np.array(train.responses[rows])
    m_current = np.array(train.m_current[rows])
    seeds = np.asarray(train.seeds_slice())
    return seeds, responses, m_current


def _u_of(m: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    return (np.asarray(m, dtype=np.float64) - center) / half


def run_line(seeds, m_current, m_anchor, gamma, workers):
    tasks = [(i, m_current[i], m_anchor, gamma) for i in range(len(m_current))]
    start = time.perf_counter()
    with mp.get_context("fork").Pool(
        processes=int(workers),
        initializer=_init_worker,
        initargs=(None, None, 0, 0),
    ) as pool:
        results = pool.map(_line_task, tasks, chunksize=8)
    elapsed = time.perf_counter() - start
    order = np.argsort([idx for idx, _ in results])
    results = [results[i] for i in order]
    t_values = np.array([r["t"] for _, r in results])
    m_line = np.stack([r["m"] for _, r in results])
    margins = np.stack([r["margins"] for _, r in results])
    min_margin = np.array([r["min_margin"] for _, r in results])
    calls = np.array([r["calls"] for _, r in results])
    found = np.array([r["found"] for _, r in results])
    payload = {
        "seeds": seeds,
        "m_current": m_current,
        "m_line": m_line,
        "t": t_values,
        "margins": margins,
        "min_margin": min_margin,
        "calls": calls,
        "found": found,
        "gamma": np.asarray(gamma),
        "wall_seconds": np.asarray(elapsed),
    }
    np.savez_compressed(OUT / f"line_gamma{gamma:.2f}.npz", **payload)
    summary = {
        "gamma": float(gamma),
        "states": int(len(t_values)),
        "found_all": bool(found.all()),
        "t_quantiles": {
            "median": float(np.median(t_values)),
            "p10": float(np.quantile(t_values, 0.10)),
            "p25": float(np.quantile(t_values, 0.25)),
            "p75": float(np.quantile(t_values, 0.75)),
            "p90": float(np.quantile(t_values, 0.90)),
            "min": float(t_values.min()),
            "max": float(t_values.max()),
        },
        "fraction_t_lt": {
            "0.25": float(np.mean(t_values < 0.25)),
            "0.50": float(np.mean(t_values < 0.50)),
            "0.75": float(np.mean(t_values < 0.75)),
            "0.90": float(np.mean(t_values > 0.90)),
        },
        "min_margin_quantiles": {
            "median": float(np.median(min_margin)),
            "p10": float(np.quantile(min_margin, 0.10)),
            "min": float(min_margin.min()),
        },
        "simulator_calls": int(calls.sum()),
        "wall_seconds": float(elapsed),
    }
    save_json(OUT / f"line_gamma{gamma:.2f}.json", summary)
    return payload, summary


def run_teacher(m_current, line_payload, gamma, workers, max_iter, population):
    start_m = line_payload["m_line"]
    basis, onehot = build_basis()
    np.savez(OUT / "basis_dirs.npz", basis=basis, onehot=onehot)
    tasks = [
        (i, m_current[i], start_m[i], gamma, 20260000 + i) for i in range(len(m_current))
    ]
    start = time.perf_counter()
    with mp.get_context("fork").Pool(
        processes=int(workers),
        initializer=_init_worker,
        initargs=(str(OUT / "basis_dirs.npz"), onehot, max_iter, population),
    ) as pool:
        results = pool.map(_teacher_task, tasks, chunksize=4)
    elapsed = time.perf_counter() - start
    order = np.argsort([idx for idx, _ in results])
    results = [results[i] for i in order]
    delta = np.stack([r.delta for _, r in results])
    corrected = np.stack([r.corrected for _, r in results])
    margins = np.stack([r.margins for _, r in results])
    min_margin = np.array([r.min_margin for _, r in results])
    passed = np.array([r.passed for _, r in results])
    calls = np.array([r.calls for _, r in results])
    iterations = np.array([r.iterations for _, r in results])
    payload = {
        "seeds": line_payload["seeds"],
        "m_current": m_current,
        "delta": delta,
        "m_teacher": corrected,
        "margins": margins,
        "min_margin": min_margin,
        "passed": passed,
        "calls": calls,
        "iterations": iterations,
        "gamma": np.asarray(gamma),
    }
    np.savez_compressed(OUT / f"teacher_gamma{gamma:.2f}.npz", **payload)
    history = {
        "gamma": float(gamma),
        "states": int(len(delta)),
        "passed_all": bool(passed.all()),
        "passed_count": int(passed.sum()),
        "call_count": int(calls.sum()),
        "iterations_mean": float(iterations.mean()),
        "iterations_max": int(iterations.max()),
        "min_margin_quantiles": {
            "median": float(np.median(min_margin)),
            "p10": float(np.quantile(min_margin, 0.10)),
            "min": float(min_margin.min()),
        },
        "wall_seconds": float(elapsed),
        "max_iter": int(max_iter),
        "population": int(population),
    }
    save_json(OUT / f"teacher_gamma{gamma:.2f}.json", history)
    return payload, history


def build_basis():
    with np.load(PAIRWISE / "local_control_map_A.npz") as archive:
        vh = archive["Vh_512"].astype(np.float64)
    observable = vh[:16].T          # 16 strongest parameter directions
    weak = vh[16:32].T              # next 16 (weak) directions
    basis = np.concatenate([observable, weak], axis=1).T  # (32,41)
    sensitive = [8, 32, 37, 35, 22, 23, 10, 24, 31, 5]
    strong = [0, 2, 3, 17]
    onehot = np.zeros((len(sensitive) + len(strong), 41))
    for row, idx in enumerate(sensitive + strong):
        onehot[row, idx] = 1.0
    return basis, onehot


def _ridge_metrics(X, Y, n_fit, lambdas):
    Xf, Yf = X[:n_fit], Y[:n_fit]
    Xh, Yh = X[n_fit:], Y[n_fit:]
    mean = Xf.mean(axis=0, keepdims=True)
    std = Xf.std(axis=0, keepdims=True) + 1e-8
    Xf = (Xf - mean) / std
    Xh = (Xh - mean) / std
    mean_y = Yf.mean(axis=0, keepdims=True)
    Yf = Yf - mean_y
    best = None
    for lam in lambdas:
        gram = Xf.T @ Xf + lam * np.eye(Xf.shape[1])
        W = np.linalg.solve(gram, Xf.T @ Yf)
        pred = Xh @ W + mean_y
        err = np.sqrt(np.mean((pred - Yh) ** 2)) / (np.sqrt(np.mean(Yh**2)) + 1e-12)
        if best is None or err < best[0]:
            best = (err, lam, pred)
    err, lam, pred = best
    mae = float(np.mean(np.abs(pred - Yh)))
    cosine = float(
        np.mean(
            np.sum(pred * Yh, axis=1)
            / (np.linalg.norm(pred, axis=1) * np.linalg.norm(Yh, axis=1) + 1e-12)
        )
    )
    rel_l2 = float(
        np.mean(
            np.linalg.norm(pred - Yh, axis=1)
            / (np.linalg.norm(Yh, axis=1) + 1e-12)
        )
    )
    return {
        "lambda": float(lam),
        "normalized_rmse": float(err),
        "mae": mae,
        "cosine": cosine,
        "relative_l2": rel_l2,
        "n_fit": int(n_fit),
        "n_holdout": int(len(X) - n_fit),
    }


def analyze(gamma, anchor, line_payload, teacher_payload, n_pilot):
    simulator = load_simulator()
    lower = np.asarray(simulator.lower_bounds)
    upper = np.asarray(simulator.upper_bounds)
    center = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-6)

    m_current = teacher_payload["m_current"]
    u_current = _u_of(m_current, center, half)
    u_anchor = _u_of(anchor[None, :], center, half)[0]
    du_anchor = u_anchor[None, :] - u_current
    du_teacher = teacher_payload["delta"] / half[None, :]
    du_line = (line_payload["m_line"] - m_current) / half[None, :]

    def norms(x):
        return {
            "l2_mean": float(np.mean(np.linalg.norm(x, axis=1))),
            "l2_median": float(np.median(np.linalg.norm(x, axis=1))),
            "l2_p90": float(np.quantile(np.linalg.norm(x, axis=1), 0.90)),
            "l1_mean": float(np.mean(np.abs(x).sum(axis=1))),
            "linf_mean": float(np.mean(np.abs(x).max(axis=1))),
        }

    anchor_norms = norms(du_anchor)
    line_norms = norms(du_line)
    teacher_norms = norms(du_teacher)
    reduction_teacher = np.linalg.norm(du_teacher, axis=1) / (
        np.linalg.norm(du_anchor, axis=1) + 1e-12
    )
    reduction_line = np.linalg.norm(du_line, axis=1) / (
        np.linalg.norm(du_anchor, axis=1) + 1e-12
    )

    sparsity = {}
    for threshold in (0.01, 0.05, 0.10):
        counts = (np.abs(du_teacher) > threshold).sum(axis=1)
        sparsity[str(threshold)] = {
            "mean_changed": float(counts.mean()),
            "median_changed": float(np.median(counts)),
            "p90_changed": float(np.quantile(counts, 0.90)),
        }
    per_param_mean = np.abs(du_teacher).mean(axis=0)
    per_param_p90 = np.quantile(np.abs(du_teacher), 0.90, axis=0)
    per_param_unchanged = np.mean(np.abs(du_teacher) < 1e-3, axis=0)

    svd_teacher = np.linalg.svd(du_teacher, compute_uv=False)
    svd_anchor = np.linalg.svd(du_anchor, compute_uv=False)
    var_teacher = np.cumsum(svd_teacher**2) / np.sum(svd_teacher**2)
    var_anchor = np.cumsum(svd_anchor**2) / np.sum(svd_anchor**2)

    with np.load(PAIRWISE / "local_control_map_A.npz") as archive:
        vh = archive["Vh_512"].astype(np.float64)
    projection = {}
    for r in (8, 16, 24, 32):
        Vr = vh[:r].T
        obs_teacher = np.sum((du_teacher @ Vr) ** 2, axis=1) / (
            np.sum(du_teacher**2, axis=1) + 1e-30
        )
        obs_anchor = np.sum((du_anchor @ Vr) ** 2, axis=1) / (
            np.sum(du_anchor**2, axis=1) + 1e-30
        )
        projection[str(r)] = {
            "teacher_observable_fraction_mean": float(np.mean(obs_teacher)),
            "anchor_observable_fraction_mean": float(np.mean(obs_anchor)),
        }

    return {
        "gamma": float(gamma),
        "states": int(n_pilot),
        "norms": {
            "anchor": anchor_norms,
            "line": line_norms,
            "teacher": teacher_norms,
        },
        "reduction": {
            "teacher_over_anchor": {
                "mean": float(reduction_teacher.mean()),
                "median": float(np.median(reduction_teacher)),
                "p10": float(np.quantile(reduction_teacher, 0.10)),
                "p90": float(np.quantile(reduction_teacher, 0.90)),
            },
            "line_over_anchor": {
                "mean": float(reduction_line.mean()),
                "median": float(np.median(reduction_line)),
                "p10": float(np.quantile(reduction_line, 0.10)),
                "p90": float(np.quantile(reduction_line, 0.90)),
            },
        },
        "sparsity": sparsity,
        "per_parameter_mean_abs": per_param_mean.tolist(),
        "per_parameter_p90_abs": per_param_p90.tolist(),
        "per_parameter_unchanged_fraction": per_param_unchanged.tolist(),
        "teacher_correction_singular_values": svd_teacher.tolist(),
        "teacher_correction_cumulative_variance": var_teacher.tolist(),
        "anchor_correction_cumulative_variance": var_anchor.tolist(),
        "projection": projection,
        "teacher_min_margin": {
            "median": float(np.median(teacher_payload["min_margin"])),
            "p10": float(np.quantile(teacher_payload["min_margin"], 0.10)),
            "min": float(teacher_payload["min_margin"].min()),
        },
    }


def consistency_and_learnability(teacher_payload, anchor, n_pilot):
    import torch

    simulator = load_simulator()
    lower = np.asarray(simulator.lower_bounds)
    upper = np.asarray(simulator.upper_bounds)
    center = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-6)
    with np.load(PAIRWISE / "latents.npz") as archive:
        z = archive["z_train"][:n_pilot, :512].astype(np.float64)
        u_current = archive["u_train"][:n_pilot].astype(np.float64)
    u_anchor = _u_of(anchor[None, :], center, half)[0]
    du_teacher = teacher_payload["delta"] / half[None, :]
    du_anchor = u_anchor[None, :] - u_current

    def pairwise(matrix):
        t = torch.as_tensor(matrix, dtype=torch.float64)
        dist = torch.cdist(t, t)
        dist.fill_diagonal_(float("inf"))
        return dist.numpy()

    dz = pairwise(z)
    d_teacher = pairwise(du_teacher)
    d_anchor = pairwise(du_anchor)
    k = 5
    nearest = np.argsort(dz, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_pilot), k)
    cols = nearest.flatten()
    nearest_teacher = d_teacher[rows, cols]
    nearest_anchor = d_anchor[rows, cols]
    rng = np.random.default_rng(2026)
    rand_rows = rng.integers(0, n_pilot, size=rows.size)
    rand_cols = rng.integers(0, n_pilot, size=rows.size)
    mask = rand_rows != rand_cols
    random_teacher = d_teacher[rand_rows[mask], rand_cols[mask]]
    random_anchor = d_anchor[rand_rows[mask], rand_cols[mask]]
    offdiag = ~np.eye(n_pilot, dtype=bool)
    corr_teacher = float(
        np.corrcoef(dz[offdiag], d_teacher[offdiag])[0, 1]
    )
    corr_anchor = float(
        np.corrcoef(dz[offdiag], d_anchor[offdiag])[0, 1]
    )

    targets = {
        "teacher_delta": du_teacher,
        "anchor_delta": du_anchor,
        "absolute_state": u_current,
    }
    lambdas = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    n_fit = int(round(0.8 * n_pilot))
    learnability = {}
    for K in (64, 128, 256, 512):
        features = z[:, :K]
        learnability[str(K)] = {
            name: _ridge_metrics(features, target, n_fit, lambdas)
            for name, target in targets.items()
        }

    return {
        "response_neighbor_consistency": {
            "k": k,
            "nearest_teacher_delta_distance_mean": float(nearest_teacher.mean()),
            "random_teacher_delta_distance_mean": float(random_teacher.mean()),
            "nearest_anchor_delta_distance_mean": float(nearest_anchor.mean()),
            "random_anchor_delta_distance_mean": float(random_anchor.mean()),
            "ratio_teacher": float(nearest_teacher.mean() / random_teacher.mean()),
            "ratio_anchor": float(nearest_anchor.mean() / random_anchor.mean()),
            "corr_response_distance_teacher_delta": corr_teacher,
            "corr_response_distance_anchor_delta": corr_anchor,
        },
        "learnability_ridge": learnability,
    }


def _plot_pilot(analysis, teacher_payload, line_payload, gamma):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    simulator = load_simulator()
    lower = np.asarray(simulator.lower_bounds)
    upper = np.asarray(simulator.upper_bounds)
    center = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-6)
    m_current = teacher_payload["m_current"]
    u_current = _u_of(m_current, center, half)
    u_anchor = _u_of(
        np.load(ROOT / "artifacts/common/anchor.npz")["m_star"][None, :], center, half
    )[0]
    du_anchor = u_anchor[None, :] - u_current
    du_teacher = teacher_payload["delta"] / half[None, :]
    ratios = np.linalg.norm(du_teacher, axis=1) / (
        np.linalg.norm(du_anchor, axis=1) + 1e-12
    )

    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(line_payload["t"], bins=30)
    ax.set(xlabel="t_line", ylabel="states", title=f"Anchor-line boundary (gamma={gamma})")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / f"t_line_gamma{gamma:.2f}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(ratios, bins=30)
    ax.set(xlabel="||du_teacher|| / ||du_anchor||", ylabel="states",
           title=f"Correction reduction (gamma={gamma})")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / f"reduction_gamma{gamma:.2f}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sv = np.asarray(analysis["teacher_correction_singular_values"])
    ax.semilogy(np.arange(1, len(sv) + 1), sv, marker="o", markersize=3)
    ax.set(xlabel="index", ylabel="singular value", title=f"Teacher corrections (gamma={gamma})")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(plot_dir / f"teacher_singular_gamma{gamma:.2f}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    counts = (np.abs(du_teacher) > 0.05).sum(axis=1)
    ax.hist(counts, bins=np.arange(0, 42, 2))
    ax.set(xlabel="# parameters with |du| > 0.05", ylabel="states",
           title=f"Teacher correction sparsity (gamma={gamma})")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / f"sparsity_gamma{gamma:.2f}.png", dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.25])
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument(
        "--stage", choices=("line", "teacher", "analyze", "all"), default="all"
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    seeds, responses, m_current = load_pilot(args.pilot)
    anchor = np.load(ROOT / "artifacts/common/anchor.npz")["m_star"].astype(np.float64)
    log(f"pilot states={len(m_current)} seeds={seeds[0]}..{seeds[-1]}")

    for gamma in args.gammas:
        line_path = OUT / f"line_gamma{gamma:.2f}.npz"
        if line_path.exists() and not args.rebuild:
            line_payload = dict(np.load(line_path))
            log(f"loaded line search gamma={gamma}")
        else:
            line_payload, line_summary = run_line(
                seeds, m_current, anchor, gamma, args.workers
            )
            log(f"line search gamma={gamma}: {json.dumps(line_summary['t_quantiles'])}")

        if args.stage == "line":
            continue

        teacher_path = OUT / f"teacher_gamma{gamma:.2f}.npz"
        if teacher_path.exists() and not args.rebuild:
            teacher_payload = dict(np.load(teacher_path))
            log(f"loaded teacher gamma={gamma}")
        else:
            teacher_payload, teacher_summary = run_teacher(
                m_current,
                line_payload,
                gamma,
                args.workers,
                args.max_iter,
                args.population,
            )
            log(
                f"teacher gamma={gamma}: passed {teacher_summary['passed_count']}/"
                f"{teacher_summary['states']}, calls {teacher_summary['call_count']}"
            )

        if args.stage in ("teacher",):
            continue

        analysis = analyze(gamma, anchor, line_payload, teacher_payload, len(m_current))
        consistency = consistency_and_learnability(
            teacher_payload, anchor, len(m_current)
        )
        analysis.update(consistency)
        analysis["simulator_calls"] = int(
            line_payload["calls"].sum() + teacher_payload["calls"].sum()
        )
        save_json(OUT / f"analysis_gamma{gamma:.2f}.json", analysis)
        _plot_pilot(analysis, teacher_payload, line_payload, gamma)
        log(
            f"gamma={gamma}: teacher/anchor norm median "
            f"{analysis['reduction']['teacher_over_anchor']['median']:.3f}, "
            f"ridge teacher nrmse {analysis['learnability_ridge']['512']['teacher_delta']['normalized_rmse']:.3f}, "
            f"absolute state nrmse {analysis['learnability_ridge']['512']['absolute_state']['normalized_rmse']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
