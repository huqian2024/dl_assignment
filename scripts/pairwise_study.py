"""Pairwise / observable-subspace control study for CM19.

Stages (cache in ``artifacts/analysis/pairwise_control``):

    pca         response PCA on train split (K=64..512)
    pairs       deterministic local/response-local/medium/random pair indices
    jacobian    Delta z ~= A Delta u, SVD -> observable parameter subspace
    residual    project the old Route-3 state residual onto observable/weak
    controllers linear c = W Delta z, anchor-target one-shot evaluation
    all         run everything

No new states are generated: only cached train/validation/internal splits and
the existing compliant anchor are used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from solution.data import CM19ResponseDataset
from solution.pairwise import (
    PAIR_TYPES,
    evaluate_delta,
    normalized_rmse,
    normalized_state,
    pca_response,
    project_latent,
    ridge_controller,
    ridge_jacobian,
    sample_pairs,
)
from solution.simulator import load_simulator
from solution.utils import get_device, resolve, save_json

OUT = ROOT / "artifacts/analysis/pairwise_control"
OUT.mkdir(parents=True, exist_ok=True)
R_CANDIDATES = (8, 12, 16, 20, 24, 32, 41)
K_CANDIDATES = (128, 256, 512)
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def log(message: str) -> None:
    print(f"[pairwise] {message}", flush=True)


def load_arrays(dataset_dir: str):
    train = CM19ResponseDataset(dataset_dir, split="train")
    val = CM19ResponseDataset(dataset_dir, split="validation")
    internal = CM19ResponseDataset(dataset_dir, split="internal_test")
    return {
        "train": (np.array(train.responses), np.array(train.m_current)),
        "validation": (np.array(val.responses), np.array(val.m_current)),
        "internal_test": (np.array(internal.responses), np.array(internal.m_current)),
    }


def save_npz(path: Path, **arrays) -> None:
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def stage_pca(arrays, simulator, device, k_max: int, rebuild: bool) -> dict:
    path = OUT / "pca_latent.npz"
    if path.exists() and not rebuild:
        pca = load_npz(path)
        log(f"loaded PCA from {path}")
    else:
        start = time.perf_counter()
        pca = pca_response(arrays["train"][0], k_max=k_max, device=device)
        save_npz(path, **pca)
        log(f"PCA done in {time.perf_counter() - start:.1f}s -> {path}")
    spectrum = {
        "k_max": int(len(pca["explained_ratio"])),
        "explained_ratio": pca["explained_ratio"].tolist(),
        "cumulative_ratio": pca["cumulative_ratio"].tolist(),
        "singular_values": pca["singular_values"].tolist(),
        "cumulative_at": {
            str(k): float(pca["cumulative_ratio"][k - 1])
            for k in (64, 128, 256, 512)
            if k <= len(pca["cumulative_ratio"])
        },
    }
    save_json(OUT / "pca_spectrum.json", spectrum)

    lower = np.asarray(simulator.lower_bounds)
    upper = np.asarray(simulator.upper_bounds)
    anchor = np.load(ROOT / "artifacts/common/anchor.npz")["m_star"]
    anchor_response = np.asarray(simulator.simulate(anchor), dtype=np.float32)[None]
    latents = {}
    for split in ("train", "validation", "internal_test"):
        responses, m_current = arrays[split]
        latents[f"z_{split}"] = project_latent(responses, pca["mean"], pca["basis"])
        latents[f"u_{split}"], center, half = normalized_state(m_current, lower, upper)
    latents["z_anchor"] = project_latent(anchor_response, pca["mean"], pca["basis"])[0]
    latents["u_anchor"], _, _ = normalized_state(anchor[None, :], lower, upper)
    latents["u_anchor"] = latents["u_anchor"][0]
    save_npz(OUT / "latents.npz", **latents)
    return {**pca, **latents, "center": center, "half": half}


def stage_jacobian(latents: dict, pairs: dict, rebuild: bool) -> dict:
    path = OUT / "local_control_map_A.npz"
    if path.exists() and not rebuild:
        archive = load_npz(path)
        log(f"loaded Jacobian maps from {path}")
    else:
        train_src = pairs["train_src"]
        train_dst = pairs["train_dst"]
        train_type = pairs["train_type"]
        local = train_type == 0
        src = train_src[local]
        dst = train_dst[local]
        fit_mask = src < 7000
        hold_mask = ~fit_mask
        archive = {}
        details = {}
        for K in K_CANDIDATES:
            dz = (latents["z_train"][dst, :K] - latents["z_train"][src, :K]).astype(np.float64)
            du = (latents["u_train"][dst] - latents["u_train"][src]).astype(np.float64)
            best = None
            for lam in LAMBDAS:
                A_fit = ridge_jacobian(du[fit_mask], dz[fit_mask], lam)
                pred = du[hold_mask] @ A_fit.T
                error = normalized_rmse(pred, dz[hold_mask])
                if best is None or error < best[0]:
                    best = (error, lam)
            holdout_error, lam = best
            A = ridge_jacobian(du, dz, lam)
            U, S, Vh = torch.linalg.svd(
                torch.as_tensor(A, dtype=torch.float64), full_matrices=False
            )
            archive[f"A_{K}"] = A.astype(np.float32)
            archive[f"U_{K}"] = U.numpy().astype(np.float32)
            archive[f"S_{K}"] = S.numpy().astype(np.float64)
            archive[f"Vh_{K}"] = Vh.numpy().astype(np.float64)
            archive[f"lambda_{K}"] = np.asarray(lam)
            details[str(K)] = {
                "lambda": float(lam),
                "holdout_dz_normalized_rmse": float(holdout_error),
                "singular_values": S.numpy().tolist(),
                "condition_number": float(S[0] / max(S[-1], 1e-30)),
                "cumulative_squared_energy": (
                    torch.cumsum(S**2, dim=0) / torch.sum(S**2)
                ).numpy().tolist(),
                "adjacent_ratio": (S[:-1] / S[1:]).numpy().tolist(),
            }
        save_npz(path, **archive)
        save_json(OUT / "singular_spectrum.json", details)
        log(f"Jacobian maps fitted -> {path}")

    # choose K by holdout reconstruction (lower is better)
    details = json.loads((OUT / "singular_spectrum.json").read_text())
    best_K = min(K_CANDIDATES, key=lambda k: details[str(k)]["holdout_dz_normalized_rmse"])
    for r in R_CANDIDATES:
        Vh = archive[f"Vh_{best_K}"]
        V_r = Vh[: min(r, Vh.shape[0])].T
        save_npz(OUT / f"observable_bases_r{r}.npz", V_r=V_r, r=np.asarray(r))
    return {
        "archive": archive,
        "details": details,
        "best_K": int(best_K),
    }


def stage_residual(latents: dict, jacobian: dict, rebuild: bool) -> dict:
    path = OUT / "residual_subspace_analysis.json"
    if path.exists() and not rebuild:
        log(f"loaded residual analysis from {path}")
        return json.loads(path.read_text())

    residual_path = ROOT / "artifacts/analysis/final_m32/validation/residuals.npz"
    if not residual_path.exists():
        log(f"WARNING: missing old residual dataset {residual_path}; skipping projection")
        return {}
    with np.load(residual_path) as archive:
        error = archive["error"].astype(np.float64)
        max_miss = archive["max_miss_db"].astype(np.float64)
        passed = archive["passed"].astype(bool)
    half = np.asarray(latents["half"])
    e_u = error / half[None, :]
    total_energy = float(np.mean(np.sum(e_u**2, axis=1)))
    archive = jacobian["archive"]
    best_K = jacobian["best_K"]
    Vh = archive[f"Vh_{best_K}"]

    rows = {}
    for r in R_CANDIDATES:
        V_r = Vh[: min(r, Vh.shape[0])].T
        e_obs = (e_u @ V_r) @ V_r.T
        e_weak = e_u - e_obs
        obs_energy = float(np.mean(np.sum(e_obs**2, axis=1)))
        weak_energy = float(np.mean(np.sum(e_weak**2, axis=1)))
        obs_norm = np.linalg.norm(e_obs, axis=1)
        weak_norm = np.linalg.norm(e_weak, axis=1)
        rows[str(r)] = {
            "observable_energy": obs_energy,
            "weak_energy": weak_energy,
            "observable_fraction": obs_energy / max(total_energy, 1e-30),
            "weak_fraction": weak_energy / max(total_energy, 1e-30),
            "corr_miss_vs_observable_norm": float(np.corrcoef(obs_norm, max_miss)[0, 1]),
            "corr_miss_vs_weak_norm": float(np.corrcoef(weak_norm, max_miss)[0, 1]),
            "mean_miss_top_quartile_observable": float(
                max_miss[obs_norm >= np.quantile(obs_norm, 0.75)].mean()
            ),
            "mean_miss_top_quartile_weak": float(
                max_miss[weak_norm >= np.quantile(weak_norm, 0.75)].mean()
            ),
        }
    result = {
        "source_residual": str(residual_path),
        "best_K": best_K,
        "num_states": int(len(error)),
        "pass_rate": float(passed.mean()),
        "by_r": rows,
    }
    save_json(path, result)
    return result


CONTROLLER_SETS = {
    "C1_local": (0,),
    "C2_local_response": (0, 1),
    "C3_mixed": (0, 1, 2, 3),
}


def fit_controller_table(latents, pairs, jacobian):
    """Ridge controllers for every (controller, K, r, lambda); train-only fitting."""
    best_K = jacobian["best_K"]
    results = {}
    for controller, types in CONTROLLER_SETS.items():
        train_mask = np.isin(pairs["train_type"], types)
        src = pairs["train_src"][train_mask]
        dst = pairs["train_dst"][train_mask]
        fit = src < 7000
        hold = ~fit
        val_mask = np.isin(pairs["val_type"], types)
        vsrc = pairs["val_src"][val_mask]
        vdst = pairs["val_dst"][val_mask]
        results[controller] = {}
        for K in K_CANDIDATES:
            dz_train = (
                latents["z_train"][dst, :K] - latents["z_train"][src, :K]
            ).astype(np.float64)
            du_train = (latents["u_train"][dst] - latents["u_train"][src]).astype(np.float64)
            dz_fit, dz_hold = dz_train[fit], dz_train[hold]
            du_fit, du_hold = du_train[fit], du_train[hold]
            dz_val = (
                latents["z_validation"][vdst, :K] - latents["z_validation"][vsrc, :K]
            ).astype(np.float64)
            du_val = (
                latents["u_validation"][vdst] - latents["u_validation"][vsrc]
            ).astype(np.float64)
            gram_fit = dz_fit.T @ dz_fit
            gram_all = dz_train.T @ dz_train
            eye = np.eye(K)
            A_np = jacobian["archive"].get(f"A_{K}")
            if A_np is None:
                continue
            results[controller][str(K)] = {}
            for r in R_CANDIDATES:
                Vh_K = jacobian["archive"][f"Vh_{K}"]
                V_r = Vh_K[: min(r, 41)].T
                c_fit = du_fit @ V_r
                c_hold = du_hold @ V_r
                c_val = du_val @ V_r
                rhs_fit = dz_fit.T @ c_fit
                rhs_all = dz_train.T @ (du_train @ V_r)
                best = None
                for lam in LAMBDAS:
                    Wt = np.linalg.solve(gram_fit + lam * eye, rhs_fit)
                    pred_hold = dz_hold @ Wt
                    holdout = normalized_rmse(pred_hold, c_hold)
                    if best is None or holdout < best[0]:
                        best = (holdout, lam)
                holdout, lam = best
                Wt = np.linalg.solve(gram_all + lam * eye, rhs_all)
                pred_val = dz_val @ Wt
                du_hat = pred_val @ V_r.T
                dz_hat = du_hat @ A_np.T
                entry = {
                    "lambda": float(lam),
                    "holdout_c_normalized_rmse": float(holdout),
                    "val_c_normalized_rmse": normalized_rmse(pred_val, c_val),
                    "val_du_full_rmse": float(
                        np.sqrt(np.mean((du_hat - du_val) ** 2))
                    ),
                    "val_dz_reconstruction_normalized_rmse": normalized_rmse(
                        dz_hat, dz_val
                    ),
                    "val_dz_reconstruction_r2": float(
                        1.0
                        - np.mean((dz_hat - dz_val) ** 2)
                        / max(np.mean(dz_val**2), 1e-30)
                    ),
                }
                results[controller][str(K)][str(r)] = entry
    return results, best_K


def build_correction(latents, W, V_r, K, split: str, limit: int | None):
    z = latents[f"z_{split}"]
    count = len(z) if limit is None else min(int(limit), len(z))
    anchor = np.asarray(latents["z_anchor"]).reshape(-1)[:K]
    dz = (anchor[None, :] - z[:count, :K]).astype(np.float64)
    c = dz @ W.T
    du = c @ V_r.T
    dm = du * latents["half"][None, :]
    return dm


def stage_pairs(latents: dict, rebuild: bool) -> dict:
    path = OUT / "pair_indices.npz"
    if path.exists() and not rebuild:
        pairs = load_npz(path)
        log(f"loaded pairs from {path}")
    else:
        start = time.perf_counter()
        train_pairs = sample_pairs(
            latents["u_train"],
            latents["z_train"][:, :256],
            seed=2026,
        )
        val_pairs = sample_pairs(
            latents["u_validation"],
            latents["z_validation"][:, :256],
            seed=2027,
        )
        pairs = {
            "train_src": train_pairs["src"],
            "train_dst": train_pairs["dst"],
            "train_type": train_pairs["type"],
            "train_u_distance": train_pairs["u_distance"],
            "train_z_distance": train_pairs["z_distance"],
            "val_src": val_pairs["src"],
            "val_dst": val_pairs["dst"],
            "val_type": val_pairs["type"],
            "val_u_distance": val_pairs["u_distance"],
            "val_z_distance": val_pairs["z_distance"],
        }
        save_npz(path, **pairs)
        log(f"pairs done in {time.perf_counter() - start:.1f}s -> {path}")

    analysis = {"counts": {}, "by_type": {}}
    for prefix in ("train", "val"):
        kinds = pairs[f"{prefix}_type"]
        analysis["counts"][prefix] = int(len(kinds))
        for code, name in PAIR_TYPES.items():
            mask = kinds == code
            if not mask.any():
                continue
            z = pairs[f"{prefix}_z_distance"][mask]
            u = pairs[f"{prefix}_u_distance"][mask]
            z_lo, z_hi = np.quantile(z, [0.05, 0.95])
            u_lo, u_hi = np.quantile(u, [0.05, 0.95])
            ambiguous = (z <= z_lo) & (u >= u_hi)
            analysis["by_type"].setdefault(prefix, {})[name] = {
                "count": int(mask.sum()),
                "u_distance_mean": float(u.mean()),
                "u_distance_median": float(np.median(u)),
                "z_distance_mean": float(z.mean()),
                "z_distance_median": float(np.median(z)),
                "ambiguous_fraction": float(ambiguous.mean()),
                "ambiguous_count": int(ambiguous.sum()),
                "corr_u_z": float(np.corrcoef(u, z)[0, 1]) if len(u) > 2 else float("nan"),
            }
    save_json(OUT / "pair_distance_analysis.json", analysis)
    return pairs


def fit_W_for_config(latents, pairs, jacobian, controller, K, r, lam):
    types = CONTROLLER_SETS[controller]
    mask = np.isin(pairs["train_type"], types)
    src = pairs["train_src"][mask]
    dst = pairs["train_dst"][mask]
    dz = (latents["z_train"][dst, :K] - latents["z_train"][src, :K]).astype(np.float64)
    du = (latents["u_train"][dst] - latents["u_train"][src]).astype(np.float64)
    Vh = jacobian["archive"][f"Vh_{K}"]
    V_r = Vh[: min(r, 41)].T
    Z = torch.as_tensor(dz, dtype=torch.float64)
    C = torch.as_tensor(du @ V_r, dtype=torch.float64)
    gram = Z.T @ Z + float(lam) * torch.eye(K, dtype=torch.float64)
    rhs = Z.T @ C
    Wt = torch.linalg.solve(gram, rhs)
    return Wt.T.numpy(), V_r


def stage_controllers(
    latents,
    pairs,
    arrays,
    jacobian,
    simulator,
    rebuild: bool,
    val_eval_states: int,
):
    table_path = OUT / "linear_controller_results.json"
    if table_path.exists() and not rebuild:
        controller_table = json.loads(table_path.read_text())
        log(f"loaded controller table from {table_path}")
    else:
        controller_table, _ = fit_controller_table(latents, pairs, jacobian)
        save_json(table_path, controller_table)
        log(f"controller table fitted -> {table_path}")

    k_scores = {}
    for K in K_CANDIDATES:
        values = [
            controller_table["C3_mixed"][str(K)][str(r)]["holdout_c_normalized_rmse"]
            for r in R_CANDIDATES
            if str(r) in controller_table["C3_mixed"][str(K)]
        ]
        k_scores[str(K)] = float(np.mean(values)) if values else float("inf")
    best_K = int(min(K_CANDIDATES, key=lambda k: k_scores[str(k)]))
    log(f"controller K scores: {k_scores}; selected K={best_K}")

    m_val = arrays["validation"][1][:val_eval_states]
    m_anchor = np.load(ROOT / "artifacts/common/anchor.npz")["m_star"]
    oracle = m_anchor[None, :] - m_val
    zero = np.zeros_like(oracle)
    oracle_summary, _ = evaluate_delta(simulator, m_val, oracle)
    zero_summary, _ = evaluate_delta(simulator, m_val, zero)

    grid = {}
    for controller in CONTROLLER_SETS:
        grid[controller] = {}
        for r in R_CANDIDATES:
            entry = controller_table[controller][str(best_K)][str(r)]
            lam = entry["lambda"]
            W, V_r = fit_W_for_config(
                latents, pairs, jacobian, controller, best_K, r, lam
            )
            dm = build_correction(
                latents, W, V_r, best_K, "validation", val_eval_states
            )
            summary, _ = evaluate_delta(simulator, m_val, dm)
            A = jacobian["archive"][f"A_{best_K}"]
            anchor_z = np.asarray(latents["z_anchor"]).reshape(-1)[:best_K]
            dz_anchor = (
                anchor_z[None, :]
                - latents["z_validation"][:val_eval_states, :best_K]
            ).astype(np.float64)
            dz_hat = (dm / latents["half"][None, :]) @ A.T
            summary.update(
                {
                    "controller": controller,
                    "r": int(r),
                    "K": int(best_K),
                    "lambda": float(lam),
                    "anchor_dz_normalized_rmse": normalized_rmse(dz_hat, dz_anchor),
                    "anchor_dz_r2": float(
                        1.0
                        - np.mean((dz_hat - dz_anchor) ** 2)
                        / max(np.mean(dz_anchor**2), 1e-30)
                    ),
                    "pair_holdout_c_rmse": entry["holdout_c_normalized_rmse"],
                    "pair_val_c_rmse": entry["val_c_normalized_rmse"],
                }
            )
            grid[controller][str(r)] = summary
            log(
                f"{controller} K={best_K} r={r}: pass={summary['pass_rate']:.3f} "
                f"mean_miss={summary['mean_max_miss_db']:.2f} "
                f"dz_rmse={summary['anchor_dz_normalized_rmse']:.4f}"
            )

    baseline = {"oracle_anchor_correction": oracle_summary, "zero_correction": zero_summary}

    candidates = [
        (summary["pass_rate"], -summary["mean_max_miss_db"], controller, r)
        for controller, entries in grid.items()
        for r, summary in entries.items()
    ]
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, best_controller, best_r = candidates[0]
    best_entry = grid[best_controller][best_r]
    lam = best_entry["lambda"]
    W, V_r = fit_W_for_config(
        latents, pairs, jacobian, best_controller, best_K, int(best_r), lam
    )

    full_validation = {}
    dm_full = build_correction(latents, W, V_r, best_K, "validation", None)
    full_validation["best"] = evaluate_delta(simulator, arrays["validation"][1], dm_full)[0]
    full_validation["best"].update(
        {"controller": best_controller, "r": int(best_r), "K": best_K}
    )
    for controller in CONTROLLER_SETS:
        entries = grid[controller]
        best_r_c = min(
            entries,
            key=lambda key: (-entries[key]["pass_rate"], entries[key]["mean_max_miss_db"]),
        )
        entry = entries[best_r_c]
        Wc, Vc = fit_W_for_config(
            latents, pairs, jacobian, controller, best_K, int(best_r_c), entry["lambda"]
        )
        dm_c = build_correction(latents, Wc, Vc, best_K, "validation", None)
        summary_c, _ = evaluate_delta(simulator, arrays["validation"][1], dm_c)
        summary_c.update({"controller": controller, "r": int(best_r_c), "K": best_K})
        full_validation[controller] = summary_c

    dm_internal = build_correction(latents, W, V_r, best_K, "internal_test", None)
    internal_summary, _ = evaluate_delta(
        simulator, arrays["internal_test"][1], dm_internal
    )
    internal_summary.update({"controller": best_controller, "r": int(best_r), "K": best_K})

    result = {
        "selected": {
            "controller": best_controller,
            "r": int(best_r),
            "K": int(best_K),
            "lambda": float(lam),
            "pass_rate": float(best_entry["pass_rate"]),
            "mean_max_miss_db": float(best_entry["mean_max_miss_db"]),
        },
        "K_scores_for_controller": k_scores,
        "grid_validation_subset": grid,
        "baselines": baseline,
        "full_validation": full_validation,
        "internal_test": internal_summary,
    }
    save_json(OUT / "validation_results.json", result)
    _plot_controllers(grid, jacobian)
    return result



def _plot_controllers(grid: dict, jacobian: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    r_values = sorted(R_CANDIDATES)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for controller, entries in grid.items():
        misses = [entries[str(r)]["mean_max_miss_db"] for r in r_values]
        ax.plot(r_values, misses, marker="o", label=controller)
    ax.set(xlabel="observable dimension r", ylabel="mean max miss (dB)", title="Anchor correction vs r")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "mean_miss_vs_r.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for controller, entries in grid.items():
        rates = [entries[str(r)]["pass_rate"] for r in r_values]
        ax.plot(r_values, rates, marker="o", label=controller)
    ax.set(xlabel="observable dimension r", ylabel="one-shot pass rate", title="Pass rate vs r")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "pass_rate_vs_r.png", dpi=130)
    plt.close(fig)

    best_K = jacobian["best_K"]
    singular = np.asarray(jacobian["details"][str(best_K)]["singular_values"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.semilogy(np.arange(1, len(singular) + 1), singular, marker="o", markersize=3)
    ax.set(xlabel="index", ylabel="singular value (log)", title=f"Observable spectrum (K={best_K})")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(plot_dir / "singular_spectrum.png", dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("pca", "pairs", "jacobian", "residual", "controllers", "all"),
        default="all",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k-max", type=int, default=512)
    parser.add_argument("--val-eval-states", type=int, default=300)
    args = parser.parse_args()

    simulator = load_simulator()
    device = str(get_device(args.device))
    arrays = load_arrays("cache/dataset")

    if args.stage in ("pca", "all"):
        latents = stage_pca(arrays, simulator, device, args.k_max, args.rebuild)
    else:
        latents = load_npz(OUT / "latents.npz")
        latents.update(load_npz(OUT / "pca_latent.npz"))
        lower = np.asarray(simulator.lower_bounds)
        upper = np.asarray(simulator.upper_bounds)
        _, center, half = normalized_state(arrays["train"][1], lower, upper)
        latents["center"] = center
        latents["half"] = half
    if args.stage == "pca":
        spectrum = json.loads((OUT / "pca_spectrum.json").read_text())
        print(json.dumps({"cumulative": spectrum["cumulative_at"]}, indent=2))
        return 0

    if args.stage in ("pairs", "jacobian", "residual", "controllers", "all"):
        pairs = stage_pairs(latents, args.rebuild) if args.stage in (
            "pairs",
            "all",
        ) else load_npz(OUT / "pair_indices.npz")
    if args.stage == "pairs":
        print(json.dumps(json.loads((OUT / "pair_distance_analysis.json").read_text())["counts"], indent=2))
        return 0

    jacobian = stage_jacobian(latents, pairs, args.rebuild)
    if args.stage == "jacobian":
        print(json.dumps(jacobian["details"], indent=2)[:4000])
        return 0

    if args.stage in ("residual", "all"):
        stage_residual(latents, jacobian, args.rebuild)
    if args.stage == "residual":
        print(json.dumps(json.loads((OUT / "residual_subspace_analysis.json").read_text()), indent=2)[:4000])
        return 0

    result = stage_controllers(
        latents,
        pairs,
        arrays,
        jacobian,
        simulator,
        args.rebuild,
        args.val_eval_states,
    )
    print(json.dumps(result["selected"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

