"""Stage 6: existing-8000 system-ID feature study and ambiguity tests.

No new states are sampled.  Uses cached train 8000 / validation 1000 and the
already extracted order-64 Chebyshev rational coefficients.

Subcommands:
    features   train-only group whitening / raw PCA; saves stage6_features.npz
    models     global ridge, local weighted ridge, small ResMLP, learning curve
    ambiguity  near-collision, cross-correction, conditional parameter spread
    oneshot    validation one-shot for the serious models
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

import numpy as np
import torch
from torch import nn

from solution.data import CM19ResponseDataset
from solution.pairwise import evaluate_delta, normalized_state, pca_response, project_latent
from solution.simulator import load_simulator
from solution.utils import get_device, save_json

OUT = ROOT / "artifacts/stage6"
OUT.mkdir(parents=True, exist_ok=True)
WEAK = [21, 23, 26, 31]
IMPROVED = [1, 37, 38, 39, 40]


def log(message: str) -> None:
    print(f"[stage6] {message}", flush=True)


def load_split(split: str, n: int):
    dataset = CM19ResponseDataset("cache/dataset", split=split, limit=n)
    rows = np.asarray(dataset._indices)
    responses = np.array(dataset.responses[rows]).astype(np.float32)
    m_current = np.array(dataset.m_current[rows]).astype(np.float64)
    seeds = np.asarray(dataset.seeds_slice())
    return responses, m_current, seeds


def grouped_indices(order: int = 64):
    """Coefficient layout: [real d,a,b | imag d,a,b] with d=order, a=b=order+1."""
    size = order + (order + 1) * 2
    d = np.arange(0, order)
    a = np.arange(order, order + order + 1)
    b = np.arange(order + order + 1, size)
    return {
        "d": np.concatenate([d, d + size]),
        "a": np.concatenate([a, a + size]),
        "b": np.concatenate([b, b + size]),
    }


def allocate(total: int, sizes: list[int]) -> list[int]:
    total = int(total)
    if total >= sum(sizes):
        return list(sizes)
    raw = np.asarray(sizes, dtype=np.float64) / sum(sizes) * total
    alloc = np.floor(raw).astype(int)
    alloc = np.minimum(alloc, np.asarray(sizes))
    while alloc.sum() < total:
        candidates = [i for i in range(len(sizes)) if alloc[i] < sizes[i]]
        if not candidates:
            break
        alloc[max(candidates, key=lambda i: raw[i] - alloc[i])] += 1
    return alloc.tolist()



def group_whiten(
    train: np.ndarray,
    validation: np.ndarray,
    groups: dict[str, np.ndarray],
    total_dim: int,
    seed: int = 2026,
) -> dict:
    names = list(groups)
    sizes = [len(groups[name]) for name in names]
    alloc = allocate(total_dim, sizes)
    train_parts, val_parts, spectra = [], [], {}
    torch.manual_seed(int(seed))
    for name, dim in zip(names, alloc):
        idx = groups[name]
        X = torch.as_tensor(train[:, idx], dtype=torch.float32)
        mean = X.mean(dim=0)
        centered = X - mean
        q = int(min(dim, min(centered.shape) - 1))
        _, singular, basis = torch.pca_lowrank(centered, q=q, center=False, niter=6)
        pivot = basis.abs().argmax(dim=0)
        signs = torch.sign(basis[pivot, torch.arange(q)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        basis = basis * signs.unsqueeze(0)
        scores = centered @ basis
        scale = scores.std(dim=0, unbiased=False).clamp_min(1e-6)
        train_parts.append((scores / scale).numpy().astype(np.float32))
        Xv = torch.as_tensor(validation[:, idx], dtype=torch.float32) - mean
        val_parts.append(((Xv @ basis) / scale).numpy().astype(np.float32))
        spectra[name] = {
            "components": int(q),
            "singular_values": singular[:q].numpy().astype(float).tolist(),
            "variance_ratio": (
                singular[:q] ** 2 / (singular[:q] ** 2).sum()
            ).numpy().astype(float).tolist(),
        }
    return {
        "train": np.concatenate(train_parts, axis=1),
        "validation": np.concatenate(val_parts, axis=1),
        "allocation": dict(zip(names, alloc)),
        "spectra": spectra,
    }


def fit_ridge(z: np.ndarray, y: np.ndarray, lam: float):
    zm = z.mean(axis=0)
    ym = y.mean(axis=0)
    zc = z - zm
    yc = y - ym
    gram = zc.T @ zc + float(lam) * np.eye(z.shape[1])
    weight = np.linalg.solve(gram, zc.T @ yc)
    return weight, ym - zm @ weight


def predict_ridge(z: np.ndarray, weight: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return z @ weight + intercept


def select_ridge(z: np.ndarray, y: np.ndarray, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    n = len(z)
    split = int(0.8 * n)
    best = None
    for lam in lams:
        w, b = fit_ridge(z[:split], y[:split], lam)
        mae = float(np.mean(np.abs(predict_ridge(z[split:], w, b) - y[split:])))
        row = {"lambda": float(lam), "holdout_mae": mae}
        if best is None or mae < best["holdout_mae"]:
            best = row
    w, b = fit_ridge(z, y, best["lambda"])
    return w, b, best


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target
    abs_err = np.abs(err)
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "mae": float(abs_err.mean()),
        "median_abs": float(np.median(abs_err)),
        "p95_abs": float(np.quantile(abs_err, 0.95)),
        "mean_r2": float(np.mean(r2)),
        "median_r2": float(np.median(r2)),
        "count_r2_below_0_1": int((r2 < 0.1).sum()),
        "per_parameter_r2": r2.astype(float).tolist(),
        "weak_r2": {str(p): float(r2[p]) for p in WEAK},
        "improved_r2": {str(p): float(r2[p]) for p in IMPROVED},
    }



def stage_features():
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    rational_train = np.load(ROOT / "artifacts/system_id/rational_train.npz", allow_pickle=True)
    rational_val = np.load(ROOT / "artifacts/system_id/rational_validation.npz", allow_pickle=True)
    r1_train = rational_train["features_r1_64"].astype(np.float32)
    r1_val = rational_val["features_r1_64"].astype(np.float32)
    responses_train, m_train, seeds_train = load_split("train", 8000)
    responses_val, m_val, seeds_val = load_split("validation", 1000)

    sim = load_simulator()
    lower = np.asarray(sim.lower_bounds, dtype=np.float64)
    upper = np.asarray(sim.upper_bounds, dtype=np.float64)
    u_train, center, half = normalized_state(m_train, lower, upper)
    u_val, _, _ = normalized_state(m_val, lower, upper)
    anchor = np.load(ROOT / "artifacts/common/anchor.npz", allow_pickle=True)
    m_star = np.asarray(anchor["m_star"], dtype=np.float64)
    u_star, _, _ = normalized_state(m_star[None, :], lower, upper)
    u_star = u_star[0]

    groups = grouped_indices(64)
    payload = {
        "u_train": u_train.astype(np.float32),
        "u_val": u_val.astype(np.float32),
        "m_train": m_train.astype(np.float64),
        "m_val": m_val.astype(np.float64),
        "seeds_train": seeds_train,
        "seeds_val": seeds_val,
        "u_star": u_star.astype(np.float32),
        "center": center.astype(np.float64),
        "half": half.astype(np.float64),
    }
    meta = {
        "train_states": int(len(r1_train)),
        "validation_states": int(len(r1_val)),
        "rational_holdout_rel_l2_median": float(np.median(rational_train["holdout_rel_l2_64"])),
        "groups": {key: int(len(val)) for key, val in groups.items()},
        "whitening": {},
        "raw_pca": {},
    }
    for dim in (64, 128, 256, 388):
        result = group_whiten(r1_train, r1_val, groups, dim)
        payload[f"z_{dim}_train"] = result["train"]
        payload[f"z_{dim}_val"] = result["validation"]
        meta["whitening"][str(dim)] = {
            "allocation": result["allocation"],
            "spectra": result["spectra"],
            "train_dim": int(result["train"].shape[1]),
        }
        log(f"whitened dim {dim}: allocation {result['allocation']}")

    pca = pca_response(responses_train, k_max=64, device=device)
    raw_train = project_latent(responses_train, pca["mean"], pca["basis"])
    raw_val = project_latent(responses_val, pca["mean"], pca["basis"])
    payload["raw64_train"] = raw_train.astype(np.float32)
    payload["raw64_val"] = raw_val.astype(np.float32)
    meta["raw_pca"] = {
        "components": int(raw_train.shape[1]),
        "explained_ratio": pca["explained_ratio"].astype(float).tolist(),
        "cumulative_ratio": pca["cumulative_ratio"].astype(float).tolist(),
    }
    np.savez_compressed(OUT / "stage6_features.npz", **payload)
    save_json(OUT / "stage6_features.json", meta)
    log(f"saved {OUT / 'stage6_features.npz'}")
    return meta



def local_weighted_ridge(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_query: np.ndarray,
    k: int,
    lam: float = 1e-2,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    device = torch.device(device)
    zt = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    zq = torch.as_tensor(z_query, dtype=torch.float32, device=device)
    predictions = []
    for begin in range(0, len(zq), 128):
        q = zq[begin:begin + 128]
        dist = torch.cdist(q, zt)
        vals, idx = torch.topk(dist, int(k), largest=False, dim=1)
        sigma = vals[:, -1:].clamp_min(1e-6)
        weights = torch.exp(-0.5 * (vals / sigma) ** 2) + 1e-6
        out = torch.empty((len(q), yt.shape[1]), dtype=torch.float32, device=device)
        for row in range(len(q)):
            nb = idx[row]
            X = zt[nb]
            Y = yt[nb]
            w = weights[row][:, None]
            wsum = w.sum().clamp_min(1e-12)
            xbar = (w * X).sum(dim=0) / wsum
            ybar = (w * Y).sum(dim=0) / wsum
            Xc = X - xbar
            Yc = Y - ybar
            gram = Xc.T @ (w * Xc) + float(lam) * torch.eye(X.shape[1], device=device)
            rhs = Xc.T @ (w * Yc)
            coef = torch.linalg.solve(gram, rhs)
            out[row] = ybar + (q[row] - xbar) @ coef
        predictions.append(out.cpu())
    return torch.cat(predictions, dim=0).numpy().astype(np.float64)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.norm1(x))
        h = self.fc1(h)
        h = torch.nn.functional.gelu(self.norm2(h))
        h = self.fc2(h)
        return x + h


class SmallResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 384, out_dim: int = 41):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(ResidualBlock(hidden), ResidualBlock(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def train_resmlp(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    n_train: int,
    device: str | torch.device,
    epochs: int = 600,
    patience: int = 120,
    seed: int = 2026,
) -> tuple[SmallResMLP, dict]:
    torch.manual_seed(int(seed))
    device = torch.device(device)
    z = torch.as_tensor(z_train[:n_train], dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train[:n_train], dtype=torch.float32, device=device)
    split = int(0.8 * n_train)
    model = SmallResMLP(z.shape[1], hidden=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = {"mae": float("inf"), "state": None, "epoch": 0}
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(split, device=device)
        total = 0.0
        for begin in range(0, split, 256):
            idx = perm[begin:begin + 256]
            opt.zero_grad(set_to_none=True)
            pred = model(z[idx])
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            hold_pred = model(z[split:])
            hold_mae = float(torch.mean(torch.abs(hold_pred - y[split:])))
            zt_val = torch.as_tensor(z_val, dtype=torch.float32, device=device)
            yt_val = torch.as_tensor(y_val, dtype=torch.float32, device=device)
            val_mae = float(torch.mean(torch.abs(model(zt_val) - yt_val)))
        history.append({"epoch": epoch, "train_loss": total / max(split, 1), "holdout_mae": hold_mae, "val_mae": val_mae})
        if hold_mae < best["mae"] - 1e-6:
            best = {"mae": hold_mae, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
        elif epoch - best["epoch"] > int(patience):
            break
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        zt_val = torch.as_tensor(z_val, dtype=torch.float32, device=device)
        val_pred = model(zt_val).cpu().numpy().astype(np.float64)
    return model, {"best_epoch": best["epoch"], "holdout_mae": best["mae"], "history": history, "val_pred": val_pred}



def stage_models():
    data = np.load(OUT / "stage6_features.npz", allow_pickle=True)
    u_train = data["u_train"].astype(np.float64)
    u_val = data["u_val"].astype(np.float64)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {"global_ridge": {}, "local_ridge": {}, "resmlp": {}, "learning_curve": {}}

    feature_sets = {
        "rational64": (data["z_64_train"], data["z_64_val"]),
        "rational128": (data["z_128_train"], data["z_128_val"]),
        "rational256": (data["z_256_train"], data["z_256_val"]),
        "rational_full": (data["z_388_train"], data["z_388_val"]),
        "rational256_plus_raw64": (
            np.concatenate([data["z_256_train"], data["raw64_train"]], axis=1),
            np.concatenate([data["z_256_val"], data["raw64_val"]], axis=1),
        ),
        "raw_pca64": (data["raw64_train"], data["raw64_val"]),
    }
    predictions = {}
    for name, (ztr, zva) in feature_sets.items():
        w, b, selection = select_ridge(np.asarray(ztr, np.float64), u_train)
        pred = predict_ridge(np.asarray(zva, np.float64), w, b)
        metrics = regression_metrics(pred, u_val)
        summary["global_ridge"][name] = {**metrics, "selection": selection, "feature_dim": int(ztr.shape[1])}
        predictions[f"global_{name}"] = pred.astype(np.float32)
        log(f"global ridge {name}: MAE {metrics['mae']:.5f}, mean R2 {metrics['mean_r2']:.3f}")

    ztr = np.asarray(data["z_256_train"], np.float64)
    zva = np.asarray(data["z_256_val"], np.float64)
    local_predictions = {}
    local_best = None
    for k in (32, 64, 128, 256):
        for lam in (1e-2, 10.0):
            start = time.perf_counter()
            pred = local_weighted_ridge(ztr, u_train, zva, k, lam=lam, device=device)
            metrics = regression_metrics(pred, u_val)
            key = f"{k}_lam{lam:g}"
            summary["local_ridge"][key] = {
                **metrics, "k": k, "lambda": lam, "seconds": time.perf_counter() - start
            }
            if local_best is None or metrics["mae"] < summary["local_ridge"][local_best]["mae"]:
                local_best = key
                local_predictions["best"] = pred
            log(f"local ridge k={k} lambda={lam:g}: MAE {metrics['mae']:.5f}, mean R2 {metrics['mean_r2']:.3f}")
    predictions["local_best"] = local_predictions["best"].astype(np.float32)
    summary["local_best_key"] = local_best
    summary["local_best_k"] = int(summary["local_ridge"][local_best]["k"])

    ztr_best = np.asarray(feature_sets["rational256_plus_raw64"][0], np.float64)
    zva_best = np.asarray(feature_sets["rational256_plus_raw64"][1], np.float64)
    model, info = train_resmlp(ztr_best, u_train, zva_best, u_val, 8000, device)
    metrics = regression_metrics(info["val_pred"], u_val)
    summary["resmlp"]["8000"] = {
        **metrics,
        "best_epoch": info["best_epoch"],
        "holdout_mae": info["holdout_mae"],
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "history": info["history"],
    }
    predictions["resmlp_8000"] = info["val_pred"].astype(np.float32)
    torch.save(model.state_dict(), OUT / "stage6_resmlp.pt")
    log(f"ResMLP 8000: MAE {metrics['mae']:.5f}, mean R2 {metrics['mean_r2']:.3f}, params {summary['resmlp']['8000']['parameters']}")

    for n in (1000, 2000, 4000):
        w, b, selection = select_ridge(ztr_best[:n], u_train[:n])
        pred = predict_ridge(zva_best, w, b)
        ridge_metrics = regression_metrics(pred, u_val)
        model_n, info_n = train_resmlp(ztr_best, u_train, zva_best, u_val, n, device)
        mlp_metrics = regression_metrics(info_n["val_pred"], u_val)
        summary["learning_curve"][str(n)] = {
            "ridge": {**ridge_metrics, "selection": selection},
            "resmlp": {**mlp_metrics, "best_epoch": info_n["best_epoch"], "holdout_mae": info_n["holdout_mae"]},
        }
        log(f"curve N={n}: ridge MAE {ridge_metrics['mae']:.5f}, mlp MAE {mlp_metrics['mae']:.5f}")
    summary["learning_curve"]["8000"] = {
        "ridge": summary["global_ridge"]["rational256_plus_raw64"],
        "resmlp": {k: v for k, v in summary["resmlp"]["8000"].items() if k != "history"},
    }

    np.savez_compressed(
        OUT / "stage6_predictions.npz",
        **predictions,
        u_val=u_val.astype(np.float32),
    )
    save_json(OUT / "stage6_models.json", summary)
    log(f"saved {OUT / 'stage6_models.json'}")
    return summary



def _pair_stats(ds: np.ndarray, du: np.ndarray, indices: np.ndarray) -> dict:
    return {
        "pairs": int(len(indices)),
        "d_s_median": float(np.median(ds[indices])),
        "d_s_max": float(np.max(ds[indices])),
        "d_u_median": float(np.median(du[indices])),
        "d_u_p90": float(np.quantile(du[indices], 0.90)),
        "d_u_max": float(np.max(du[indices])),
    }


def stage_ambiguity():
    data = np.load(OUT / "stage6_features.npz", allow_pickle=True)
    z_train = np.asarray(data["z_256_train"], np.float32)
    z_val = np.asarray(data["z_256_val"], np.float32)
    u_train = data["u_train"].astype(np.float64)
    u_val = data["u_val"].astype(np.float64)
    m_train = data["m_train"].astype(np.float64)
    seeds_train = data["seeds_train"]
    responses_train, _, _ = load_split("train", 8000)
    responses_val, m_val, seeds_val = load_split("validation", 1000)
    sim = load_simulator()
    anchor = np.load(ROOT / "artifacts/common/anchor.npz", allow_pickle=True)
    m_star = np.asarray(anchor["m_star"], dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zt = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    nearest = np.empty(len(z_train), dtype=np.int64)
    nearest_feature_distance = np.empty(len(z_train), dtype=np.float32)
    for begin in range(0, len(z_train), 1000):
        end = min(len(z_train), begin + 1000)
        dist = torch.cdist(zt[begin:end], zt)
        rows = torch.arange(end - begin, device=device)
        dist[rows, torch.arange(begin, end, device=device)] = float("inf")
        values, indices = torch.topk(dist, 1, largest=False, dim=1)
        nearest[begin:end] = indices[:, 0].cpu().numpy()
        nearest_feature_distance[begin:end] = values[:, 0].cpu().numpy()

    flat = responses_train.reshape(len(responses_train), -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1)
    dots = np.sum(flat * flat[nearest], axis=1)
    diff = np.sqrt(np.maximum(norms ** 2 + norms[nearest] ** 2 - 2.0 * dots, 0.0))
    d_s = diff / (0.5 * (norms + norms[nearest]) + 1e-30)
    d_u = np.linalg.norm(u_train - u_train[nearest], axis=1)

    order = np.argsort(d_s)
    bins = {}
    for label, frac in (("nearest_1pct", 0.01), ("nearest_0.1pct", 0.001), ("nearest_0.01pct", 0.0001)):
        count = max(1, int(frac * len(order)))
        bins[label] = _pair_stats(d_s, d_u, order[:count])
    for threshold in (1e-2, 1e-3, 1e-4, 1e-5):
        idx = np.where(d_s < threshold)[0]
        bins[f"d_s_below_{threshold:g}"] = _pair_stats(d_s, d_u, idx) if len(idx) else {"pairs": 0}

    extreme = []
    for idx in order[:30]:
        j = int(nearest[idx])
        extreme.append(
            {
                "i": int(idx),
                "j": j,
                "seed_i": int(seeds_train[idx]),
                "seed_j": int(seeds_train[j]),
                "d_s": float(d_s[idx]),
                "d_u": float(d_u[idx]),
                "feature_distance": float(nearest_feature_distance[idx]),
            }
        )

    cross_rows = []
    for idx in order[:30]:
        i = int(idx)
        j = int(nearest[idx])
        delta_i = m_star - m_train[i]
        delta_j = m_star - m_train[j]
        cases = {
            "i_plus_delta_i": (m_train[i], delta_i),
            "j_plus_delta_j": (m_train[j], delta_j),
            "j_plus_delta_i": (m_train[j], delta_i),
            "i_plus_delta_j": (m_train[i], delta_j),
        }
        row = {"i": i, "j": j, "seed_i": int(seeds_train[i]), "seed_j": int(seeds_train[j]),
               "d_s": float(d_s[i]), "d_u": float(d_u[i])}
        for name, (current, delta) in cases.items():
            result = sim.apply_correction(current, delta)
            row[name] = {
                "passed": bool(result.all_specs_pass),
                "max_miss_db": float(result.max_spec_miss_db),
            }
        cross_rows.append(row)
    # exact raw complex-response nearest neighbors (independent of feature space)
    X_raw = torch.as_tensor(flat, dtype=torch.float32, device=device)
    raw_norms = torch.linalg.vector_norm(X_raw, dim=1)
    nearest_raw = np.empty(len(X_raw), dtype=np.int64)
    raw_min_distance = np.empty(len(X_raw), dtype=np.float32)
    for begin in range(0, len(X_raw), 500):
        end = min(len(X_raw), begin + 500)
        dist = torch.cdist(X_raw[begin:end], X_raw)
        rows = torch.arange(end - begin, device=device)
        dist[rows, torch.arange(begin, end, device=device)] = float("inf")
        values, indices = torch.topk(dist, 1, largest=False, dim=1)
        nearest_raw[begin:end] = indices[:, 0].cpu().numpy()
        raw_min_distance[begin:end] = values[:, 0].cpu().numpy()
    raw_nn = torch.as_tensor(nearest_raw, dtype=torch.long, device=device)
    d_s_raw = (
        torch.as_tensor(raw_min_distance, device=device)
        / (0.5 * (raw_norms + raw_norms[raw_nn]) + 1e-30)
    ).cpu().numpy()
    d_u_raw = np.linalg.norm(u_train - u_train[nearest_raw], axis=1)
    order_raw = np.argsort(d_s_raw)
    bins_raw = {}
    for label, frac in (("nearest_1pct", 0.01), ("nearest_0.1pct", 0.001), ("nearest_0.01pct", 0.0001)):
        count = max(1, int(frac * len(order_raw)))
        bins_raw[label] = _pair_stats(d_s_raw, d_u_raw, order_raw[:count])
    for threshold in (1e-2, 1e-3, 1e-4, 1e-5):
        idx = np.where(d_s_raw < threshold)[0]
        bins_raw[f"d_s_below_{threshold:g}"] = _pair_stats(d_s_raw, d_u_raw, idx) if len(idx) else {"pairs": 0}
    extreme_raw = []
    for idx in order_raw[:30]:
        j = int(nearest_raw[idx])
        extreme_raw.append(
            {
                "i": int(idx),
                "j": j,
                "seed_i": int(seeds_train[idx]),
                "seed_j": int(seeds_train[j]),
                "d_s": float(d_s_raw[idx]),
                "d_u": float(d_u_raw[idx]),
            }
        )
    cross_raw = []
    for idx in order_raw[:30]:
        i = int(idx)
        j = int(nearest_raw[idx])
        delta_i = m_star - m_train[i]
        delta_j = m_star - m_train[j]
        cases = {
            "i_plus_delta_i": (m_train[i], delta_i),
            "j_plus_delta_j": (m_train[j], delta_j),
            "j_plus_delta_i": (m_train[j], delta_i),
            "i_plus_delta_j": (m_train[i], delta_j),
        }
        row = {"i": i, "j": j, "seed_i": int(seeds_train[i]), "seed_j": int(seeds_train[j]),
               "d_s": float(d_s_raw[i]), "d_u": float(d_u_raw[i])}
        for name, (current, delta) in cases.items():
            result = sim.apply_correction(current, delta)
            row[name] = {
                "passed": bool(result.all_specs_pass),
                "max_miss_db": float(result.max_spec_miss_db),
            }
        cross_raw.append(row)




    zvt = torch.as_tensor(z_val, dtype=torch.float32, device=device)
    u_t = torch.as_tensor(u_train, dtype=torch.float32, device=device)
    spread_sum = np.zeros(41, dtype=np.float64)
    spread_count = 0
    k_spread = 20
    for begin in range(0, len(z_val), 128):
        q = zvt[begin:begin + 128]
        dist = torch.cdist(q, zt)
        _, idx = torch.topk(dist, k_spread, largest=False, dim=1)
        neighbors = u_t[idx]
        spread_sum += neighbors.std(dim=1, unbiased=False).sum(dim=0).cpu().numpy()
        spread_count += len(q)
    conditional_spread = spread_sum / max(spread_count, 1)
    overall_spread = u_train.std(axis=0)
    spread_payload = {
        "k": k_spread,
        "conditional_std": conditional_spread.tolist(),
        "overall_std": overall_spread.tolist(),
        "conditional_over_overall": (conditional_spread / np.maximum(overall_spread, 1e-12)).tolist(),
        "weak_conditional": {str(p): float(conditional_spread[p]) for p in WEAK},
        "weak_overall": {str(p): float(overall_spread[p]) for p in WEAK},
        "improved_conditional": {str(p): float(conditional_spread[p]) for p in IMPROVED},
        "improved_overall": {str(p): float(overall_spread[p]) for p in IMPROVED},
    }

    raw_conditional_sum = np.zeros(41, dtype=np.float64)
    raw_conditional_count = 0
    X_val_raw = torch.as_tensor(
        responses_val.reshape(len(responses_val), -1), dtype=torch.float32, device=device
    )
    for begin in range(0, len(X_val_raw), 64):
        q = X_val_raw[begin:begin + 64]
        dist = torch.cdist(q, X_raw)
        _, idx = torch.topk(dist, k_spread, largest=False, dim=1)
        neighbors = u_t[idx]
        raw_conditional_sum += neighbors.std(dim=1, unbiased=False).sum(dim=0).cpu().numpy()
        raw_conditional_count += len(q)
    raw_conditional_spread = raw_conditional_sum / max(raw_conditional_count, 1)
    raw_spread_payload = {
        "k": k_spread,
        "conditional_std": raw_conditional_spread.tolist(),
        "overall_std": overall_spread.tolist(),
        "conditional_over_overall": (raw_conditional_spread / np.maximum(overall_spread, 1e-12)).tolist(),
        "weak_conditional": {str(p): float(raw_conditional_spread[p]) for p in WEAK},
        "weak_overall": {str(p): float(overall_spread[p]) for p in WEAK},
        "improved_conditional": {str(p): float(raw_conditional_spread[p]) for p in IMPROVED},
        "improved_overall": {str(p): float(overall_spread[p]) for p in IMPROVED},
    }
    payload = {
        "nearest_feature": "whitened order64 rational coefficients (256d)",
        "nearest_feature_distance_median": float(np.median(nearest_feature_distance)),
        "d_s_median": float(np.median(d_s)),
        "d_u_median": float(np.median(d_u)),
        "bins": bins,
        "extreme_pairs": extreme,
        "cross_correction": cross_rows,
        "conditional_spread": spread_payload,
        "own_pass_rate": float(np.mean([row["i_plus_delta_i"]["passed"] for row in cross_rows])),
        "cross_i_pass_rate": float(np.mean([row["i_plus_delta_j"]["passed"] for row in cross_rows])),
        "cross_j_pass_rate": float(np.mean([row["j_plus_delta_i"]["passed"] for row in cross_rows])),
        "raw_response_nn": {
            "d_s_min": float(np.min(d_s_raw)),
            "d_s_median": float(np.median(d_s_raw)),
            "d_u_median": float(np.median(d_u_raw)),
            "bins": bins_raw,
            "extreme_pairs": extreme_raw,
            "cross_correction": cross_raw,
            "own_pass_rate": float(np.mean([row["i_plus_delta_i"]["passed"] for row in cross_raw])),
            "cross_i_pass_rate": float(np.mean([row["i_plus_delta_j"]["passed"] for row in cross_raw])),
            "cross_j_pass_rate": float(np.mean([row["j_plus_delta_i"]["passed"] for row in cross_raw])),
        },
        "raw_conditional_spread": raw_spread_payload,
    }
    save_json(OUT / "stage6_ambiguity.json", payload)
    log(f"saved {OUT / 'stage6_ambiguity.json'}")
    return payload


def stage_oneshot():
    data = np.load(OUT / "stage6_features.npz", allow_pickle=True)
    preds = np.load(OUT / "stage6_predictions.npz", allow_pickle=True)
    _, m_val, seeds_val = load_split("validation", 1000)
    anchor = np.load(ROOT / "artifacts/common/anchor.npz", allow_pickle=True)
    m_star = np.asarray(anchor["m_star"], dtype=np.float64)
    center = data["center"].astype(np.float64)
    half = data["half"].astype(np.float64)
    sim = load_simulator()
    summary = {}
    for key in ("global_rational256_plus_raw64", "local_best", "resmlp_8000"):
        pred_u = preds[key].astype(np.float64)
        pred_m = center[None, :] + half[None, :] * pred_u
        delta = m_star[None, :] - pred_m
        metrics, rows = evaluate_delta(sim, m_val, delta)
        summary[key] = {
            **{k: v for k, v in metrics.items() if k != "per_requirement"},
            "per_requirement": metrics["per_requirement"],
        }
        log(f"one-shot {key}: pass {metrics['pass_rate']:.3f}, mean miss {metrics['mean_max_miss_db']:.2f} dB")
    save_json(OUT / "stage6_oneshot.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("features", "models", "ambiguity", "oneshot", "all"))
    args = parser.parse_args()
    if args.stage in ("features", "all"):
        stage_features()
    if args.stage in ("models", "all"):
        stage_models()
    if args.stage in ("ambiguity", "all"):
        stage_ambiguity()
    if args.stage in ("oneshot", "all"):
        stage_oneshot()


if __name__ == "__main__":
    main()
