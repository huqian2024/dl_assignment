"""Stage 8: learned-forward surrogate and numerical inverse refinement.

Only cached train 8000 / validation 1000 are used.  No simulator states are
sampled; no anchor is re-optimized.
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
from solution.pairwise import pca_response, project_latent
from solution.simulator import load_simulator
from solution.system_id import spec_reconstruction
from solution.utils import get_device, save_json

OUT = ROOT / "artifacts/stage8"
OUT.mkdir(parents=True, exist_ok=True)
WEAK = [21, 23, 26, 31]
DANGEROUS = [5, 8, 10, 22, 23, 24, 31, 32, 35, 37]
CENTER = None
HALF = None


def log(message: str) -> None:
    print(f"[stage8] {message}", flush=True)


def load_split(split: str, n: int):
    dataset = CM19ResponseDataset("cache/dataset", split=split, limit=n)
    rows = np.asarray(dataset._indices)
    responses = np.array(dataset.responses[rows]).astype(np.float32)
    m_current = np.array(dataset.m_current[rows]).astype(np.float64)
    seeds = np.asarray(dataset.seeds_slice())
    return responses, m_current, seeds


def normalized(m: np.ndarray):
    sim = load_simulator()
    lower = np.asarray(sim.lower_bounds, dtype=np.float64)
    upper = np.asarray(sim.upper_bounds, dtype=np.float64)
    center = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-3)
    return (m - center) / half, center, half


def stage_pca():
    responses, _, _ = load_split("train", 8000)
    # Fit PCA only on the forward/initializer training block; tune 7000:8000 is held out.
    responses = responses[:7000]
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    pca = pca_response(responses, k_max=512, device=device)
    np.savez_compressed(
        OUT / "raw_pca.npz",
        mean=pca["mean"],
        basis=pca["basis"],
        singular_values=pca["singular_values"],
        explained_ratio=pca["explained_ratio"],
        cumulative_ratio=pca["cumulative_ratio"],
    )
    save_json(
        OUT / "raw_pca.json",
        {
            "components": int(pca["basis"].shape[1]),
            "explained_ratio": pca["explained_ratio"].astype(float).tolist(),
            "cumulative_ratio": pca["cumulative_ratio"].astype(float).tolist(),
            "wall_seconds": time.perf_counter() - start,
        },
    )
    log(f"raw PCA saved: {pca['basis'].shape}, {time.perf_counter() - start:.1f}s")
    return pca



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


class ForwardMLP(nn.Module):
    def __init__(self, out_dim: int, hidden: int = 512):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(41, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.Sequential(
            ResidualBlock(hidden), ResidualBlock(hidden), ResidualBlock(hidden)
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, u):
        return self.head(self.blocks(self.stem(u)))


def train_forward(model, x, y, x_tune, y_tune, epochs=300, patience=40, lr=1e-3, device="cuda"):
    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    xt = torch.as_tensor(x_tune, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tune, dtype=torch.float32, device=device)
    best = {"loss": float("inf"), "state": None, "epoch": 0}
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(len(x), device=device)
        total = 0.0
        for begin in range(0, len(x), 256):
            idx = perm[begin:begin + 256]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            tune_loss = float(torch.nn.functional.mse_loss(model(xt), yt))
        history.append({"epoch": epoch, "train_mse": total / len(x), "tune_mse": tune_loss})
        if tune_loss < best["loss"] - 1e-8:
            best = {"loss": tune_loss, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
        elif epoch - best["epoch"] > int(patience):
            break
    model.load_state_dict(best["state"])
    model.eval()
    return model, {"best_epoch": best["epoch"], "tune_mse": best["loss"], "history": history}


def latent_metrics(pred, target):
    residual = pred - target
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = (residual ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "latent_rmse": rmse,
        "mean_r2": float(np.mean(r2)),
        "median_r2": float(np.median(r2)),
        "p10_r2": float(np.quantile(r2, 0.10)),
        "p90_r2": float(np.quantile(r2, 0.90)),
    }


def linear_forward(x, y, x_tune, y_tune, lam=1e-6):
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc = x - xm
    coef = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ (y - ym))
    pred = (x_tune - xm) @ coef + ym
    return pred, {"lambda": lam}


def prepare_data():
    responses, m_train, seeds_train = load_split("train", 8000)
    responses_val, m_val, seeds_val = load_split("validation", 1000)
    u_train, center, half = normalized(m_train)
    u_val, _, _ = normalized(m_val)
    pca = np.load(OUT / "raw_pca.npz")
    mean, basis = pca["mean"], pca["basis"]
    flat_train = responses.reshape(len(responses), -1)
    flat_val = responses_val.reshape(len(responses_val), -1)
    scores_train = ((flat_train - mean[None, :]) @ basis).astype(np.float64)
    scores_val = ((flat_val - mean[None, :]) @ basis).astype(np.float64)
    base = np.load(ROOT / "artifacts/stage6/stage6_features.npz", allow_pickle=True)
    z2_train = np.concatenate([base["z_256_train"], base["raw64_train"]], axis=1).astype(np.float64)
    z2_val = np.concatenate([base["z_256_val"], base["raw64_val"]], axis=1).astype(np.float64)
    z2_mean = z2_train[:7000].mean(axis=0)
    z2_std = z2_train[:7000].std(axis=0)
    z2_std = np.where(z2_std < 1e-12, 1.0, z2_std)
    z2_train = (z2_train - z2_mean) / z2_std
    z2_val = (z2_val - z2_mean) / z2_std
    return {
        "responses": responses,
        "responses_val": responses_val,
        "m_train": m_train,
        "m_val": m_val,
        "seeds_train": seeds_train,
        "seeds_val": seeds_val,
        "u_train": u_train,
        "u_val": u_val,
        "center": center,
        "half": half,
        "mean": mean,
        "basis": basis,
        "scores_train": scores_train,
        "scores_val": scores_val,
        "z2_train": z2_train,
        "z2_val": z2_val,
    }


def raw_reconstruction_metrics(pred_scores, data, k, simulator):
    mean, basis = data["mean"], data["basis"]
    flat_hat = mean[None, :] + pred_scores @ basis[:, :k].T
    flat_true = data["responses"][7000:8000].reshape(1000, -1).astype(np.float64)
    diff = flat_hat - flat_true
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(flat_true) + 1e-30))
    true = flat_true.reshape(1000, 4, -1)
    hat = flat_hat.reshape(1000, 4, -1)
    true11 = true[:, 0] + 1j * true[:, 1]
    true21 = true[:, 2] + 1j * true[:, 3]
    hat11 = hat[:, 0] + 1j * hat[:, 1]
    hat21 = hat[:, 2] + 1j * hat[:, 3]
    db = []
    for t, h in ((true11, hat11), (true21, hat21)):
        e = np.abs(20 * np.log10(np.maximum(np.abs(h), 1e-12)) - 20 * np.log10(np.maximum(np.abs(t), 1e-12)))
        db.append(e.ravel())
    spec_errors = []
    for idx in range(1000):
        spec = spec_reconstruction(
            simulator, true11[idx], true21[idx], hat11[idx], hat21[idx],
            np.ones(true11.shape[1], dtype=bool),
        )
        spec_errors.append(spec["max_abs_value_error_db"])
    spec_errors = np.asarray(spec_errors)
    db = np.concatenate(db)
    return {
        "relative_l2": rel,
        "db_median": float(np.median(db)),
        "db_p90": float(np.quantile(db, 0.90)),
        "db_p99": float(np.quantile(db, 0.99)),
        "spec_error_median": float(np.median(spec_errors)),
        "spec_error_p90": float(np.quantile(spec_errors, 0.90)),
    }


def stage_forward():
    from solution.system_id import spec_reconstruction
    from scripts.stage6_study import predict_ridge, select_ridge

    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_data()
    u_train, u_val = data["u_train"], data["u_val"]
    u0_w, u0_b, u0_sel = select_ridge(data["z2_train"][:7000], u_train[:7000])
    u0_tune = predict_ridge(data["z2_train"][7000:8000], u0_w, u0_b)
    u0_val = predict_ridge(data["z2_val"], u0_w, u0_b)
    sim = load_simulator()
    payload = {
        "initializer": {
            "feature": "rational PCA256 + raw PCA64",
            "lambda": u0_sel,
            "tune_normalized_mae": float(np.abs(u0_tune - u_train[7000:8000]).mean()),
            "tune_physical_mae": float(np.abs((u0_tune - u_train[7000:8000]) * data["half"][None, :]).mean()),
        }
    }
    for k in (256, 512):
        std = data["scores_train"][:7000, :k].std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        ytr = data["scores_train"][:7000, :k] / std
        ytune = data["scores_train"][7000:8000, :k] / std
        lin, lin_info = linear_forward(u_train[:7000], ytr, u_train[7000:8000], ytune)
        lin_metrics = latent_metrics(lin, ytune)
        model, info = train_forward(
            ForwardMLP(k), u_train[:7000], ytr, u_train[7000:8000], ytune,
            epochs=300, patience=40, lr=1e-3, device=device,
        )
        with torch.no_grad():
            mlp = model(torch.as_tensor(u_train[7000:8000], dtype=torch.float32, device=device)).cpu().numpy()
        mlp_metrics = latent_metrics(mlp, ytune)
        torch.save(model.state_dict(), OUT / f"forward_z1_{k}.pt")
        payload[f"z1_{k}"] = {
            "linear": {**lin_metrics, **lin_info},
            "mlp": {**mlp_metrics, "best_epoch": info["best_epoch"], "tune_mse": info["tune_mse"]},
            "raw_reconstruction_linear": raw_reconstruction_metrics(lin * std, data, k, sim),
            "raw_reconstruction_mlp": raw_reconstruction_metrics(mlp * std, data, k, sim),
        }
        log(f"z1 {k}: linear R2 {lin_metrics['mean_r2']:.3f}, mlp R2 {mlp_metrics['mean_r2']:.3f}")

    ytr = data["z2_train"][:7000]
    ytune = data["z2_train"][7000:8000]
    lin, lin_info = linear_forward(u_train[:7000], ytr, u_train[7000:8000], ytune)
    lin_metrics = latent_metrics(lin, ytune)
    model, info = train_forward(
        ForwardMLP(ytr.shape[1]), u_train[:7000], ytr, u_train[7000:8000], ytune,
        epochs=300, patience=40, lr=1e-3, device=device,
    )
    with torch.no_grad():
        mlp = model(torch.as_tensor(u_train[7000:8000], dtype=torch.float32, device=device)).cpu().numpy()
    mlp_metrics = latent_metrics(mlp, ytune)
    torch.save(model.state_dict(), OUT / "forward_z2.pt")
    payload["z2_320"] = {
        "linear": {**lin_metrics, **lin_info},
        "mlp": {**mlp_metrics, "best_epoch": info["best_epoch"], "tune_mse": info["tune_mse"]},
    }
    log(f"z2 320: linear R2 {lin_metrics['mean_r2']:.3f}, mlp R2 {mlp_metrics['mean_r2']:.3f}")

    np.savez_compressed(
        OUT / "stage8_data.npz",
        u_train=u_train.astype(np.float32),
        u_val=u_val.astype(np.float32),
        u0_tune=u0_tune.astype(np.float32),
        u0_val=u0_val.astype(np.float32),
        scores_train=data["scores_train"].astype(np.float32),
        scores_val=data["scores_val"].astype(np.float32),
        z2_train=data["z2_train"].astype(np.float32),
        z2_val=data["z2_val"].astype(np.float32),
        half=data["half"],
        center=data["center"],
        m_train=data["m_train"],
        m_val=data["m_val"],
    )
    save_json(OUT / "forward_results.json", payload)
    log("saved forward results")
    return payload


def load_forward(path: str, k: int, device):
    model = ForwardMLP(k).to(device)
    model.load_state_dict(torch.load(OUT / path, map_location=device))
    model.eval()
    return model


def stage_floor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(OUT / "stage8_data.npz")
    u_train = data["u_train"].astype(np.float64)
    u0 = data["u0_tune"].astype(np.float64)
    u_true = u_train[7000:8000]
    payload = {
        "initializer_true_mae": float(np.abs(u0 - u_true).mean()),
        "initializer_true_physical_mae": float(np.abs((u0 - u_true) * data["half"][None, :]).mean()),
    }
    for k in (256, 512):
        std = data["scores_train"][:7000, :k].std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        z_obs = data["scores_train"][7000:8000, :k] / std
        model = load_forward(f"forward_z1_{k}.pt", k, device)
        with torch.no_grad():
            pred_true = model(torch.as_tensor(u_true, dtype=torch.float32, device=device)).cpu().numpy()
            pred0 = model(torch.as_tensor(u0, dtype=torch.float32, device=device)).cpu().numpy()
        r_floor = np.linalg.norm(pred_true - z_obs, axis=1)
        r0 = np.linalg.norm(pred0 - z_obs, axis=1)
        payload[f"z1_{k}"] = {
            "r_floor_median": float(np.median(r_floor)),
            "r_floor_p90": float(np.quantile(r_floor, 0.90)),
            "r0_median": float(np.median(r0)),
            "r0_p90": float(np.quantile(r0, 0.90)),
            "ratio_median": float(np.median(r0 / np.maximum(r_floor, 1e-12))),
            "ratio_p90": float(np.quantile(r0 / np.maximum(r_floor, 1e-12), 0.90)),
        }
    z_obs = data["z2_train"][7000:8000]
    model = load_forward("forward_z2.pt", z_obs.shape[1], device)
    with torch.no_grad():
        pred_true = model(torch.as_tensor(u_true, dtype=torch.float32, device=device)).cpu().numpy()
        pred0 = model(torch.as_tensor(u0, dtype=torch.float32, device=device)).cpu().numpy()
    r_floor = np.linalg.norm(pred_true - z_obs, axis=1)
    r0 = np.linalg.norm(pred0 - z_obs, axis=1)
    payload["z2_320"] = {
        "r_floor_median": float(np.median(r_floor)),
        "r_floor_p90": float(np.quantile(r_floor, 0.90)),
        "r0_median": float(np.median(r0)),
        "r0_p90": float(np.quantile(r0, 0.90)),
        "ratio_median": float(np.median(r0 / np.maximum(r_floor, 1e-12))),
        "ratio_p90": float(np.quantile(r0 / np.maximum(r_floor, 1e-12), 0.90)),
    }
    save_json(OUT / "surrogate_floor.json", payload)
    log(f"floor: z2 ratio {payload['z2_320']['ratio_median']:.3f}")
    return payload




def forward_jacobian(model, u: torch.Tensor, eps: float = 1e-4):
    """Finite-difference Jacobian of the learned forward map: N x K x 41."""
    u = u.detach()
    outputs_up, outputs_down = [], []
    for k in range(u.shape[1]):
        up = u.clone()
        down = u.clone()
        up[:, k] += eps
        down[:, k] -= eps
        outputs_up.append(model(up))
        outputs_down.append(model(down))
    jac = torch.stack(
        [(a - b) / (2.0 * eps) for a, b in zip(outputs_up, outputs_down)], dim=2
    )
    return jac.detach()


def stage_jacobian(n_query: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(OUT / "stage8_data.npz")
    u_train = data["u_train"].astype(np.float64)
    u_query = u_train[7000:7000 + n_query]
    payload = {}
    for name, path, target, dim in (
        ("z1_512", "forward_z1_512.pt", data["scores_train"][:, :512], 512),
        ("z2_320", "forward_z2.pt", data["z2_train"], 320),
    ):
        if name.startswith("z1"):
            std = data["scores_train"][:7000, :dim].std(axis=0)
            std = np.where(std < 1e-12, 1.0, std)
            z_train = target[:7000] / std
            z_query = target[7000:7000 + n_query] / std
        else:
            z_train = target[:7000]
            z_query = target[7000:7000 + n_query]
        model = load_forward(path, dim, device)
        ut = torch.as_tensor(u_query, dtype=torch.float32, device=device)
        train_t = torch.as_tensor(u_train[:7000], dtype=torch.float32, device=device)
        dist = torch.cdist(ut, train_t)
        nearest = torch.topk(dist, 1, largest=False, dim=1).indices[:, 0]
        jac = forward_jacobian(model, ut)
        du = torch.as_tensor(u_query - u_train[:7000][nearest.cpu().numpy()], dtype=torch.float32, device=device)
        dz = torch.as_tensor(z_query - z_train[nearest.cpu().numpy()], dtype=torch.float32, device=device)
        dz_lin = torch.einsum("nkj,nj->nk", jac, du)
        cos = torch.sum(dz_lin * dz, dim=1) / (
            torch.linalg.vector_norm(dz_lin, dim=1) * torch.linalg.vector_norm(dz, dim=1) + 1e-12
        )
        rel = torch.linalg.vector_norm(dz_lin - dz, dim=1) / (torch.linalg.vector_norm(dz, dim=1) + 1e-12)
        explained = 1.0 - (torch.linalg.vector_norm(dz_lin - dz, dim=1) ** 2) / (
            torch.linalg.vector_norm(dz, dim=1) ** 2 + 1e-12
        )
        du_norm = torch.linalg.vector_norm(du, dim=1)
        payload[name] = {
            "pairs": int(n_query),
            "cosine_median": float(torch.median(cos)),
            "cosine_p10": float(torch.quantile(cos, 0.10)),
            "relative_l2_median": float(torch.median(rel)),
            "explained_fraction_median": float(torch.median(explained)),
            "explained_fraction_p10": float(torch.quantile(explained, 0.10)),
            "du_norm_median": float(torch.median(du_norm)),
            "du_norm_p90": float(torch.quantile(du_norm, 0.90)),
        }
        log(f"jacobian {name}: cos {payload[name]['cosine_median']:.3f}, rel {payload[name]['relative_l2_median']:.3f}")
    save_json(OUT / "jacobian_quality.json", payload)
    return payload



def state_metrics(pred, target, half):
    error = pred - target
    abs_error = np.abs(error)
    physical = np.abs(half[None, :] * error)
    ss_res = (error ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "mae": float(abs_error.mean()),
        "physical_mae": float(physical.mean()),
        "p90_abs": float(np.quantile(abs_error, 0.90)),
        "p95_abs": float(np.quantile(abs_error, 0.95)),
        "p99_abs": float(np.quantile(abs_error, 0.99)),
        "linf_mean": float(abs_error.max(axis=1).mean()),
        "mean_r2": float(np.mean(r2)),
        "median_r2": float(np.median(r2)),
        "count_r2_below_0_1": int((r2 < 0.1).sum()),
        "per_parameter_r2": r2.tolist(),
        "weak_r2": {str(p): float(r2[p]) for p in WEAK},
        "dangerous_r2": {str(p): float(r2[p]) for p in DANGEROUS},
        "dangerous_physical_mae": float(physical[:, DANGEROUS].mean()),
        "dangerous_physical_p95": float(np.quantile(physical[:, DANGEROUS], 0.95)),
    }


def adam_refine(model, z_obs, u0, sigma, steps, lr, trust, device):
    u0t = torch.as_tensor(u0, dtype=torch.float32, device=device)
    zt = torch.as_tensor(z_obs, dtype=torch.float32, device=device)
    v = torch.tensor(np.arctanh(np.clip(u0, -0.998, 0.998)), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([v], lr=float(lr))
    if sigma is None:
        weight = torch.ones(zt.shape[1], device=device)
    else:
        weight = 1.0 / (torch.as_tensor(sigma, dtype=torch.float32, device=device) ** 2 + 1e-8)
        weight = weight / weight.mean()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        u = torch.tanh(v)
        residual = model(u) - zt
        loss = (residual ** 2 * weight[None, :]).mean() + float(trust) * ((u - u0t) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.tanh(v).cpu().numpy().astype(np.float64)


def lm_refine(model, z_obs, u0, sigma, iterations, damping, trust, device, eps=1e-4):
    u0t = torch.as_tensor(u0, dtype=torch.float32, device=device)
    zt = torch.as_tensor(z_obs, dtype=torch.float32, device=device)
    v = torch.tensor(np.arctanh(np.clip(u0, -0.998, 0.998)), dtype=torch.float32, device=device)
    if sigma is None:
        weight = torch.ones(zt.shape[1], device=device)
    else:
        weight = 1.0 / (torch.as_tensor(sigma, dtype=torch.float32, device=device) ** 2 + 1e-8)
        weight = weight / weight.mean()
    for _ in range(int(iterations)):
        u = torch.tanh(v)
        residual = (model(u) - zt) * torch.sqrt(weight)[None, :]
        jac = torch.empty((v.shape[0], zt.shape[1], v.shape[1]), dtype=torch.float32, device=device)
        with torch.no_grad():
            for k in range(v.shape[1]):
                up = v.clone()
                down = v.clone()
                up[:, k] += eps
                down[:, k] -= eps
                jac[:, :, k] = (model(torch.tanh(up)) - model(torch.tanh(down))) / (2.0 * eps)
                del up, down
        jac = jac * torch.sqrt(weight)[None, :, None]
        trust_res = float(trust) ** 0.5 * (u - u0t)
        trust_jac = float(trust) ** 0.5 * torch.diag_embed(1.0 - u ** 2)
        residual = torch.cat([residual, trust_res], dim=1)
        jac = torch.cat([jac, trust_jac], dim=1)
        normal = torch.einsum("nki,nkj->nij", jac, jac)
        diag = torch.diagonal(normal, dim1=1, dim2=2)
        normal = normal + float(damping) * torch.diag_embed(diag) + 1e-8 * torch.eye(normal.shape[1], device=device)[None, :, :]
        rhs = -torch.einsum("nki,nk->ni", jac, residual)
        delta = torch.linalg.solve(normal, rhs)
        v = v + delta
    with torch.no_grad():
        return torch.tanh(v).cpu().numpy().astype(np.float64)



def stage_refine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(OUT / "stage8_data.npz")
    u_train = data["u_train"].astype(np.float64)
    u_true = u_train[7000:8000]
    u0 = data["u0_tune"].astype(np.float64)
    half = data["half"]
    payload = {
        "initial": state_metrics(u0, u_true, half),
        "grid": {},
    }
    for name, path, dim, target in (
        ("z1_512", "forward_z1_512.pt", 512, data["scores_train"][:, :512]),
        ("z2_320", "forward_z2.pt", 320, data["z2_train"]),
    ):
        if name.startswith("z1"):
            std = data["scores_train"][:7000, :dim].std(axis=0)
            std = np.where(std < 1e-12, 1.0, std)
            ytr = target[:7000] / std
            z_obs = target[7000:8000] / std
        else:
            ytr = target[:7000]
            z_obs = target[7000:8000]
        model = load_forward(path, dim, device)
        with torch.no_grad():
            train_pred = model(torch.as_tensor(u_train[:7000], dtype=torch.float32, device=device)).cpu().numpy()
        sigma = np.sqrt(np.mean((train_pred - ytr) ** 2, axis=0) + 1e-10)
        obs_payload = {"initial": state_metrics(u0, u_true, half), "adam": {}, "lm": {}}
        best_adam = None
        for steps in (5, 10, 20):
            for lr in (1e-2, 3e-2, 1e-1):
                pred = adam_refine(model, z_obs, u0, None, steps, lr, 1e-2, device)
                mae = float(np.abs(pred - u_true).mean())
                key = f"steps{steps}_lr{lr:g}"
                obs_payload["adam"][key] = {"true_mae": mae}
                if best_adam is None or mae < best_adam[0]:
                    best_adam = (mae, key, steps, lr, pred)
                log(f"{name} adam {key}: MAE {mae:.5f}")
        best_lm = None
        for iterations in (1, 2, 4, 8):
            for damping in (1e-3, 1e-2, 1e-1, 1.0):
                pred = lm_refine(model, z_obs, u0, None, iterations, damping, 1e-2, device)
                mae = float(np.abs(pred - u_true).mean())
                key = f"iter{iterations}_damp{damping:g}"
                obs_payload["lm"][key] = {"true_mae": mae}
                if best_lm is None or mae < best_lm[0]:
                    best_lm = (mae, key, iterations, damping, pred)
                log(f"{name} lm {key}: MAE {mae:.5f}")
        obs_payload["best_adam"] = {
            "key": best_adam[1], "true_mae": best_adam[0],
            "metrics": state_metrics(best_adam[4], u_true, half),
        }
        obs_payload["best_lm"] = {
            "key": best_lm[1], "true_mae": best_lm[0],
            "metrics": state_metrics(best_lm[4], u_true, half),
        }
        use_lm = best_lm[0] <= best_adam[0]
        if use_lm:
            _, key, iterations, damping, _ = best_lm
            weighted_pred = lm_refine(model, z_obs, u0, sigma, iterations, damping, 1e-2, device)
            weighted_name = f"lm_{key}_weighted"
        else:
            _, key, steps, lr, _ = best_adam
            weighted_pred = adam_refine(model, z_obs, u0, sigma, steps, lr, 1e-2, device)
            weighted_name = f"adam_{key}_weighted"
        obs_payload["weighted"] = {
            "name": weighted_name,
            "true_mae": float(np.abs(weighted_pred - u_true).mean()),
            "metrics": state_metrics(weighted_pred, u_true, half),
        }
        refined = best_lm[4] if use_lm else best_adam[4]
        with torch.no_grad():
            zt = torch.as_tensor(z_obs, dtype=torch.float32, device=device)
            r_true = torch.linalg.vector_norm(
                model(torch.as_tensor(u_true, dtype=torch.float32, device=device)) - zt, dim=1
            ).cpu().numpy()
            r0 = torch.linalg.vector_norm(
                model(torch.as_tensor(u0, dtype=torch.float32, device=device)) - zt, dim=1
            ).cpu().numpy()
            rref = torch.linalg.vector_norm(
                model(torch.as_tensor(refined, dtype=torch.float32, device=device)) - zt, dim=1
            ).cpu().numpy()
        obs_payload["surrogate"] = {
            "r_floor_median": float(np.median(r_true)),
            "r0_median": float(np.median(r0)),
            "r_ref_median": float(np.median(rref)),
            "r0_over_floor_median": float(np.median(r0 / np.maximum(r_true, 1e-12))),
            "true_mae_initial": float(np.abs(u0 - u_true).mean()),
            "true_mae_refined": float(np.abs(refined - u_true).mean()),
            "exploitation_flag": bool(
                float(np.median(rref)) < float(np.median(r0))
                and float(np.abs(refined - u_true).mean()) >= float(np.abs(u0 - u_true).mean())
            ),
        }
        payload["grid"][name] = obs_payload
    save_json(OUT / "refinement_results.json", payload)
    log("saved refinement results")
    return payload



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pca", "forward", "floor", "jacobian", "refine", "all"))
    args = parser.parse_args()
    if args.stage in ("pca", "all"):
        stage_pca()
    if args.stage in ("forward", "all"):
        stage_forward()
    if args.stage in ("floor", "all"):
        stage_floor()
    if args.stage in ("jacobian", "all"):
        stage_jacobian()
    if args.stage in ("refine", "all"):
        stage_refine()


if __name__ == "__main__":
    main()
