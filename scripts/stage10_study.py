"""Stage 10: S11-targeted specification-gradient analysis and oracle gates."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
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

from solution.simulator import load_simulator
from solution.utils import save_json
from scripts.stage9_study import (
    ResidualHead,
    base_corrections,
    extract_features,
    hard_miss,
    load_base,
    load_cached,
    smooth_terms,
    stable_smax,
    _sim_worker,
    _init_sim_worker,
)

OUT = ROOT / "artifacts/stage10"
OUT.mkdir(parents=True, exist_ok=True)
TAU = 0.10
BETA = 0.50
TAU_W = 0.25
SIM_CALLS = 0


def log(message: str) -> None:
    print(f"[stage10] {message}", flush=True)


def simulate(sim, m):
    global SIM_CALLS
    SIM_CALLS += 1
    return sim.simulate(np.clip(m, sim.lower_bounds, sim.upper_bounds))


def load_policy(split="train", n=8000, start=7000, window=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sim, cfg, info = load_base(
        str(ROOT / "configs/route_delay_fno_r3_m32.json"),
        str(ROOT / "runs/route3_modes/r3_m32_s2026/best.pt"),
        device,
    )
    responses, m_all, seeds = load_cached(split, n)
    lo = start if split == "train" else 0
    hi = min(len(m_all), lo + int(window)) if split == "train" else len(m_all)
    responses = responses[lo:hi]
    m = m_all[lo:hi]
    seeds = seeds[lo:hi]
    features = extract_features(model, responses, device)
    base_delta = base_corrections(model, responses, device)
    head = ResidualHead(features.shape[1]).to(device)
    head.load_state_dict(torch.load(OUT.parent / "stage9/residual_head_full.pt", map_location=device))
    head.eval()
    with torch.no_grad():
        residual = head(features.to(device)).cpu().numpy()
    half = 0.5 * (np.asarray(sim.upper_bounds) - np.asarray(sim.lower_bounds))
    y = base_delta / half[None, :] + residual
    return {
        "model": model,
        "sim": sim,
        "responses": responses,
        "m": m,
        "seeds": seeds,
        "features": features,
        "base_delta": base_delta,
        "residual": residual,
        "half": half,
        "y": y,
        "device": device,
    }


def per_requirement_losses(response, sim):
    s11_db, s21_db = sim.response_db(response)
    freqs = np.asarray(sim.frequency_hz)
    q = []
    for req in sim.requirements:
        band = (freqs >= float(req.start_mhz) * 1e6) & (freqs <= float(req.stop_mhz) * 1e6)
        trace = s11_db if str(req.trace).upper() == "S11" else s21_db
        if req.operator == "<=":
            qk = stable_smax(trace[band], TAU) - float(req.threshold_db)
        else:
            qk = float(req.threshold_db) - (-stable_smax(-trace[band], TAU))
        q.append(qk)
    q = np.asarray(q)
    soft = BETA * np.logaddexp(0.0, q / BETA)
    l11_1, l11_2 = soft[0], soft[1]
    l11 = stable_smax(np.asarray([l11_1, l11_2]), TAU_W)
    l21 = soft[2:]
    l21_barrier = float(np.sum(l21))
    lfull = smooth_terms(response, sim, TAU, BETA, TAU_W)["loss"]
    return {
        "q": q,
        "l11_1": float(l11_1),
        "l11_2": float(l11_2),
        "l11": float(l11),
        "l21_barrier": float(l21_barrier),
        "l21": l21.tolist(),
        "lfull": float(lfull),
    }



class S11Head(nn.Module):
    def __init__(self, feature_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU())
        self.norm1 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features):
        h = self.stem(features)
        h = h + self.fc2(torch.nn.functional.gelu(self.norm2(self.fc1(torch.nn.functional.gelu(self.norm1(h))))))
        return self.head(h)


def eval_hard(y, m, half, sim):
    return hard_summary(y, m, half, sim)


def stage_audit(n_states=256):
    policy = load_policy("train", 8000, 7000)
    sim = policy["sim"]
    m = policy["m"][:n_states]
    y = policy["y"][:n_states]
    half = policy["half"]
    lower = np.asarray(sim.lower_bounds, dtype=np.float64)
    upper = np.asarray(sim.upper_bounds, dtype=np.float64)
    raw = m + half[None, :] * y
    clipped = np.clip(raw, lower, upper)
    outside = (raw < lower[None, :]) | (raw > upper[None, :])
    s11_miss, overall = [], []
    for row in range(n_states):
        response = simulate(sim, clipped[row])
        details = sim.check_response(response)
        s11_miss.append(max(float(details["passband_s11_1_miss_db"]), float(details["passband_s11_2_miss_db"])))
        overall.append(float(details["max_spec_miss_db"]))
    s11_miss = np.asarray(s11_miss)
    overall = np.asarray(overall)
    median_s11 = float(np.median(s11_miss))
    high = s11_miss > median_s11
    low = ~high
    abs_y = np.abs(y)
    payload = {
        "n_states": int(n_states),
        "seeds": policy["seeds"][:n_states].tolist(),
        "y_abs_mean": float(abs_y.mean()),
        "y_abs_p90": float(np.quantile(abs_y, 0.90)),
        "y_l2_mean": float(np.linalg.norm(y, axis=1).mean()),
        "y_linf_mean": float(abs_y.max(axis=1).mean()),
        "outside_coordinate_fraction": float(outside.mean()),
        "sample_with_clip_fraction": float(outside.any(axis=1).mean()),
        "raw_overall_mean": float(overall.mean()),
        "s11_mean": float(s11_miss.mean()),
        "s11_median": median_s11,
        "per_parameter_clip_rate": outside.mean(axis=0).tolist(),
        "per_parameter_lower_hit_rate": (raw < lower[None, :]).mean(axis=0).tolist(),
        "per_parameter_upper_hit_rate": (raw > upper[None, :]).mean(axis=0).tolist(),
        "high_s11": {
            "count": int(high.sum()),
            "y_abs_mean": float(abs_y[high].mean()),
            "outside_rate": float(outside[high].mean()),
            "clip_sample_rate": float(outside[high].any(axis=1).mean()),
        },
        "low_s11": {
            "count": int(low.sum()),
            "y_abs_mean": float(abs_y[low].mean()),
            "outside_rate": float(outside[low].mean()),
            "clip_sample_rate": float(outside[low].any(axis=1).mean()),
        },
        "simulator_calls": int(SIM_CALLS),
    }
    save_json(OUT / "clipping_audit.json", payload)
    log(f"audit: outside {payload['outside_coordinate_fraction']:.4f}, clip samples {payload['sample_with_clip_fraction']:.3f}, s11 {payload['s11_mean']:.2f}")
    return payload



def _cos_rows(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def stage_gradients(n_states=32, eps=0.01):
    policy = load_policy("train", 8000, 7000)
    sim = policy["sim"]
    m = policy["m"][:n_states]
    y0 = policy["y"][:n_states]
    half = policy["half"]
    keys = ("l11_1", "l11_2", "l11", "l21_barrier", "lfull")
    grads = {key: np.zeros((n_states, 41), dtype=np.float64) for key in keys}
    base_loss = {key: np.zeros(n_states, dtype=np.float64) for key in keys}
    for row in range(n_states):
        base = per_requirement_losses(simulate(sim, m[row] + half * y0[row]), sim)
        for key in keys:
            base_loss[key][row] = base[key]
        for j in range(41):
            yp = y0[row].copy()
            ym = y0[row].copy()
            yp[j] += eps
            ym[j] -= eps
            lp = per_requirement_losses(simulate(sim, m[row] + half * yp), sim)
            lm = per_requirement_losses(simulate(sim, m[row] + half * ym), sim)
            for key in keys:
                grads[key][row, j] = (lp[key] - lm[key]) / (2.0 * eps)
        if row % 4 == 0:
            log(f"gradient state {row + 1}/{n_states}")

    def stat(x):
        return {
            "median": float(np.median(x)),
            "p10": float(np.quantile(x, 0.10)),
            "p90": float(np.quantile(x, 0.90)),
        }

    norms = {key: np.linalg.norm(grads[key], axis=1) for key in keys}
    conflict = _cos_rows(grads["l11"], grads["l21_barrier"])
    band_conflict = _cos_rows(grads["l11_1"], grads["l11_2"])
    _, sv, vt = np.linalg.svd(grads["l11"], full_matrices=False)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    d95 = int(np.searchsorted(energy, 0.95) + 1)
    d = int(min(16, max(1, d95)))
    B11 = vt[:d].T
    payload = {
        "n_states": int(n_states),
        "eps": float(eps),
        "base_loss_mean": {key: float(base_loss[key].mean()) for key in keys},
        "gradient_norm": {key: stat(norms[key]) for key in keys},
        "conflict_11_21": {
            "median": float(np.median(conflict)),
            "p10": float(np.quantile(conflict, 0.10)),
            "p90": float(np.quantile(conflict, 0.90)),
            "fraction_negative": float(np.mean(conflict < 0)),
            "fraction_below_minus_0_5": float(np.mean(conflict < -0.5)),
        },
        "conflict_s11_bands": {
            "median": float(np.median(band_conflict)),
            "p10": float(np.quantile(band_conflict, 0.10)),
            "p90": float(np.quantile(band_conflict, 0.90)),
            "fraction_negative": float(np.mean(band_conflict < 0)),
            "fraction_below_minus_0_5": float(np.mean(band_conflict < -0.5)),
        },
        "s11_singular_values": sv.tolist(),
        "s11_energy": energy.tolist(),
        "s11_rank_d95": d95,
        "s11_basis_dim": d,
        "top_parameters": [
            {"parameter": int(i), "mean_abs": float(np.abs(grads["l11"][:, i]).mean())}
            for i in np.argsort(-np.abs(grads["l11"]).mean(axis=0))[:15]
        ],
        "simulator_calls": int(SIM_CALLS),
    }
    np.savez_compressed(
        OUT / "s11_gradients.npz",
        m=m, y0=y0, half=half,
        l11=grads["l11"], l11_1=grads["l11_1"], l11_2=grads["l11_2"],
        l21_barrier=grads["l21_barrier"], lfull=grads["lfull"],
        B11=B11,
    )
    save_json(OUT / "gradient_analysis.json", payload)
    log(f"gradients: rank95 {d95}, basis {d}, conflict {payload['conflict_11_21']['median']:.3f}")
    return payload



def bound_penalty(y, m, half, lower, upper):
    raw = m + half[None, :] * y
    return float(np.mean(np.maximum(lower[None, :] - raw, 0.0) ** 2 + np.maximum(raw - upper[None, :], 0.0) ** 2))


def bound_gradient(y, m, half, lower, upper):
    raw = m + half[None, :] * y
    return (1.0 / 41.0) * (
        -2.0 * np.maximum(lower[None, :] - raw, 0.0) * half[None, :]
        + 2.0 * np.maximum(raw - upper[None, :], 0.0) * half[None, :]
    )


def hard_summary(y, m, half, sim):
    rows = []
    for row in range(len(m)):
        response = simulate(sim, m[row] + half * y[row])
        details = sim.check_response(response)
        rows.append({
            "s11_1": float(details["passband_s11_1_miss_db"]),
            "s11_2": float(details["passband_s11_2_miss_db"]),
            "s21": max(float(details["passband_s21_%d_miss_db" % k]) for k in range(1, 6)),
            "overall": float(details["max_spec_miss_db"]),
            "pass": bool(details["all_specs_pass"]),
            "s11_1_pass": bool(details["passband_s11_1_pass"]),
            "s11_2_pass": bool(details["passband_s11_2_pass"]),
        })
    out = {}
    for key in ("s11_1", "s11_2", "s21", "overall"):
        values = np.asarray([row[key] for row in rows])
        out[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
        }
    out["s11_mean"] = float(
        (np.asarray([r["s11_1"] for r in rows]) + np.asarray([r["s11_2"] for r in rows])).mean() / 2.0
    )
    out["s11_1_pass_count"] = int(sum(r["s11_1_pass"] for r in rows))
    out["s11_2_pass_count"] = int(sum(r["s11_2_pass"] for r in rows))
    out["pass_count"] = int(sum(r["pass"] for r in rows))
    return out



def stage_oracle(n_multi=16, eta_values=(0.005, 0.01, 0.02, 0.05)):
    data = np.load(OUT / "s11_gradients.npz")
    m, y0, half = data["m"], data["y0"], data["half"]
    g11, g21, B11 = data["l11"], data["l21_barrier"], data["B11"]
    sim = load_simulator()
    lower = np.asarray(sim.lower_bounds, dtype=np.float64)
    upper = np.asarray(sim.upper_bounds, dtype=np.float64)
    ratio = np.linalg.norm(g11, axis=1) / (np.linalg.norm(g21, axis=1) + 1e-12)
    lam21 = float(np.clip(np.median(ratio), 0.1, 10.0))
    lam_b, lam_t = 1e-3, 1e-4
    gJ = g11 + lam21 * g21 + lam_b * bound_gradient(y0, m, half, lower, upper)
    g_sub = (gJ @ B11) @ B11.T
    payload = {
        "lam21": lam21,
        "lambda_bound": lam_b,
        "lambda_trust": lam_t,
        "basis_dim": int(B11.shape[1]),
        "base": hard_summary(y0, m, half, sim),
        "line_search": {"full41": {}, "B11": {}},
    }
    for name, direction in (("full41", gJ), ("B11", g_sub)):
        for eta in eta_values:
            candidate = y0 - float(eta) * direction
            payload["line_search"][name][str(eta)] = hard_summary(candidate, m, half, sim)
            log(f"oracle {name} eta {eta}: S11 {payload['line_search'][name][str(eta)]['s11_mean']:.2f}, overall {payload['line_search'][name][str(eta)]['overall']['mean']:.2f}")
    all_rows = [
        (payload["line_search"][name][eta]["s11_mean"], name, float(eta))
        for name in ("full41", "B11") for eta in payload["line_search"][name]
    ]
    _, best_name, best_eta = min(all_rows, key=lambda row: row[0])
    payload["multi_step"] = {"direction": best_name, "eta": best_eta, "steps": {}}
    y = y0[:n_multi].copy()
    m_multi = m[:n_multi]
    for step in (1, 2, 4, 8):
        while payload["multi_step"].get("current_step", 0) < step:
            cur = payload["multi_step"].get("current_step", 0)
            g = np.zeros((n_multi, 41), dtype=np.float64)
            for row in range(n_multi):
                for j in range(41):
                    yp = y[row].copy()
                    ym = y[row].copy()
                    yp[j] += 0.01
                    ym[j] -= 0.01
                    lp = per_requirement_losses(simulate(sim, m_multi[row] + half * yp), sim)
                    lm = per_requirement_losses(simulate(sim, m_multi[row] + half * ym), sim)
                    jp = lp["l11"] + lam21 * lp["l21_barrier"]
                    jm = lm["l11"] + lam21 * lm["l21_barrier"]
                    g[row, j] = (jp - jm) / 0.02
                g[row] += bound_gradient(y[row:row+1], m_multi[row:row+1], half, lower, upper)[0]
            if best_name == "B11":
                g = (g @ B11) @ B11.T
            y = y - best_eta * g
            payload["multi_step"]["current_step"] = cur + 1
        payload["multi_step"]["steps"][str(step)] = hard_summary(y, m_multi, half, sim)
        log(f"oracle multistep {step}: S11 {payload['multi_step']['steps'][str(step)]['s11_mean']:.2f}, overall {payload['multi_step']['steps'][str(step)]['overall']['mean']:.2f}")
    payload["best_line_search_s11"] = min(all_rows, key=lambda row: row[0])
    ms = payload["multi_step"]["steps"]
    traj = [payload["base"]["s11_mean"]] + [ms[str(step)]["s11_mean"] for step in (1, 2, 4, 8)]
    continuous = all(traj[i + 1] < traj[i] - 1e-6 for i in range(len(traj) - 1))
    s21_ok = ms["8"]["s21"]["mean"] < 1.0
    payload["gate"] = {
        "s11_below_8": payload["best_line_search_s11"][0] < 8.0,
        "s11_pass_nonzero": any(
            payload["line_search"][name][eta]["s11_1_pass_count"] > 0
            or payload["line_search"][name][eta]["s11_2_pass_count"] > 0
            for name in payload["line_search"] for eta in payload["line_search"][name]
        ),
        "overall_below_10_multistep": bool(ms["8"]["overall"]["mean"] < 10.0 and s21_ok),
        "continuous_decline": bool(continuous),
        "multistep_s11_trajectory": traj,
        "multistep_s21_final": ms["8"]["s21"]["mean"],
    }
    payload["gate"]["passed"] = bool(
        payload["gate"]["s11_below_8"]
        or payload["gate"]["s11_pass_nonzero"]
        or payload["gate"]["overall_below_10_multistep"]
        or payload["gate"]["continuous_decline"]
    )
    payload["simulator_calls"] = int(SIM_CALLS)
    save_json(OUT / "oracle.json", payload)
    log(f"oracle gate {payload['gate']}")
    return payload





def eval_J(y, m, y0, half, lower, upper, lam21, lam_b, lam_t):
    values = []
    for row in range(len(m)):
        loss = per_requirement_losses(simulate(load_simulator(), m[row] + half * y[row]), load_simulator())
        values.append(
            loss["l11"] + lam21 * loss["l21_barrier"]
            + lam_b * bound_penalty(y[row:row+1], m[row:row+1], half, lower, upper)
            + lam_t * float(np.sum((y[row] - y0[row]) ** 2))
        )
    return float(np.mean(values))


def stage_lowd_spsa(n_states=16, eps=0.01):
    data = np.load(OUT / "s11_gradients.npz")
    m, y0, half, g11, g21, B11 = data["m"][:n_states], data["y0"][:n_states], data["half"], data["l11"][:n_states], data["l21_barrier"][:n_states], data["B11"]


def stage_s11head(steps=200, batch_size=16, k_dirs=8, eps=0.01, lr=1e-4, full=False, workers=8):
    global SIM_CALLS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = mp.get_context("fork").Pool(processes=int(workers), initializer=_init_sim_worker)
    try:
        data = np.load(OUT / "s11_gradients.npz")
        B11 = data["B11"]
        oracle = json.loads((OUT / "oracle.json").read_text())
        lam21 = float(oracle["lam21"])
        lam_b, lam_t = 1e-3, 1e-3
        policy = load_policy("train", 8000, 0, window=8000)
        sim = policy["sim"]
        if full:
            m_train = policy["m"]
            y_stage_train = policy["y"]
            features_train = policy["features"]
        else:
            m_train = policy["m"][:7000]
            y_stage_train = policy["y"][:7000]
            features_train = policy["features"][:7000]
        if full:
            val = load_policy("validation", 1000)
            eval_m = val["m"]
            eval_y_stage = val["y"]
            eval_features = val["features"]
            eval_name = "validation"
        else:
            eval_m = policy["m"][7000:8000]
            eval_y_stage = policy["y"][7000:8000]
            eval_features = policy["features"][7000:8000]
            eval_name = "tune"
        base_eval = hard_summary(eval_y_stage, eval_m, policy["half"], sim)
        log(f"base {eval_name}: S11 {base_eval['s11_mean']:.2f}, overall {base_eval['overall']['mean']:.2f}")
        head = S11Head(features_train.shape[1], B11.shape[1]).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=1e-5)
        rng = np.random.default_rng(2026)
        history = []
        for step in range(1, int(steps) + 1):
            idx = rng.choice(len(m_train), size=min(int(batch_size), len(m_train)), replace=False)
            f = features_train[idx].to(device)
            c = head(f)
            y = torch.as_tensor(y_stage_train[idx], dtype=torch.float32, device=device) + (
                torch.as_tensor(B11, dtype=torch.float32, device=device) @ c.T
            ).T
            v = rng.choice([-1.0, 1.0], size=(len(idx), int(k_dirs), B11.shape[1]))
            c_np = c.detach().cpu().numpy()
            b_np = B11
            m_batch = m_train[idx]
            candidates = []
            for i in range(len(idx)):
                for k in range(int(k_dirs)):
                    for sign in (1.0, -1.0):
                        cpert = c_np[i] + sign * float(eps) * v[i, k]
                        ypert = y_stage_train[idx[i]] + b_np @ cpert
                        candidates.append(np.clip(m_batch[i] + policy["half"] * ypert, sim.lower_bounds, sim.upper_bounds))
            responses_corr = pool.map(_sim_worker, candidates, chunksize=8)
            SIM_CALLS += len(candidates)
            losses = []
            for resp in responses_corr:
                loss = per_requirement_losses(resp, sim)
                losses.append(loss["l11"] + lam21 * loss["l21_barrier"])
            losses = np.asarray(losses).reshape(len(idx), int(k_dirs), 2)
            grad = np.mean(
                ((losses[:, :, 0] - losses[:, :, 1]) / (2.0 * float(eps)))[..., None] * v,
                axis=1,
            )
            grad_t = torch.as_tensor(grad, dtype=torch.float32, device=device)
            pseudo = (c * grad_t).sum(dim=1).mean()
            bound = bound_penalty(y.detach().cpu().numpy(), m_batch, policy["half"], np.asarray(sim.lower_bounds), np.asarray(sim.upper_bounds))
            trust = c.pow(2).mean()
            loss = pseudo + lam_b * bound + lam_t * trust
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            history.append({"step": step, "loss": float(loss.detach()), "pseudo": float(pseudo.detach()), "bound": float(bound), "trust": float(trust.detach()), "c_norm": float(torch.linalg.vector_norm(c.detach(), dim=1).mean())})
            if step % 10 == 0:
                log(f"step {step}: loss {history[-1]['loss']:.3f}, c_norm {history[-1]['c_norm']:.4f}")
            if step in (50, 100, 150, 200):
                with torch.no_grad():
                    c_sub = head(eval_features[:64].to(device)).cpu().numpy()
                y_sub = eval_y_stage[:64] + (B11 @ c_sub.T).T
                sub = hard_summary(y_sub, eval_m[:64], policy["half"], sim)
                log(f"step {step} {eval_name}64: S11 {sub['s11_mean']:.2f}, overall {sub['overall']['mean']:.2f}")
        with torch.no_grad():
            c_eval = head(eval_features.to(device)).cpu().numpy()
        y_eval = eval_y_stage + (B11 @ c_eval.T).T
        refined = hard_summary(y_eval, eval_m, policy["half"], sim)
        torch.save(head.state_dict(), OUT / ("s11_head_full.pt" if full else "s11_head.pt"))
        payload = {
            "steps": int(steps), "batch_size": int(batch_size), "k_dirs": int(k_dirs),
            "eps": float(eps), "lr": float(lr), "lambda21": lam21, "lambda_bound": lam_b,
            "lambda_trust": lam_t, "basis_dim": int(B11.shape[1]), "eval_split": eval_name,
            "base_eval": base_eval, "refined_eval": refined,
            "c_stats": {"mean_abs": float(np.abs(c_eval).mean()), "p90_abs": float(np.quantile(np.abs(c_eval), 0.90))},
            "history": history, "simulator_calls": int(SIM_CALLS),
        }
        save_json(OUT / ("s11_head_training_full.json" if full else "s11_head_training.json"), payload)
        log(f"S11 head {eval_name}: S11 {refined['s11_mean']:.2f}, overall {refined['overall']['mean']:.2f}, pass {refined['pass_count']}")
        return payload
    finally:
        pool.close()
        pool.join()

    sim = load_simulator()
    lower = np.asarray(sim.lower_bounds, dtype=np.float64)
    upper = np.asarray(sim.upper_bounds, dtype=np.float64)
    ratio = np.linalg.norm(g11, axis=1) / (np.linalg.norm(g21, axis=1) + 1e-12)
    lam21 = float(np.clip(np.median(ratio), 0.1, 10.0))
    lam_b, lam_t = 1e-3, 1e-4
    gJ = g11 + lam21 * g21 + lam_b * bound_gradient(y0, m, half, lower, upper)
    g_exact = gJ @ B11
    rng = np.random.default_rng(2026)
    payload = {"eps": eps, "basis_dim": int(B11.shape[1]), "lam21": lam21, "K": {}}
    for k in (2, 4, 8):
        cosines, signs = [], []
        for row in range(n_states):
            estimate = np.zeros(B11.shape[1], dtype=np.float64)
            for _ in range(k):
                v = rng.choice([-1.0, 1.0], size=B11.shape[1])
                yp = y0[row] + eps * (B11 @ v)
                ym = y0[row] - eps * (B11 @ v)
                lp = per_requirement_losses(simulate(sim, m[row] + half * yp), sim)
                lm = per_requirement_losses(simulate(sim, m[row] + half * ym), sim)
                jp = lp["l11"] + lam21 * lp["l21_barrier"] + lam_b * bound_penalty(yp[None, :], m[row:row+1], half, lower, upper)
                jm = lm["l11"] + lam21 * lm["l21_barrier"] + lam_b * bound_penalty(ym[None, :], m[row:row+1], half, lower, upper)
                estimate += (jp - jm) / (2.0 * eps) * v
            estimate /= k
            g = g_exact[row]
            cosines.append(float(estimate @ g / (np.linalg.norm(estimate) * np.linalg.norm(g) + 1e-12)))
            signs.append(float(np.mean(np.sign(estimate) == np.sign(g))))
        payload["K"][str(k)] = {
            "cosine_median": float(np.median(cosines)),
            "cosine_p10": float(np.quantile(cosines, 0.10)),
            "sign_agreement_median": float(np.median(signs)),
        }
        log(f"lowd spsa K={k}: cos {np.median(cosines):.3f}, sign {np.median(signs):.3f}")
    payload["simulator_calls"] = int(SIM_CALLS)
    save_json(OUT / "lowd_spsa.json", payload)
    return payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "gradients", "oracle", "lowd", "s11head", "s11headfull", "all"))
    args = parser.parse_args()
    if args.stage in ("audit", "all"):
        stage_audit()
    if args.stage in ("gradients", "all"):
        stage_gradients()
    if args.stage in ("oracle", "all"):
        stage_oracle()
    if args.stage in ("lowd", "all"):
        stage_lowd_spsa()
    if args.stage in ("s11head", "all"):
        stage_s11head()
    if args.stage in ("s11headfull", "all"):
        stage_s11head(full=True)


if __name__ == "__main__":
    main()
