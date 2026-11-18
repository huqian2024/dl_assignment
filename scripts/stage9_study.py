"""Stage 9: smooth specification objective and zero-order policy calibration.

Only cached states are used.  Simulator calls are on candidate corrected
vectors derived from cached m_current; no new states are sampled.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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

from solution.config import load_config
from solution.data import CM19ResponseDataset
from solution.models import build_model
from solution.simulator import load_simulator
from solution.trainer import load_checkpoint_into
from solution.utils import get_device, save_json

OUT = ROOT / "artifacts/stage9"
OUT.mkdir(parents=True, exist_ok=True)
WEAK = [21, 23, 26, 31]
DANGEROUS = [5, 8, 10, 22, 23, 24, 31, 32, 35, 37]
SIM_CALLS = 0
_WORKER_SIM = None


def _init_sim_worker():
    global _WORKER_SIM
    _WORKER_SIM = load_simulator()


def _sim_worker(m):
    assert _WORKER_SIM is not None
    return _WORKER_SIM.simulate(np.clip(m, _WORKER_SIM.lower_bounds, _WORKER_SIM.upper_bounds))


def log(message: str) -> None:
    print(f"[stage9] {message}", flush=True)


def load_base(config_path: str, checkpoint: str, device):
    cfg = load_config(config_path)
    sim = load_simulator()
    model = build_model(cfg, sim).to(device)
    info = load_checkpoint_into(model, checkpoint)
    model.eval()
    return model, sim, cfg, info


def load_cached(split: str, n: int):
    dataset = CM19ResponseDataset("cache/dataset", split=split, limit=n)
    rows = np.asarray(dataset._indices)
    responses = np.array(dataset.responses[rows]).astype(np.float32)
    m_current = np.array(dataset.m_current[rows]).astype(np.float64)
    seeds = np.asarray(dataset.seeds_slice())
    return responses, m_current, seeds


def simulate(sim, m: np.ndarray) -> np.ndarray:
    global SIM_CALLS
    SIM_CALLS += 1
    return sim.simulate(np.clip(m, sim.lower_bounds, sim.upper_bounds))


def stable_smax(x: np.ndarray, tau: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    return float(tau * (m / tau + np.log(np.sum(np.exp((x - m) / tau))) - np.log(len(x))) + 0.0)


def stable_smin(x: np.ndarray, tau: float) -> float:
    return -stable_smax(-np.asarray(x, dtype=np.float64), tau)


def smooth_terms(response, sim, tau=0.25, beta=0.5, tau_w=0.5, lambda_w=1.0):
    s11_db, s21_db = sim.response_db(response)
    freqs = np.asarray(sim.frequency_hz)
    q = []
    details = []
    for req in sim.requirements:
        band = (freqs >= float(req.start_mhz) * 1e6) & (freqs <= float(req.stop_mhz) * 1e6)
        trace = s11_db if str(req.trace).upper() == "S11" else s21_db
        if req.operator == "<=":
            value = stable_smax(trace[band], tau)
            qk = value - float(req.threshold_db)
        else:
            value = stable_smin(trace[band], tau)
            qk = float(req.threshold_db) - value
        q.append(qk)
        details.append({"name": req.name, "trace": str(req.trace), "operator": req.operator, "q": float(qk)})
    q = np.asarray(q, dtype=np.float64)
    soft = beta * np.logaddexp(0.0, q / beta)
    spec_loss = float(np.mean(soft))
    worst_loss = float(tau_w * (np.max(q / tau_w) + np.log(np.sum(np.exp((q - np.max(q)) / tau_w)))))
    total = spec_loss + float(lambda_w) * worst_loss
    return {
        "loss": total,
        "spec_loss": spec_loss,
        "worst_loss": worst_loss,
        "q": q.tolist(),
        "max_q": float(np.max(q)),
        "details": details,
    }


def hard_miss(response, sim):
    details = sim.check_response(response)
    return float(details["max_spec_miss_db"]), details


def base_corrections(model, responses, device, batch_size=64):
    deltas = []
    with torch.no_grad():
        for begin in range(0, len(responses), batch_size):
            x = torch.as_tensor(responses[begin:begin + batch_size], dtype=torch.float32, device=device)
            deltas.append(model(x).cpu().numpy().astype(np.float64))
    return np.concatenate(deltas, axis=0)


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


class ResidualHead(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 256, out_dim: int = 41):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU())
        self.block = ResidualBlock(hidden)
        self.head = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features):
        return self.head(self.block(self.stem(features)))


def extract_features(model, responses, device, batch_size=32):
    outputs = []
    with torch.no_grad():
        for begin in range(0, len(responses), batch_size):
            x = torch.as_tensor(responses[begin:begin + batch_size], dtype=torch.float32, device=device)
            parts = [branch(x) for branch in model.branches]
            features = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            outputs.append(features.flatten(start_dim=1).cpu())
    return torch.cat(outputs, dim=0)


def evaluate_corrections(sim, m, delta):
    misses = []
    passes = []
    for row in range(len(m)):
        response = simulate(sim, m[row] + delta[row])
        miss, details = hard_miss(response, sim)
        misses.append(miss)
        passes.append(bool(details["all_specs_pass"]))
    misses = np.asarray(misses)
    return {
        "mean": float(misses.mean()),
        "median": float(np.median(misses)),
        "p90": float(np.quantile(misses, 0.90)),
        "worst": float(misses.max()),
        "pass_rate": float(np.mean(passes)),
    }


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def rankdata(x):
    order = np.argsort(np.asarray(x, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def stage_calibration(n_states=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sim, cfg, info = load_base(
        str(ROOT / "configs/route_delay_fno_r3_m32.json"),
        str(ROOT / "runs/route3_modes/r3_m32_s2026/best.pt"),
        device,
    )
    responses, m_current, seeds = load_cached("train", 8000)
    idx = selected_state_indices(n_states)
    m = m_current[idx]
    delta = base_corrections(model, responses[idx], device)
    half = 0.5 * (np.asarray(sim.upper_bounds) - np.asarray(sim.lower_bounds))
    corrected = np.clip(m + delta, sim.lower_bounds, sim.upper_bounds)
    hard, hard_details = [], []
    smooth_cache = []
    for row in corrected:
        response = simulate(sim, row)
        miss, details = hard_miss(response, sim)
        hard.append(miss)
        hard_details.append(details)
        smooth_cache.append(response)
    hard = np.asarray(hard)
    combos = []
    for tau in (0.10, 0.25, 0.50):
        for beta in (0.25, 0.50):
            for tau_w in (0.25, 0.50):
                values = [smooth_terms(resp, sim, tau, beta, tau_w)["loss"] for resp in smooth_cache]
                combos.append(
                    {
                        "tau": tau, "beta": beta, "tau_w": tau_w,
                        "pearson": pearson(values, hard),
                        "spearman": spearman(values, hard),
                        "loss_mean": float(np.mean(values)),
                        "loss_median": float(np.median(values)),
                    }
                )
    combos.sort(key=lambda row: row["spearman"], reverse=True)
    best = combos[0]
    payload = {
        "base_checkpoint": str(ROOT / "runs/route3_modes/r3_m32_s2026/best.pt"),
        "checkpoint_info": info,
        "n_states": int(n_states),
        "seeds": seeds[idx].tolist(),
        "base_hard_miss": {
            "mean": float(hard.mean()),
            "median": float(np.median(hard)),
            "p90": float(np.quantile(hard, 0.90)),
            "worst": float(hard.max()),
            "pass_rate": float(np.mean([d["all_specs_pass"] for d in hard_details])),
        },
        "grid": combos,
        "best": best,
        "simulator_calls": int(SIM_CALLS),
    }
    save_json(OUT / "smooth_calibration.json", payload)
    log(f"calibration best {best}")
    return payload


def selected_state_indices(count: int, start: int = 7000):
    return np.arange(start, start + count, dtype=np.int64)


def calibration_params():
    path = OUT / "smooth_calibration.json"
    if not path.exists():
        raise RuntimeError("run smooth calibration first")
    best = json.loads(path.read_text())["best"]
    return float(best["tau"]), float(best["beta"]), float(best["tau_w"])


def stage_coordinate(n_states=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sim, cfg, info = load_base(
        str(ROOT / "configs/route_delay_fno_r3_m32.json"),
        str(ROOT / "runs/route3_modes/r3_m32_s2026/best.pt"),
        device,
    )
    responses, m_current, seeds = load_cached("train", 8000)
    idx = selected_state_indices(n_states)
    m = m_current[idx]
    delta = base_corrections(model, responses[idx], device)
    half = 0.5 * (np.asarray(sim.upper_bounds) - np.asarray(sim.lower_bounds))
    y0 = delta / half
    tau, beta, tau_w = calibration_params()
    eps_values = (0.005, 0.010, 0.020)
    gradients = {}
    for eps in eps_values:
        gradients[eps] = np.zeros((n_states, 41), dtype=np.float64)
        for row in range(n_states):
            for j in range(41):
                yp = y0[row].copy()
                ym = y0[row].copy()
                yp[j] += eps
                ym[j] -= eps
                lp = smooth_terms(simulate(sim, m[row] + half * yp), sim, tau, beta, tau_w)["loss"]
                lm = smooth_terms(simulate(sim, m[row] + half * ym), sim, tau, beta, tau_w)["loss"]
                gradients[eps][row, j] = (lp - lm) / (2.0 * eps)
        log(f"coordinate eps {eps}: grad norm mean {np.linalg.norm(gradients[eps], axis=1).mean():.4f}")
    stability = {}
    for a, b in ((0.005, 0.010), (0.010, 0.020), (0.005, 0.020)):
        cos = []
        for i in range(n_states):
            x, y = gradients[a][i], gradients[b][i]
            cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
        stability[f"{a}_{b}"] = {
            "cosine_median": float(np.median(cos)),
            "cosine_p10": float(np.quantile(cos, 0.10)),
        }
    payload = {
        "n_states": int(n_states),
        "seeds": seeds[idx].tolist(),
        "tau_beta_tau_w": [tau, beta, tau_w],
        "eps_values": list(eps_values),
        "gradient_norm_mean": {
            str(eps): float(np.linalg.norm(gradients[eps], axis=1).mean()) for eps in eps_values
        },
        "gradient_norm_p90": {
            str(eps): float(np.quantile(np.linalg.norm(gradients[eps], axis=1), 0.90)) for eps in eps_values
        },
        "stability": stability,
        "simulator_calls": int(SIM_CALLS),
    }
    np.savez_compressed(
        OUT / "coordinate_gradients.npz",
        m=m, delta=delta, y0=y0, half=half,
        g005=gradients[0.005], g010=gradients[0.010], g020=gradients[0.020],
    )
    save_json(OUT / "coordinate_gradient.json", payload)
    log(f"coordinate stability {stability}")
    return payload



def stage_oracle():
    sim = load_simulator()
    data = np.load(OUT / "coordinate_gradients.npz")
    m, delta, y0, half, grad = data["m"], data["delta"], data["y0"], data["half"], data["g010"]
    n = len(m)
    base_miss = []
    for row in range(n):
        response = simulate(sim, m[row] + delta[row])
        base_miss.append(hard_miss(response, sim)[0])
    base_miss = np.asarray(base_miss)
    payload = {
        "base": {
            "mean": float(base_miss.mean()),
            "median": float(np.median(base_miss)),
            "p90": float(np.quantile(base_miss, 0.90)),
        },
        "eta": {},
    }
    for eta in (0.05, 0.10, 0.20, 0.50):
        misses = []
        for row in range(n):
            y_new = y0[row] - eta * grad[row]
            response = simulate(sim, m[row] + half * y_new)
            misses.append(hard_miss(response, sim)[0])
        misses = np.asarray(misses)
        payload["eta"][str(eta)] = {
            "mean": float(misses.mean()),
            "median": float(np.median(misses)),
            "p90": float(np.quantile(misses, 0.90)),
            "improved_fraction": float(np.mean(misses < base_miss)),
        }
        log(f"oracle eta {eta}: mean {misses.mean():.2f} dB vs base {base_miss.mean():.2f}")
    best = min(payload["eta"], key=lambda k: payload["eta"][k]["mean"])
    payload["best_eta"] = best
    payload["simulator_calls"] = int(SIM_CALLS)
    save_json(OUT / "oracle.json", payload)
    return payload


def eval_pool(pool, sim, m, delta, chunk=8):
    candidates = [np.clip(m[i] + delta[i], sim.lower_bounds, sim.upper_bounds) for i in range(len(m))]
    responses = pool.map(_sim_worker, candidates, chunksize=chunk)
    global SIM_CALLS
    SIM_CALLS += len(candidates)
    misses, passes = [], []
    for response in responses:
        miss, details = hard_miss(response, sim)
        misses.append(miss)
        passes.append(bool(details["all_specs_pass"]))
    misses = np.asarray(misses)
    return {
        "mean": float(misses.mean()),
        "median": float(np.median(misses)),
        "p90": float(np.quantile(misses, 0.90)),
        "worst": float(misses.max()),
        "pass_rate": float(np.mean(passes)),
        "pass_count": int(np.sum(passes)),
    }


def stage_train(steps=200, batch_size=16, k_dirs=16, eps=0.01, lr=1e-4, workers=8, full=False):
    global SIM_CALLS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = mp.get_context("fork").Pool(processes=int(workers), initializer=_init_sim_worker)
    try:
        model, sim, cfg, info = load_base(
            str(ROOT / "configs/route_delay_fno_r3_m32.json"),
            str(ROOT / "runs/route3_modes/r3_m32_s2026/best.pt"),
            device,
        )
        responses, m_all, seeds = load_cached("train", 8000)
        if full:
            m_train = m_all
            responses_train = responses
            eval_responses, eval_m, eval_seeds = load_cached("validation", 1000)
        else:
            m_train = m_all[:7000]
            responses_train = responses[:7000]
            eval_responses = responses[7000:8000]
            eval_m = m_all[7000:8000]
            eval_seeds = seeds[7000:8000]
        features_train = extract_features(model, responses_train, device)
        eval_features = extract_features(model, eval_responses, device)
        base_delta_train = base_corrections(model, responses_train, device)
        base_delta_eval = base_corrections(model, eval_responses, device)
        half = 0.5 * (np.asarray(sim.upper_bounds) - np.asarray(sim.lower_bounds))
        lower = np.asarray(sim.lower_bounds, dtype=np.float64)
        upper = np.asarray(sim.upper_bounds, dtype=np.float64)
        tau, beta, tau_w = calibration_params()
        base_eval = eval_pool(pool, sim, eval_m, base_delta_eval)
        log(f"base {'validation' if full else 'tune'}: {base_eval}")
        head = ResidualHead(features_train.shape[1]).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=1e-5)
        rng = np.random.default_rng(2026)
        history = []

        for step in range(1, int(steps) + 1):
            idx = rng.choice(len(m_train), size=min(int(batch_size), len(m_train)), replace=False)
            f = features_train[idx].to(device)
            m_batch = m_train[idx]
            y_base = base_delta_train[idx] / half[None, :]
            residual = head(f)
            y = torch.as_tensor(y_base, dtype=torch.float32, device=device) + residual
            v = rng.choice([-1.0, 1.0], size=(len(idx), int(k_dirs), 41))
            y_np = y.detach().cpu().numpy()
            candidates = []
            for i in range(len(idx)):
                for k in range(int(k_dirs)):
                    for sign in (1.0, -1.0):
                        candidates.append(
                            np.clip(
                                m_batch[i] + half * (y_np[i] + sign * float(eps) * v[i, k]),
                                lower, upper,
                            )
                        )
            responses_corr = pool.map(_sim_worker, candidates, chunksize=8)
            SIM_CALLS += len(candidates)
            losses = np.asarray(
                [smooth_terms(resp, sim, tau, beta, tau_w)["loss"] for resp in responses_corr],
                dtype=np.float64,
            ).reshape(len(idx), int(k_dirs), 2)
            grad = np.mean(
                ((losses[:, :, 0] - losses[:, :, 1]) / (2.0 * float(eps)))[..., None] * v,
                axis=1,
            )
            grad_t = torch.as_tensor(grad, dtype=torch.float32, device=device)
            pseudo = (y * grad_t).sum(dim=1).mean()
            raw = torch.as_tensor(m_batch, dtype=torch.float32, device=device) + torch.as_tensor(
                half, dtype=torch.float32, device=device
            )[None, :] * y
            low_t = torch.as_tensor(lower, dtype=torch.float32, device=device)
            up_t = torch.as_tensor(upper, dtype=torch.float32, device=device)
            bound = (
                torch.relu(low_t[None, :] - raw) ** 2
                + torch.relu(raw - up_t[None, :]) ** 2
            ).mean()
            trust = residual.pow(2).mean()
            loss = pseudo + 1e-3 * bound + 1e-3 * trust
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "pseudo": float(pseudo.detach()),
                "bound": float(bound.detach()),
                "trust": float(trust.detach()),
                "grad_norm": float(np.linalg.norm(grad, axis=1).mean()),
                "residual_norm": float(torch.linalg.vector_norm(residual.detach(), dim=1).mean()),
            })
            if step % 10 == 0:
                row = history[-1]
                log(f"step {step}: loss {row['loss']:.3f}, grad {row['grad_norm']:.3f}, res {row['residual_norm']:.4f}")
            if step in (50, 100, 150, 200):
                with torch.no_grad():
                    res_sub = head(eval_features[:64].to(device)).cpu().numpy()
                delta_sub = base_delta_eval[:64] + half[None, :] * res_sub
                eval_sub = eval_pool(pool, sim, eval_m[:64], delta_sub)
                log(f"step {step} eval64: {eval_sub}")
        with torch.no_grad():
            residual_eval = head(eval_features.to(device)).cpu().numpy()
        delta_eval = base_delta_eval + half[None, :] * residual_eval
        refined_eval = eval_pool(pool, sim, eval_m, delta_eval)
        torch.save(head.state_dict(), OUT / ("residual_head_full.pt" if full else "residual_head.pt"))
        payload = {
            "steps": int(steps),
            "batch_size": int(batch_size),
            "k_dirs": int(k_dirs),
            "eps": float(eps),
            "lr": float(lr),
            "lambda_bound": 1e-3,
            "lambda_trust": 1e-3,
            "smooth_params": [tau, beta, tau_w],
            "eval_split": "validation" if full else "tune",
            "base_eval": base_eval,
            "refined_eval": refined_eval,
            "residual_stats": {
                "mean_abs": float(np.abs(residual_eval).mean()),
                "p90_abs": float(np.quantile(np.abs(residual_eval), 0.90)),
                "signed_mean": residual_eval.mean(axis=0).tolist(),
            },
            "history": history,
            "simulator_calls": int(SIM_CALLS),
        }
        save_json(OUT / ("residual_training_full.json" if full else "residual_training.json"), payload)
        log(f"refined {'validation' if full else 'tune'}: {refined_eval}")
        return payload
    finally:
        pool.close()
        pool.join()


def eval_pool_details(pool, sim, m, delta, chunk=8):
    candidates = [np.clip(m[i] + delta[i], sim.lower_bounds, sim.upper_bounds) for i in range(len(m))]
    responses = pool.map(_sim_worker, candidates, chunksize=chunk)
    global SIM_CALLS
    SIM_CALLS += len(candidates)
    misses, passes = [], []
    per_req = {}
    for response in responses:
        details = sim.check_response(response)
        misses.append(float(details["max_spec_miss_db"]))
        passes.append(bool(details["all_specs_pass"]))
        for req in sim.requirements:
            row = per_req.setdefault(req.name, {"pass": 0, "miss": [], "trace": str(req.trace)})
            row["pass"] += int(bool(details[f"{req.name}_pass"]))
            row["miss"].append(float(details[f"{req.name}_miss_db"]))
    misses = np.asarray(misses)
    return {
        "mean": float(misses.mean()),
        "median": float(np.median(misses)),
        "p90": float(np.quantile(misses, 0.90)),
        "worst": float(misses.max()),
        "pass_count": int(np.sum(passes)),
        "pass_rate": float(np.mean(passes)),
        "per_requirement": {
            name: {
                "pass_rate": value["pass"] / len(m),
                "mean_miss": float(np.mean(value["miss"])),
                "trace": value["trace"],
            }
            for name, value in per_req.items()
        },
    }


def stage_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = mp.get_context("fork").Pool(processes=8, initializer=_init_sim_worker)
    try:
        model, sim, cfg, info = load_base(
            str(ROOT / "configs/route_delay_fno_r3_m32.json"),
            str(ROOT / "runs/route3_modes/r3_m32_s2026/best.pt"),
            device,
        )
        responses, m_all, seeds = load_cached("train", 8000)
        m_tune = m_all[7000:8000]
        responses_tune = responses[7000:8000]
        features = extract_features(model, responses_tune, device)
        base_delta = base_corrections(model, responses_tune, device)
        result = json.loads((OUT / "residual_training.json").read_text())
        head = ResidualHead(features.shape[1]).to(device)
        head.load_state_dict(torch.load(OUT / "residual_head.pt", map_location=device))
        head.eval()
        with torch.no_grad():
            residual = head(features.to(device)).cpu().numpy()
        refined_delta = base_delta + 0.5 * (
            np.asarray(sim.upper_bounds) - np.asarray(sim.lower_bounds)
        )[None, :] * residual
        base = eval_pool_details(pool, sim, m_tune, base_delta)
        refined = eval_pool_details(pool, sim, m_tune, refined_delta)
        grad = np.load(OUT / "coordinate_gradients.npz")["g010"]
        top = np.argsort(-np.abs(grad).mean(axis=0))[:10]
        payload = {
            "base": base,
            "refined": refined,
            "residual_stats": result["residual_stats"],
            "gradient_top10": [
                {"parameter": int(i), "mean_abs_gradient": float(np.abs(grad[:, i]).mean())}
                for i in top
            ],
            "simulator_calls": int(SIM_CALLS),
        }
        save_json(OUT / "analysis.json", payload)
        log(f"analysis refined: {refined['mean']:.2f} dB")
        return payload
    finally:
        pool.close()
        pool.join()


def stage_analysis_full():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = mp.get_context("fork").Pool(processes=8, initializer=_init_sim_worker)
    try:
        model, sim, cfg, info = load_base(
            str(ROOT / "configs/route_delay_fno_r3_m32.json"),
            str(ROOT / "runs/route3_modes/r3_m32_s2026/best.pt"),
            device,
        )
        responses, m_val, seeds = load_cached("validation", 1000)
        features = extract_features(model, responses, device)
        base_delta = base_corrections(model, responses, device)
        result = json.loads((OUT / "residual_training_full.json").read_text())
        head = ResidualHead(features.shape[1]).to(device)
        head.load_state_dict(torch.load(OUT / "residual_head_full.pt", map_location=device))
        head.eval()
        with torch.no_grad():
            residual = head(features.to(device)).cpu().numpy()
        refined_delta = base_delta + 0.5 * (
            np.asarray(sim.upper_bounds) - np.asarray(sim.lower_bounds)
        )[None, :] * residual
        base = eval_pool_details(pool, sim, m_val, base_delta)
        refined = eval_pool_details(pool, sim, m_val, refined_delta)
        grad = np.load(OUT / "coordinate_gradients.npz")["g010"]
        top = np.argsort(-np.abs(grad).mean(axis=0))[:10]
        payload = {
            "split": "validation",
            "base": base,
            "refined": refined,
            "residual_stats": result["residual_stats"],
            "gradient_top10": [
                {"parameter": int(i), "mean_abs_gradient": float(np.abs(grad[:, i]).mean())}
                for i in top
            ],
            "simulator_calls": int(SIM_CALLS),
        }
        save_json(OUT / "analysis_full.json", payload)
        log(f"full analysis refined: {refined['mean']:.2f} dB, pass {refined['pass_count']}")
        return payload
    finally:
        pool.close()
        pool.join()


def stage_spsa():
    sim = load_simulator()
    data = np.load(OUT / "coordinate_gradients.npz")
    m, y0, half, grad = data["m"], data["y0"], data["half"], data["g010"]
    n = len(m)
    tau, beta, tau_w = calibration_params()
    eps = 0.010
    rng = np.random.default_rng(2026)
    payload = {"eps": eps, "K": {}}
    for k in (4, 8, 16):
        cosines, sign_agreements, top5 = [], [], []
        for row in range(n):
            estimate = np.zeros(41, dtype=np.float64)
            for _ in range(k):
                v = rng.choice([-1.0, 1.0], size=41)
                lp = smooth_terms(simulate(sim, m[row] + half * (y0[row] + eps * v)), sim, tau, beta, tau_w)["loss"]
                lm = smooth_terms(simulate(sim, m[row] + half * (y0[row] - eps * v)), sim, tau, beta, tau_w)["loss"]
                estimate += (lp - lm) / (2.0 * eps) * v
            estimate /= k
            g = grad[row]
            cosines.append(float(estimate @ g / (np.linalg.norm(estimate) * np.linalg.norm(g) + 1e-12)))
            sign_agreements.append(float(np.mean(np.sign(estimate) == np.sign(g))))
            top5.append(len(set(np.argsort(-np.abs(estimate))[:5]) & set(np.argsort(-np.abs(g))[:5])) / 5.0)
        payload["K"][str(k)] = {
            "cosine_median": float(np.median(cosines)),
            "cosine_p10": float(np.quantile(cosines, 0.10)),
            "sign_agreement_median": float(np.median(sign_agreements)),
            "top5_overlap_median": float(np.median(top5)),
        }
        log(f"spsa K={k}: cos {np.median(cosines):.3f}, sign {np.median(sign_agreements):.3f}")
    payload["simulator_calls"] = int(SIM_CALLS)
    save_json(OUT / "spsa_fidelity.json", payload)
    return payload



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibration", "coordinate", "oracle", "spsa", "train", "trainfull", "analysis", "analysisfull", "all"))
    args = parser.parse_args()
    if args.stage in ("calibration", "all"):
        stage_calibration()
    if args.stage in ("coordinate", "all"):
        stage_coordinate()
    if args.stage in ("oracle", "all"):
        stage_oracle()
    if args.stage in ("spsa", "all"):
        stage_spsa()
    if args.stage in ("train", "all"):
        stage_train()
    if args.stage in ("trainfull", "all"):
        stage_train(full=True)
    if args.stage in ("analysis", "all"):
        stage_analysis()
    if args.stage in ("analysisfull", "all"):
        stage_analysis_full()


if __name__ == "__main__":
    main()
