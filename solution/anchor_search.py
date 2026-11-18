"""Black-box search for a compliant anchor state ``m*``.

Only the public interface is used: ``simulate`` / ``check_response`` plus the
published bounds and requirement table.  The optimizer is a CEM in the
normalized box ``[-1, 1]^41`` followed by a local (1+lambda) refinement; the
objective is the minimum signed margin over the seven requirements.

Sign convention (positive = compliant):

    max S11 <= T   ->  margin = T - max S11
    min S21 >= T   ->  margin = min S21 - T
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time

import numpy as np

from .simulator import load_simulator, signed_margins
from .utils import resolve, save_json

_WORKER_SIM = None
_WORKER_CENTER = None
_WORKER_HALF = None


def _worker_init() -> None:
    global _WORKER_SIM, _WORKER_CENTER, _WORKER_HALF
    _WORKER_SIM = load_simulator()
    _WORKER_CENTER = 0.5 * (_WORKER_SIM.lower_bounds + _WORKER_SIM.upper_bounds)
    _WORKER_HALF = 0.5 * (_WORKER_SIM.upper_bounds - _WORKER_SIM.lower_bounds)


def _worker_eval(u_batch: np.ndarray) -> list[np.ndarray]:
    outputs = []
    for u in np.asarray(u_batch, dtype=np.float64):
        vector = np.clip(
            _WORKER_CENTER + _WORKER_HALF * u,
            _WORKER_SIM.lower_bounds,
            _WORKER_SIM.upper_bounds,
        )
        response = _WORKER_SIM.simulate(vector)
        outputs.append(signed_margins(_WORKER_SIM, response))
    return outputs


def softmin(margins: np.ndarray, tau: float) -> float:
    """Numerically stable smooth minimum (higher is better)."""
    z = -np.asarray(margins, dtype=np.float64) / float(tau)
    z_max = float(np.max(z))
    return float(-float(tau) * (z_max + np.log(np.mean(np.exp(z - z_max)))))


def objective_value(margins: np.ndarray, kind: str = "softmin", tau: float = 0.5) -> float:
    key = str(kind).lower()
    if key == "min":
        return float(np.min(margins))
    if key == "softmin":
        return softmin(margins, tau)
    if key == "mean":
        return float(np.mean(margins))
    raise ValueError(f"Unknown objective '{kind}'")


def anchor_metadata(path: str) -> dict:
    with np.load(resolve(path), allow_pickle=False) as archive:
        return {
            "m_star_shape": tuple(int(v) for v in archive["m_star"].shape),
            "min_margin_db": float(archive["min_margin"]),
            "margins_db": archive["margins"].astype(float).tolist(),
            "compliant": bool(float(archive["min_margin"]) > 0.0),
            "simulator_calls": int(archive["simulator_calls"]),
            "wall_time_seconds": float(archive["wall_time_seconds"]),
        }


class CEMAnchorSearch:
    """Cross-entropy method + local refinement in normalized parameter space."""

    def __init__(self, cfg: dict, output_path: str | None = None) -> None:
        self.cfg = dict(cfg)
        self.simulator = load_simulator()
        self.lower = np.asarray(self.simulator.lower_bounds, dtype=np.float64)
        self.upper = np.asarray(self.simulator.upper_bounds, dtype=np.float64)
        self.center = 0.5 * (self.lower + self.upper)
        self.half = np.maximum(0.5 * (self.upper - self.lower), 1e-6)
        self.dim = int(self.lower.size)
        self.names = [req.name for req in self.simulator.requirements]
        self.output_path = resolve(
            output_path or self.cfg.get("output", "artifacts/common/anchor.npz")
        )
        self.calls = 0
        self.history: list[dict] = []
        self._pool = None

    # -- plumbing --------------------------------------------------------
    def _start_pool(self) -> None:
        global _WORKER_SIM, _WORKER_CENTER, _WORKER_HALF
        workers = int(self.cfg.get("workers", 8))
        if workers > 1:
            self._pool = mp.get_context("fork").Pool(
                processes=workers, initializer=_worker_init
            )
        else:
            _WORKER_SIM = self.simulator
            _WORKER_CENTER = self.center
            _WORKER_HALF = self.half

    def _stop_pool(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def _evaluate(self, u_batch: np.ndarray) -> np.ndarray:
        """Return signed margins shaped (N, 7) for normalized candidates."""
        u_batch = np.asarray(u_batch, dtype=np.float64)
        if u_batch.ndim == 1:
            u_batch = u_batch[None, :]
        self.calls += int(len(u_batch))
        if self._pool is None:
            results = _worker_eval(u_batch)
        else:
            chunks = max(1, int(self.cfg.get("workers", 8)) * 2)
            pieces = np.array_split(u_batch, min(chunks, len(u_batch)))
            results = []
            for piece in self._pool.map(_worker_eval, [p for p in pieces if len(p)]):
                results.extend(piece)
        return np.stack(results, axis=0)

    def to_vector(self, u: np.ndarray) -> np.ndarray:
        return np.clip(
            self.center + self.half * np.asarray(u, dtype=np.float64),
            self.lower,
            self.upper,
        )

    # -- search ----------------------------------------------------------
    def run(self, init_u: np.ndarray | None = None) -> dict:
        cfg = self.cfg
        rng = np.random.default_rng(int(cfg.get("seed", 2026)))
        population = int(cfg.get("population", 256))
        elite_frac = float(cfg.get("elite_frac", 0.08))
        n_elite = max(4, int(round(population * elite_frac)))
        iterations = int(cfg.get("iterations", 150))
        smoothing = float(cfg.get("smoothing", 0.7))
        min_std = float(cfg.get("min_std", 0.01))
        extra_std = float(cfg.get("extra_std", 0.0))
        objective = str(cfg.get("objective", "softmin"))
        tau = float(cfg.get("softmin_tau", 0.5))
        early_margin = float(cfg.get("early_stop_margin", 0.5))
        log_every = int(cfg.get("log_every", 5))

        mean = np.zeros(self.dim, dtype=np.float64)
        if init_u is not None:
            mean = np.clip(np.asarray(init_u, dtype=np.float64), -1.0, 1.0)
        std = np.full(self.dim, float(cfg.get("init_std", 0.35)), dtype=np.float64)

        start = time.perf_counter()
        self._start_pool()
        try:
            best_u = mean.copy()
            best_margins = self._evaluate(best_u[None, :])[0]
            best_min = float(np.min(best_margins))
            best_m = self.to_vector(best_u)
            for iteration in range(iterations):
                samples = rng.normal(mean, std, size=(population, self.dim))
                samples = np.clip(samples, -1.0, 1.0)
                margins = self._evaluate(samples)
                scores = np.array(
                    [objective_value(row, objective, tau) for row in margins]
                )
                order = np.argsort(-scores)
                elite = samples[order[:n_elite]]
                candidate_margins = margins[order[0]]
                candidate_min = float(np.min(candidate_margins))
                if candidate_min > best_min:
                    best_min = candidate_min
                    best_u = samples[order[0]].copy()
                    best_margins = candidate_margins.copy()
                    best_m = self.to_vector(best_u)

                mean = smoothing * mean + (1.0 - smoothing) * elite.mean(axis=0)
                std = smoothing * std + (1.0 - smoothing) * elite.std(axis=0)
                std = np.maximum(std, min_std) + extra_std
                record = {
                    "iteration": iteration + 1,
                    "population_best_min_margin_db": candidate_min,
                    "best_min_margin_db": best_min,
                    "best_margins_db": [float(value) for value in best_margins],
                    "population_objective_best": float(scores[order[0]]),
                    "population_objective_mean": float(scores.mean()),
                    "mean_std": float(std.mean()),
                    "elapsed_seconds": float(time.perf_counter() - start),
                    "simulator_calls": int(self.calls),
                }
                self.history.append(record)
                if log_every and (iteration + 1) % log_every == 0:
                    print(
                        f"[anchor] iter {iteration + 1}/{iterations} "
                        f"best_min={best_min:+.4f} dB "
                        f"pop_best={candidate_min:+.4f} dB calls={self.calls}",
                        flush=True,
                    )
                if best_min >= early_margin:
                    print(
                        f"[anchor] early stop at margin {best_min:.4f} dB "
                        f"(target {early_margin})",
                        flush=True,
                    )
                    break

            best_u, best_m, best_margins, best_min = self._refine(
                best_u, best_m, best_margins, best_min, rng
            )
        finally:
            self._stop_pool()

        elapsed = float(time.perf_counter() - start)
        return self._save(best_u, best_m, best_margins, best_min, elapsed)

    def _refine(self, best_u, best_m, best_margins, best_min, rng):
        cfg = self.cfg
        steps = int(cfg.get("local_refine_steps", 0))
        if steps <= 0:
            return best_u, best_m, best_margins, best_min
        sigma0 = float(cfg.get("local_refine_sigma", 0.02))
        candidates = max(8, int(cfg.get("local_refine_population", 32)))
        for step in range(steps):
            sigma = sigma0 * (0.1 ** (step / max(1, steps - 1)))
            samples = np.clip(
                best_u + rng.normal(0.0, sigma, size=(candidates, self.dim)), -1.0, 1.0
            )
            margins = self._evaluate(samples)
            mins = margins.min(axis=1)
            idx = int(np.argmax(mins))
            if float(mins[idx]) > best_min:
                best_min = float(mins[idx])
                best_u = samples[idx].copy()
                best_margins = margins[idx].copy()
                best_m = self.to_vector(best_u)
        print(
            f"[anchor] local refinement done: min margin {best_min:+.4f} dB "
            f"(calls={self.calls})",
            flush=True,
        )
        return best_u, best_m, best_margins, best_min

    def _save(self, best_u, best_m, best_margins, best_min, elapsed):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls += 1  # final independent verification simulation
        details = self.simulator.check_response(self.simulator.simulate(best_m))
        np.savez(
            self.output_path,
            m_star=np.asarray(best_m, dtype=np.float64),
            u_star=np.asarray(best_u, dtype=np.float64),
            margins=np.asarray(best_margins, dtype=np.float64),
            min_margin=np.asarray(best_min, dtype=np.float64),
            search_seed=np.asarray(int(self.cfg.get("seed", 2026))),
            simulator_calls=np.asarray(int(self.calls)),
            wall_time_seconds=np.asarray(float(elapsed)),
            requirement_names=np.asarray(self.names),
            config_json=np.asarray(json.dumps(self.cfg, sort_keys=True)),
            iteration=np.asarray(int(len(self.history))),
        )
        summary_path = self.output_path.with_suffix(".json")
        save_json(
            summary_path,
            {
                "anchor_path": str(self.output_path),
                "compliant": bool(best_min > 0.0),
                "min_margin_db": float(best_min),
                "margins_db": {
                    name: float(value)
                    for name, value in zip(self.names, best_margins)
                },
                "all_specs_pass": bool(details["all_specs_pass"]),
                "max_spec_miss_db": float(details["max_spec_miss_db"]),
                "simulator_calls": int(self.calls),
                "wall_time_seconds": float(elapsed),
                "iterations": len(self.history),
                "config": self.cfg,
                "history": self.history,
            },
        )
        return {
            "anchor_path": str(self.output_path),
            "compliant": bool(best_min > 0.0),
            "min_margin_db": float(best_min),
            "margins_db": {
                name: float(value) for name, value in zip(self.names, best_margins)
            },
            "simulator_calls": int(self.calls),
            "wall_time_seconds": float(elapsed),
        }
