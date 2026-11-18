"""One-shot evaluation exactly following the assignment protocol.

The evaluator is the *only* component that sees ``m_current``; the model gets a
raw response tensor and nothing else.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .losses import aggregate_metric_dicts, state_error_metrics
from .simulator import requirement_infos
from .utils import PUBLIC_EVAL_SEEDS, get_device


def assert_not_public_seeds(seeds) -> None:
    overlap = np.intersect1d(
        np.asarray(list(seeds), dtype=np.int64), np.asarray(PUBLIC_EVAL_SEEDS)
    )
    if overlap.size:
        raise ValueError(f"Public evaluation seeds are forbidden here: {overlap[:5]}")


@torch.no_grad()
def evaluate_one_shot(
    model,
    simulator,
    seeds,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    progress: bool = False,
) -> tuple[dict, list[dict]]:
    """Fraction of states satisfying all seven requirements after one correction."""
    seed_list = [int(seed) for seed in seeds]
    assert_not_public_seeds(seed_list)
    device = get_device(device) if isinstance(device, str) else device
    model = model.to(device).eval()
    requirements = requirement_infos(simulator)
    lower = np.asarray(simulator.lower_bounds)
    upper = np.asarray(simulator.upper_bounds)

    rows: list[dict] = []
    start = time.perf_counter()
    for begin in range(0, len(seed_list), int(batch_size)):
        chunk = seed_list[begin:begin + int(batch_size)]
        states = [simulator.sample_state(seed) for seed in chunk]
        current = np.stack([state[0] for state in states], axis=0)
        responses = np.stack([state[1] for state in states], axis=0)
        delta = model(torch.from_numpy(responses).to(device)).cpu().numpy()
        for idx, seed in enumerate(chunk):
            delta_row = np.asarray(delta[idx], dtype=np.float64)
            finite = bool(np.isfinite(delta_row).all()) and delta_row.shape == (41,)
            if not finite:
                rows.append(
                    {
                        "seed": seed,
                        "passed": False,
                        "max_miss_db": float("inf"),
                        "correction_l2": float("inf"),
                        "correction_max_abs": float("inf"),
                        "clipped_entry_rate": 1.0,
                        "requirements": {},
                    }
                )
                continue
            result = simulator.apply_correction(current[idx], delta_row)
            raw = current[idx] + delta_row
            clipped = np.clip(raw, lower, upper)
            details = result.details
            per_requirement = {}
            for req in requirements:
                per_requirement[req.name] = {
                    "value_db": float(details[f"{req.name}_value_db"]),
                    "pass": bool(details[f"{req.name}_pass"]),
                    "miss_db": float(details[f"{req.name}_miss_db"]),
                }
            rows.append(
                {
                    "seed": seed,
                    "passed": bool(result.all_specs_pass),
                    "max_miss_db": float(result.max_spec_miss_db),
                    "correction_l2": float(np.linalg.norm(delta_row)),
                    "correction_max_abs": float(np.max(np.abs(delta_row))),
                    "clipped_entry_rate": float(np.mean(clipped != raw)),
                    "requirements": per_requirement,
                }
            )
        if progress:
            print(f"[eval] {len(rows)}/{len(seed_list)} states", flush=True)

    elapsed = time.perf_counter() - start
    misses = np.asarray([row["max_miss_db"] for row in rows], dtype=np.float64)
    l2 = np.asarray([row["correction_l2"] for row in rows], dtype=np.float64)
    clipped = np.asarray([row["clipped_entry_rate"] for row in rows], dtype=np.float64)
    per_requirement = {}
    for req in requirements:
        values = [row["requirements"].get(req.name) for row in rows]
        values = [value for value in values if value is not None]
        per_requirement[req.name] = {
            "pass_rate": float(np.mean([value["pass"] for value in values])) if values else 0.0,
            "mean_miss_db": float(np.mean([value["miss_db"] for value in values])) if values else float("inf"),
            "max_miss_db": float(np.max([value["miss_db"] for value in values])) if values else float("inf"),
            "mean_value_db": float(np.mean([value["value_db"] for value in values])) if values else float("nan"),
        }
    summary = {
        "num_states": len(rows),
        "pass_count": int(sum(row["passed"] for row in rows)),
        "pass_rate": float(np.mean([row["passed"] for row in rows])) if rows else 0.0,
        "mean_max_miss_db": float(np.mean(misses)) if rows else float("inf"),
        "median_max_miss_db": float(np.median(misses)) if rows else float("inf"),
        "p90_max_miss_db": float(np.quantile(misses, 0.90)) if rows else float("inf"),
        "worst_max_miss_db": float(np.max(misses)) if rows else float("inf"),
        "correction_l2_mean": float(np.mean(l2)) if rows else float("inf"),
        "correction_l2_median": float(np.median(l2)) if rows else float("inf"),
        "correction_l2_std": float(np.std(l2)) if rows else float("inf"),
        "correction_l2_max": float(np.max(l2)) if rows else float("inf"),
        "clipped_entry_rate_mean": float(np.mean(clipped)) if rows else 1.0,
        "per_requirement": per_requirement,
        "wall_time_seconds": float(elapsed),
        "public_eval_seeds_used": False,
    }
    return summary, rows


@torch.no_grad()
def evaluate_identification(
    model,
    dataset,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> dict:
    """State-identification error on a cached split (no simulator needed)."""
    device = get_device(device) if isinstance(device, str) else device
    model = model.to(device).eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False
    )
    predictions = []
    targets = []
    for response, m_current in loader:
        response = response.to(device)
        _, pred_m, _ = model.forward_state(response)
        predictions.append(pred_m.detach().float().cpu())
        targets.append(m_current.float().cpu())
    aggregate = state_error_metrics(torch.cat(predictions), torch.cat(targets))
    aggregate["per_parameter_mae"] = np.asarray(
        aggregate["per_parameter_mae"]
    ).tolist()
    aggregate["per_parameter_p95"] = np.asarray(
        aggregate["per_parameter_p95"]
    ).tolist()
    aggregate["num_states"] = int(len(dataset))
    return aggregate


def save_response_plot(simulator, response: np.ndarray, title: str, path: str) -> str:
    """Save S11/S21 dB curves with the seven requirement bands highlighted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s11_db, s21_db = simulator.response_db(np.asarray(response))
    frequency_mhz = np.asarray(simulator.frequency_hz) / 1e6
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(frequency_mhz, s11_db, label="S11", linewidth=1.1)
    ax.plot(frequency_mhz, s21_db, label="S21", linewidth=1.1)
    for req in simulator.requirements:
        color = "tab:blue" if str(req.trace).upper() == "S11" else "tab:orange"
        ax.hlines(
            float(req.threshold_db),
            float(req.start_mhz),
            float(req.stop_mhz),
            color=color,
            linewidth=4,
            alpha=0.3,
        )
    ax.set(xlabel="Frequency (MHz)", ylabel="Magnitude (dB)", title=title)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    from .utils import resolve
    from pathlib import Path

    target = Path(resolve(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return str(target)
