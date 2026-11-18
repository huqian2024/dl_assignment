"""Per-state compliant teacher construction for CM19.

For an existing state ``m_i`` we search a compliant parameter vector close to
it, instead of forcing every state to the single global anchor.  All simulator
calls go through the public ``simulate`` / ``check_response`` API; no new
states are sampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulator import requirement_infos, signed_margins


@dataclass
class TeacherResult:
    delta: np.ndarray
    corrected: np.ndarray
    margins: np.ndarray
    min_margin: float
    passed: bool
    calls: int
    iterations: int
    history: list[dict]


def margins_of(simulator, m_vector: np.ndarray) -> np.ndarray:
    return signed_margins(
        simulator, simulator.simulate(np.asarray(m_vector, dtype=np.float64))
    )


def line_search_anchor(
    simulator,
    m_current: np.ndarray,
    m_anchor: np.ndarray,
    required_margin: float = 0.0,
    grid_step: float = 0.05,
    refine_steps: int = 6,
) -> dict:
    """Find the compliant point on the segment current -> anchor closest to current.

    The anchor (t=1) is expected to be compliant; we walk from t=1 towards t=0
    while margins stay >= required_margin.  The returned ``t`` is the smallest t
    of that connected compliant segment, optionally refined by bisection.
    """
    m_current = np.asarray(m_current, dtype=np.float64)
    m_anchor = np.asarray(m_anchor, dtype=np.float64)
    grid = np.arange(0.0, 1.0 + 1e-9, grid_step)
    calls = 0
    last_pass = None
    fail_below = 0.0
    last_margins = None
    for t in grid[::-1]:
        m = (1.0 - t) * m_current + t * m_anchor
        margins = margins_of(simulator, m)
        calls += 1
        if margins.min() >= required_margin:
            last_pass = float(t)
            last_margins = margins
        else:
            fail_below = float(t)
            break
    if last_pass is None:
        margins_anchor = margins_of(simulator, m_anchor)
        return {
            "t": 1.0,
            "m": m_anchor,
            "margins": margins_anchor,
            "min_margin": float(margins_anchor.min()),
            "calls": calls + 1,
            "found": False,
        }

    low, high = fail_below, last_pass
    high_margins = last_margins
    if high > low:
        for _ in range(int(refine_steps)):
            mid = 0.5 * (low + high)
            m = (1.0 - mid) * m_current + mid * m_anchor
            margins = margins_of(simulator, m)
            calls += 1
            if margins.min() >= required_margin:
                high = mid
                high_margins = margins
            else:
                low = mid
    m_line = (1.0 - high) * m_current + high * m_anchor
    return {
        "t": float(high),
        "m": m_line,
        "margins": high_margins,
        "min_margin": float(high_margins.min()),
        "calls": calls,
        "found": True,
    }


def _to_u(m: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    return (np.asarray(m, dtype=np.float64) - center) / half


def _to_m(u: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    return center + half * np.asarray(u, dtype=np.float64)


def local_teacher_search(
    simulator,
    m_current: np.ndarray,
    start_m: np.ndarray,
    required_margin: float,
    basis_dirs: np.ndarray,
    onehot_dirs: np.ndarray,
    max_iter: int = 10,
    population: int = 36,
    eta0: float = 0.05,
    sigma0: float = 0.05,
    seed: int = 0,
    basis_sample: int = 10,
) -> TeacherResult:
    """Constrained local search for a compliant point closer to ``m_current``.

    Lexicographic objective: (1) min margin >= required_margin, (2) smallest
    ||u - u_current||_2, (3) largest min margin on ties.
    """
    lower = np.asarray(simulator.lower_bounds, dtype=np.float64)
    upper = np.asarray(simulator.upper_bounds, dtype=np.float64)
    center = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-6)
    rng = np.random.default_rng(int(seed))

    u_current = _to_u(m_current, center, half)
    u_best = _to_u(start_m, center, half)
    best_m = np.asarray(start_m, dtype=np.float64).copy()
    best_margins = margins_of(simulator, best_m)
    calls = 1
    if best_margins.min() < required_margin:
        return TeacherResult(
            delta=best_m - m_current,
            corrected=best_m,
            margins=best_margins,
            min_margin=float(best_margins.min()),
            passed=False,
            calls=calls,
            iterations=0,
            history=[],
        )
    best_dist = float(np.linalg.norm(u_best - u_current))

    eta, sigma = float(eta0), float(sigma0)
    history = []
    no_improve = 0
    basis = np.asarray(basis_dirs, dtype=np.float64)
    onehot = np.asarray(onehot_dirs, dtype=np.float64)
    directions = np.concatenate([basis, onehot], axis=0) if onehot.size else basis

    for iteration in range(int(max_iter)):
        direction = u_current - u_best
        norm = np.linalg.norm(direction)
        d0 = direction / norm if norm > 1e-12 else np.zeros_like(direction)
        candidates = []
        for scale in (eta, 0.5 * eta, 0.25 * eta, 1.5 * eta):
            candidates.append(u_best + scale * d0)
        selected = rng.choice(
            len(directions), size=min(basis_sample, len(directions)), replace=False
        )
        for idx in selected:
            base = directions[idx]
            for sign in (1.0, -1.0):
                vec = sign * base
                candidates.append(u_best + sigma * vec)
                candidates.append(u_best + eta * d0 + sigma * vec)
        n_random = max(2, int(population) - len(candidates))
        for _ in range(n_random):
            vec = rng.normal(size=u_best.size)
            vec /= np.linalg.norm(vec) + 1e-12
            candidates.append(u_best + eta * d0 + sigma * vec)

        best_candidate = None
        best_candidate_dist = best_dist
        best_candidate_margins = None
        best_candidate_m = None
        for candidate in candidates:
            clipped = np.clip(candidate, -1.0, 1.0)
            m_candidate = _to_m(clipped, center, half)
            margins = margins_of(simulator, m_candidate)
            calls += 1
            if margins.min() < required_margin:
                continue
            dist = float(np.linalg.norm(clipped - u_current))
            better = dist < best_candidate_dist - 1e-9
            tie = (
                abs(dist - best_candidate_dist) <= 1e-9
                and best_candidate_margins is not None
                and margins.min() > best_candidate_margins.min()
            )
            if better or tie:
                best_candidate = clipped
                best_candidate_dist = dist
                best_candidate_margins = margins
                best_candidate_m = m_candidate

        improved = (
            best_candidate is not None and best_candidate_dist < best_dist - 1e-9
        )
        if improved:
            u_best = best_candidate
            best_m = best_candidate_m
            best_margins = best_candidate_margins
            best_dist = best_candidate_dist
            eta = min(eta * 1.15, 0.5)
            sigma = min(sigma * 1.15, 0.5)
            no_improve = 0
        else:
            eta *= 0.6
            sigma *= 0.6
            no_improve += 1
        history.append(
            {
                "iteration": iteration + 1,
                "best_distance": best_dist,
                "min_margin": float(best_margins.min()),
                "eta": float(eta),
                "sigma": float(sigma),
                "improved": bool(improved),
                "calls": int(calls),
            }
        )
        if no_improve >= 3 or (eta < 1e-4 and sigma < 1e-4):
            break

    return TeacherResult(
        delta=best_m - m_current,
        corrected=best_m,
        margins=best_margins,
        min_margin=float(best_margins.min()),
        passed=bool(best_margins.min() >= required_margin),
        calls=calls,
        iterations=len(history),
        history=history,
    )
