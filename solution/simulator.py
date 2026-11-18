"""Thin helpers around the assignment's public ``CM19Simulator``.

Only the documented student-facing API is used here:

    sample_state, simulate, check_response, apply_correction,
    lower_bounds, upper_bounds, requirements, frequency_hz, response_db

No simulator internals are touched or modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .utils import ensure_assignments_on_path


def load_simulator():
    ensure_assignments_on_path()
    from cm19_simulator import CM19Simulator  # noqa: WPS433 (runtime import)

    return CM19Simulator()


@dataclass(frozen=True)
class RequirementInfo:
    name: str
    trace: str  # "S11" or "S21"
    start_mhz: float
    stop_mhz: float
    operator: str  # "<=" or ">="
    threshold_db: float


def requirement_infos(simulator) -> list[RequirementInfo]:
    return [
        RequirementInfo(
            name=req.name,
            trace=req.trace.upper(),
            start_mhz=float(req.start_mhz),
            stop_mhz=float(req.stop_mhz),
            operator=str(req.operator),
            threshold_db=float(req.threshold_db),
        )
        for req in simulator.requirements
    ]


def signed_margins(simulator, response: np.ndarray) -> np.ndarray:
    """Signed passes for the seven published requirements.

    ``margin > 0`` means the requirement is satisfied; the sign convention
    removes the max/min asymmetry of the assignment specification.
    """
    details = simulator.check_response(response)
    margins = []
    for req in requirement_infos(simulator):
        value = float(details[f"{req.name}_value_db"])
        if req.operator == "<=":
            margins.append(req.threshold_db - value)
        else:
            margins.append(value - req.threshold_db)
    return np.asarray(margins, dtype=np.float64)


def min_margin(simulator, response: np.ndarray) -> float:
    return float(np.min(signed_margins(simulator, response)))


def check_vector(simulator, m_vector: np.ndarray) -> dict:
    """Simulate one 41-vector and return compliance details + margins."""
    response = simulator.simulate(np.asarray(m_vector, dtype=np.float64))
    details = dict(simulator.check_response(response))
    details["margins"] = signed_margins(simulator, response)
    details["min_margin_db"] = float(np.min(details["margins"]))
    return details


def states_to_responses(simulator, seed: int) -> tuple[np.ndarray, np.ndarray]:
    m_current, response = simulator.sample_state(int(seed))
    return (
        np.asarray(m_current, dtype=np.float32),
        np.asarray(response, dtype=np.float32),
    )


def summarize_details(simulator, rows: Iterable[dict]) -> dict:
    """Aggregate per-requirement pass rates and margins over states."""
    rows = list(rows)
    if not rows:
        return {"count": 0}
    margins = np.stack([row["margins"] for row in rows])
    names = [req.name for req in requirement_infos(simulator)]
    return {
        "count": len(rows),
        "pass_rate": float(np.mean([row["all_specs_pass"] for row in rows])),
        "mean_min_margin_db": float(np.mean(np.min(margins, axis=1))),
        "per_requirement_pass_rate": {
            name: float(np.mean(margins[:, idx] > 0.0))
            for idx, name in enumerate(names)
        },
    }
