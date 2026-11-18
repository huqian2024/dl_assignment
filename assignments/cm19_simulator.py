"""Public simulator interface for the CM19 course assignment.

The module exposes only the operations students need: generate a detuned state,
simulate its complex S-parameters, apply a 41-dimensional correction, and check
the published compliance requirements. It contains no reference solver, label
generation method, or neural-network implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Support both repository imports and a standalone assignments directory.
if __package__:
    from .envs.filter_env import CM19Filter
    from .utils.filter_utils import get_tunable_vector, set_tunable_vector
else:
    from envs.filter_env import CM19Filter
    from utils.filter_utils import get_tunable_vector, set_tunable_vector


@dataclass(frozen=True)
class Requirement:
    name: str
    trace: str
    start_mhz: float
    stop_mhz: float
    operator: str
    threshold_db: float


@dataclass(frozen=True)
class CorrectionResult:
    corrected_vector: np.ndarray
    response: np.ndarray
    all_specs_pass: bool
    max_spec_miss_db: float
    details: dict[str, float | bool]


class CM19Simulator:
    """Deterministic simulator and compliance checker for the assignment."""

    response_channels = ("re_s11", "im_s11", "re_s21", "im_s21")
    response_shape = (4, 2001)
    correction_dim = 41

    def __init__(self) -> None:
        self._env = CM19Filter()
        if self._env.num_tunable != self.correction_dim:
            raise RuntimeError(
                f"Expected {self.correction_dim} tunable entries, got {self._env.num_tunable}"
            )
        if self._env.num_freq_points != self.response_shape[1]:
            raise RuntimeError(
                f"Expected {self.response_shape[1]} frequency points, got {self._env.num_freq_points}"
            )

        self.frequency_hz = np.asarray(self._env.freqs, dtype=np.float64).copy()
        self.lower_bounds = np.asarray(self._env.tunable_lower, dtype=np.float64).copy()
        self.upper_bounds = np.asarray(self._env.tunable_upper, dtype=np.float64).copy()
        self.requirements = tuple(
            Requirement(
                name=spec.name,
                trace=spec.trace.upper(),
                start_mhz=float(spec.start / 1e6),
                stop_mhz=float(spec.end / 1e6),
                operator=spec.op,
                threshold_db=float(spec.threshold),
            )
            for spec in self._env.specs
            if spec.hard
        )

    def _validate_vector(self, vector: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.correction_dim,):
            raise ValueError(f"{name} must have shape ({self.correction_dim},), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return values

    def sample_state(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one reproducible detuned vector and its raw complex response."""
        self._env.reset(seed=int(seed))
        vector = get_tunable_vector(
            self._env.current_matrix,
            self._env.tunable_positions,
        )
        return vector, self.simulate(vector)

    def simulate(self, m_vector: np.ndarray) -> np.ndarray:
        """Map a 41-vector to ``[Re/Im S11, Re/Im S21]`` with shape ``(4,F)``."""
        values = self._validate_vector(m_vector, "m_vector")
        if np.any(values < self.lower_bounds) or np.any(values > self.upper_bounds):
            raise ValueError("m_vector lies outside the published parameter bounds")
        matrix = set_tunable_vector(self._env, values, clip_to_bounds=False)
        s11, s21 = self._env.compute_complex_sparams(matrix)
        response = np.stack(
            [s11.real, s11.imag, s21.real, s21.imag],
            axis=0,
        ).astype(np.float32)
        if response.shape != self.response_shape or not np.isfinite(response).all():
            raise RuntimeError("Simulator returned an invalid response")
        return response

    def response_db(self, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert a raw response tensor to S11/S21 magnitude in dB."""
        values = np.asarray(response, dtype=np.float64)
        if values.shape != self.response_shape:
            raise ValueError(f"response must have shape {self.response_shape}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("response contains NaN or Inf")
        s11 = values[0] + 1j * values[1]
        s21 = values[2] + 1j * values[3]
        s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-10))
        s21_db = 20.0 * np.log10(np.maximum(np.abs(s21), 1e-10))
        return s11_db.astype(np.float32), s21_db.astype(np.float32)

    def check_response(self, response: np.ndarray) -> dict[str, float | bool]:
        """Evaluate every published requirement for one simulated response."""
        s11_db, s21_db = self.response_db(response)
        observation = np.concatenate([s11_db, s21_db])
        info = self._env.check_observation(observation)
        details: dict[str, float | bool] = {}
        misses = []
        for requirement in self.requirements:
            prefix = requirement.name
            details[f"{prefix}_value_db"] = float(info[f"{prefix}_value"])
            details[f"{prefix}_pass"] = bool(info[f"{prefix}_pass"])
            miss = float(info[f"{prefix}_miss"])
            details[f"{prefix}_miss_db"] = miss
            misses.append(miss)
        details["all_specs_pass"] = bool(info["all_specs_pass"])
        details["max_spec_miss_db"] = max(misses) if misses else float("inf")
        return details

    def apply_correction(self, current_vector: np.ndarray, delta_m: np.ndarray) -> CorrectionResult:
        """Apply one correction, clip to legal bounds, simulate, and score it."""
        current = self._validate_vector(current_vector, "current_vector")
        correction = self._validate_vector(delta_m, "delta_m")
        corrected = np.clip(current + correction, self.lower_bounds, self.upper_bounds)
        response = self.simulate(corrected)
        details = self.check_response(response)
        return CorrectionResult(
            corrected_vector=corrected,
            response=response,
            all_specs_pass=bool(details["all_specs_pass"]),
            max_spec_miss_db=float(details["max_spec_miss_db"]),
            details=details,
        )

    def print_requirements(self) -> None:
        header = f"{'Trace':<6} {'Frequency (MHz)':<24} {'Rule':<4} {'Threshold (dB)':>14}"
        print(header)
        print("-" * len(header))
        for item in self.requirements:
            band = f"{item.start_mhz:g}–{item.stop_mhz:g}"
            print(f"{item.trace:<6} {band:<24} {item.operator:<4} {item.threshold_db:>14.2f}")
