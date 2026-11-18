"""CM19 design assets and the seven published compliance requirements."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


DESIGN_DIR = Path(__file__).resolve().parent / "designs"


@dataclass(frozen=True)
class RFSpec:
    name: str
    trace: str
    start: float
    end: float
    op: str
    threshold: float
    hard: bool = True

    def mask(self, freqs):
        return (freqs >= self.start) & (freqs <= self.end)

    def summary_value(self, values):
        if values.size == 0:
            return float("nan")
        if self.op == "<=":
            return float(np.max(values))
        return float(np.min(values))

    def passed(self, summary):
        if self.op == "<=":
            return bool(summary <= self.threshold)
        return bool(summary >= self.threshold)


@dataclass
class CM19Design:
    nominal_matrix: np.ndarray
    min_matrix: np.ndarray
    max_matrix: np.ndarray
    freqs: np.ndarray
    specs: List[RFSpec]
    f1: float
    f2: float
    q_values: List[float]
    source_port: int = 0
    load_port: int = 1
    detune_noise_std: float = 0.05

    @property
    def matrix_size(self):
        return int(self.nominal_matrix.shape[0])

    @property
    def num_freq_points(self):
        return int(len(self.freqs))


def load_cm19_design():
    paths = [DESIGN_DIR / f"cm19_{part}.csv" for part in ("matrix", "min", "max")]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing CM19 design file: {path}")
    matrix, min_matrix, max_matrix = [np.loadtxt(path, delimiter=",") for path in paths]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"CM19 nominal matrix must be square, got {matrix.shape}")
    if min_matrix.shape != matrix.shape or max_matrix.shape != matrix.shape:
        raise ValueError("CM19 matrix and bounds must have matching shapes")
    return CM19Design(
        nominal_matrix=matrix,
        min_matrix=min_matrix,
        max_matrix=max_matrix,
        freqs=np.linspace(1.5e9, 2.4e9, 2001),
        specs=_cm19_specs(),
        f1=1.710e9,
        f2=2.170e9,
        q_values=[2000.0] + [2400.0] * 16,
    )


def symmetrize_nominal(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix + matrix.T - np.diag(np.diag(matrix))


def get_design_tuning_data(design):
    min_mask = design.min_matrix != 0
    max_mask = design.max_matrix != 0
    if not np.array_equal(min_mask, max_mask):
        mismatch = np.argwhere(min_mask != max_mask)
        first = mismatch[0].tolist() if mismatch.size else None
        raise ValueError(f"CM19 min/max tunable masks do not match; first mismatch={first}")
    positions = np.transpose(np.nonzero(min_mask))
    lower = np.array([design.min_matrix[i, j] for i, j in positions], dtype=np.float64)
    upper = np.array([design.max_matrix[i, j] for i, j in positions], dtype=np.float64)
    if np.any(lower > upper):
        bad = np.argwhere(lower > upper)[0][0]
        raise ValueError(f"CM19 has min > max at tunable position {positions[bad].tolist()}")
    return positions.astype(np.int64), lower, upper


def _cm19_specs():
    return [
        _spec("passband_s11_1", "s11", 1804.0, 1881.0, "<=", -20.0),
        _spec("passband_s11_2", "s11", 2109.0, 2171.0, "<=", -20.0),
        _spec("passband_s21_1", "s21", 1805.0, 1807.5, ">=", -1.53),
        _spec("passband_s21_2", "s21", 1807.5, 1877.5, ">=", -1.28),
        _spec("passband_s21_3", "s21", 1877.5, 1880.0, ">=", -1.20),
        _spec("passband_s21_4", "s21", 2110.0, 2167.0, ">=", -1.25),
        _spec("passband_s21_5", "s21", 2167.5, 2170.0, ">=", -1.28),
    ]


def _spec(name, trace, start_mhz, end_mhz, op, threshold):
    return RFSpec(name, trace, start_mhz * 1e6, end_mhz * 1e6, op, threshold)
