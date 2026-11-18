"""Shared-pole Vector Fitting for CM19 complex S-parameters.

Model (normalized Laplace coordinate ``s``):

    S11(s) = d11 + sum_r c11_r / (s - a_r)
    S21(s) = d21 + sum_r c21_r / (s - a_r)

Poles ``a_r`` are shared.  Frequency is mapped to s = j (w - wc) / ws so the
sampled band lies on [-j, j].  Fitting uses only public S11/S21 data.
"""

from __future__ import annotations

import numpy as np


def normalize_frequency(freqs: np.ndarray) -> tuple[np.ndarray, dict]:
    freqs = np.asarray(freqs, dtype=np.float64)
    wc = 0.5 * (freqs.max() + freqs.min())
    ws = 0.5 * (freqs.max() - freqs.min())
    s = 1j * (freqs - wc) / ws
    return s, {"wc": float(wc), "ws": float(ws)}


def initial_poles(order: int, real_part: float = -0.05) -> np.ndarray:
    """Deterministic conjugate pole pairs, imaginary parts covering [-1, 1]."""
    order = int(order)
    n_pairs = order // 2
    imags = np.linspace(0.05, 0.95, n_pairs)
    poles = []
    for imag in imags:
        poles.append(complex(real_part, float(imag)))
        poles.append(complex(real_part, -float(imag)))
    if order % 2:
        poles.append(complex(-0.1, 0.0))
    return np.asarray(poles, dtype=np.complex128)


def _partial_fraction_basis(s: np.ndarray, poles: np.ndarray) -> np.ndarray:
    return 1.0 / (s[:, None] - poles[None, :])


def _solve_scaled(matrix: np.ndarray, rhs: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    scale = np.linalg.norm(matrix, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    scaled = matrix / scale
    gram = scaled.conj().T @ scaled + float(ridge) * np.eye(scaled.shape[1])
    return np.linalg.solve(gram, scaled.conj().T @ rhs) / scale


def _stable_conjugate_poles(roots: np.ndarray, order: int) -> np.ndarray:
    roots = np.asarray(roots, dtype=np.complex128)
    roots = roots[np.isfinite(roots)]
    # stability: move non-negative real parts into the left half plane
    real = np.where(roots.real < -1e-6, roots.real, -np.maximum(np.abs(roots.real), 1e-6))
    roots = real + 1j * roots.imag
    upper = roots[roots.imag > 1e-8]
    real_poles = roots[np.abs(roots.imag) <= 1e-8]
    n_pairs = order // 2
    if upper.size < n_pairs:
        # add small deterministic imaginary pairs from real poles / defaults
        extra_needed = n_pairs - upper.size
        base = np.linspace(0.05, 0.95, extra_needed)
        upper = np.concatenate([upper, -0.05 + 1j * base])
    upper = upper[np.argsort(upper.imag)]
    upper = upper[:n_pairs]
    poles = []
    for pole in upper:
        poles.append(complex(float(pole.real), float(abs(pole.imag))))
        poles.append(complex(float(pole.real), -float(abs(pole.imag))))
    if order % 2:
        real_value = float(real_poles[0].real) if real_poles.size else -0.1
        poles.append(complex(min(real_value, -1e-6), 0.0))
    return np.asarray(poles, dtype=np.complex128)


def vector_fit(
    s11: np.ndarray,
    s21: np.ndarray,
    s: np.ndarray,
    order: int,
    fit_mask: np.ndarray,
    iterations: int = 6,
    ridge: float = 1e-10,
) -> dict:
    """Shared-pole Vector Fitting with a small deterministic relocation loop."""
    s = np.asarray(s, dtype=np.complex128)
    s11 = np.asarray(s11, dtype=np.complex128)
    s21 = np.asarray(s21, dtype=np.complex128)
    mask = np.asarray(fit_mask, dtype=bool)
    poles = initial_poles(order)
    history = []
    movement = float("inf")
    for iteration in range(int(iterations)):
        basis = _partial_fraction_basis(s[mask], poles)
        a11 = s11[mask]
        a21 = s21[mask]
        n = int(mask.sum())
        n_cols = 3 * int(order) + 2
        matrix = np.zeros((2 * n, n_cols), dtype=np.complex128)
        rhs = np.zeros(2 * n, dtype=np.complex128)
        sigma_start = 2 * int(order) + 2
        # S11 rows: c11, d11, -S11 * sigma
        matrix[:n, :order] = basis
        matrix[:n, order] = 1.0
        matrix[:n, sigma_start:] = -a11[:, None] * basis
        rhs[:n] = a11
        # S21 rows: c21, d21, -S21 * sigma
        matrix[n:, order + 1:order + 1 + order] = basis
        matrix[n:, order + 1 + order] = 1.0
        matrix[n:, sigma_start:] = -a21[:, None] * basis
        rhs[n:] = a21
        solution = _solve_scaled(matrix, rhs, ridge=ridge)
        sigma_coeff = solution[sigma_start:]
        denominator = np.poly(poles)
        numerator = denominator.copy()
        for idx, coeff in enumerate(sigma_coeff):
            without = np.delete(poles, idx)
            term = coeff * np.poly(without)
            numerator = numerator + np.concatenate([[0.0], term])
        roots = np.roots(numerator)
        new_poles = _stable_conjugate_poles(roots, int(order))
        movement = float(np.max(np.abs(new_poles - poles))) if len(new_poles) == len(poles) else float("inf")
        history.append(
            {
                "iteration": iteration + 1,
                "pole_movement": movement,
                "condition": float(np.linalg.cond(matrix)),
            }
        )
        poles = new_poles
        if movement < 1e-8:
            break

    basis = _partial_fraction_basis(s[mask], poles)
    design = np.concatenate([basis, np.ones((int(mask.sum()), 1), dtype=np.complex128)], axis=1)
    c11_d11 = _solve_scaled(design, s11[mask], ridge=ridge)
    c21_d21 = _solve_scaled(design, s21[mask], ridge=ridge)
    c11, d11 = c11_d11[:-1], c11_d11[-1]
    c21, d21 = c21_d21[:-1], c21_d21[-1]
    return {
        "poles": poles,
        "c11": c11,
        "c21": c21,
        "d11": complex(d11),
        "d21": complex(d21),
        "order": int(order),
        "iterations": len(history),
        "pole_movement": movement,
        "history": history,
    }


def evaluate_vector_fit(model: dict, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = _partial_fraction_basis(np.asarray(s, dtype=np.complex128), model["poles"])
    s11 = model["d11"] + basis @ model["c11"]
    s21 = model["d21"] + basis @ model["c21"]
    return s11, s21
