"""System-identification features for CM19 complex S-parameters.

Only public data are used: the fixed frequency grid and the complex S11/S21
responses.  No simulator internals are assumed.

Main ingredients:
  * shared-denominator rational fitting with a Chebyshev basis
  * frequency-holdout reconstruction metrics
  * complex algebra features (regularized 1/S21, S11/S21, log derivative)
  * rational coefficient / pole-zero feature extraction
"""

from __future__ import annotations

import numpy as np


def cheb_basis(x: np.ndarray, order: int) -> np.ndarray:
    """Chebyshev T_0..T_order, shape (len(x), order+1)."""
    return np.polynomial.chebyshev.chebvander(np.asarray(x, dtype=np.float64), int(order))


def fit_shared_rational(
    s11: np.ndarray,
    s21: np.ndarray,
    x: np.ndarray,
    order: int,
    fit_mask: np.ndarray | None = None,
    ridge: float = 1e-10,
) -> dict:
    """Complex least-squares fit of S11=N11/D, S21=N21/D with shared D.

    ``D(x) = 1 + sum_{k=1..order} d_k T_k(x)``
    ``N11(x) = sum_{k=0..order} a_k T_k(x)``
    ``N21(x) = sum_{k=0..order} b_k T_k(x)``
    """
    s11 = np.asarray(s11, dtype=np.complex128)
    s21 = np.asarray(s21, dtype=np.complex128)
    x = np.asarray(x, dtype=np.float64)
    mask = (
        np.ones(len(x), dtype=bool)
        if fit_mask is None
        else np.asarray(fit_mask, dtype=bool)
    )
    order = int(order)
    if mask.sum() < 4 * (order + 1):
        raise ValueError("not enough frequency samples for the requested order")

    t = cheb_basis(x, order)
    tf = t[mask]
    a11 = s11[mask]
    a21 = s21[mask]
    n_fit = int(mask.sum())
    n_cols = order + 2 * (order + 1)

    matrix = np.zeros((2 * n_fit, n_cols), dtype=np.complex128)
    rhs = np.zeros(2 * n_fit, dtype=np.complex128)
    matrix[:n_fit, :order] = a11[:, None] * tf[:, 1:]
    matrix[:n_fit, order:order + order + 1] = -tf
    rhs[:n_fit] = -a11
    matrix[n_fit:, :order] = a21[:, None] * tf[:, 1:]
    matrix[n_fit:, order + order + 1:] = -tf
    rhs[n_fit:] = -a21

    scale = np.linalg.norm(matrix, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    matrix = matrix / scale
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        condition = float("inf")
    gram = matrix.conj().T @ matrix
    gram = gram + float(ridge) * np.eye(n_cols, dtype=np.complex128)
    solution = np.linalg.solve(gram, matrix.conj().T @ rhs) / scale
    return {
        "d": solution[:order],
        "a": solution[order:order + order + 1],
        "b": solution[order + order + 1:],
        "order": order,
        "condition": condition,
    }


def evaluate_rational(coeffs: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = int(coeffs["order"])
    t = cheb_basis(x, order)
    denom = 1.0 + t[:, 1:] @ coeffs["d"]
    s11 = (t @ coeffs["a"]) / denom
    s21 = (t @ coeffs["b"]) / denom
    return s11, s21


def reconstruction_metrics(
    s11: np.ndarray,
    s21: np.ndarray,
    s11_hat: np.ndarray,
    s21_hat: np.ndarray,
    mask: np.ndarray,
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    true = np.concatenate([s11[mask], s21[mask]])
    pred = np.concatenate([s11_hat[mask], s21_hat[mask]])
    diff = pred - true
    finite = np.isfinite(diff)
    rel_l2 = float(
        np.linalg.norm(diff[finite]) / (np.linalg.norm(true[finite]) + 1e-30)
    )
    max_abs = float(np.max(np.abs(diff[finite]))) if finite.any() else float("inf")
    true_db = 20.0 * np.log10(np.maximum(np.abs(true), 1e-12))
    pred_db = 20.0 * np.log10(np.maximum(np.abs(pred), 1e-12))
    db_err = np.abs(pred_db - true_db)
    return {
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "median_db_error": float(np.median(db_err[finite])) if finite.any() else float("inf"),
        "p95_db_error": float(np.quantile(db_err[finite], 0.95)) if finite.any() else float("inf"),
        "nonfinite_fraction": float(1.0 - finite.mean()) if len(finite) else 1.0,
    }


def spec_reconstruction(
    simulator,
    s11: np.ndarray,
    s21: np.ndarray,
    s11_hat: np.ndarray,
    s21_hat: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compare the seven specification summaries on true vs reconstructed traces."""
    freqs = np.asarray(simulator.frequency_hz, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    true_db = {
        "s11": 20.0 * np.log10(np.maximum(np.abs(s11), 1e-12)),
        "s21": 20.0 * np.log10(np.maximum(np.abs(s21), 1e-12)),
    }
    pred_db = {
        "s11": 20.0 * np.log10(np.maximum(np.abs(s11_hat), 1e-12)),
        "s21": 20.0 * np.log10(np.maximum(np.abs(s21_hat), 1e-12)),
    }
    errors = {}
    pass_matches = []
    for req in simulator.requirements:
        band = (freqs >= float(req.start_mhz) * 1e6) & (
            freqs <= float(req.stop_mhz) * 1e6
        )
        band &= mask
        if not band.any():
            continue
        trace = "s11" if str(req.trace).upper() == "S11" else "s21"
        if req.operator == "<=":
            t_val = float(np.max(true_db[trace][band]))
            p_val = float(np.max(pred_db[trace][band]))
            t_pass = t_val <= float(req.threshold_db)
            p_pass = p_val <= float(req.threshold_db)
        else:
            t_val = float(np.min(true_db[trace][band]))
            p_val = float(np.min(pred_db[trace][band]))
            t_pass = t_val >= float(req.threshold_db)
            p_pass = p_val >= float(req.threshold_db)
        errors[req.name] = {
            "true_value_db": t_val,
            "reconstructed_value_db": p_val,
            "error_db": p_val - t_val,
            "pass_match": bool(t_pass == p_pass),
        }
        pass_matches.append(bool(t_pass == p_pass))
    return {
        "per_requirement": errors,
        "pass_match_rate": float(np.mean(pass_matches)) if pass_matches else 0.0,
        "max_abs_value_error_db": (
            float(max(abs(item["error_db"]) for item in errors.values()))
            if errors
            else float("inf")
        ),
    }


def coefficient_features(coeffs: dict) -> np.ndarray:
    vec = np.concatenate([coeffs["d"], coeffs["a"], coeffs["b"]])
    return np.concatenate([vec.real, vec.imag]).astype(np.float64)


def pole_zero_features(coeffs: dict) -> np.ndarray:
    """Deterministically sorted poles and zeros in the monomial basis."""
    order = int(coeffs["order"])

    def sorted_roots(poly: np.ndarray, count: int) -> np.ndarray:
        roots = np.roots(poly) if len(poly) > 1 else np.array([], dtype=np.complex128)
        roots = roots[np.isfinite(roots)]
        if roots.size < count:
            roots = np.concatenate(
                [roots, np.zeros(count - roots.size, dtype=np.complex128)]
            )
        roots = roots[:count]
        idx = np.lexsort((roots.imag, roots.real))
        return roots[idx]

    try:
        d_mono = np.polynomial.chebyshev.cheb2poly(
            np.concatenate([[1.0 + 0j], coeffs["d"]])
        )
        a_mono = np.polynomial.chebyshev.cheb2poly(coeffs["a"])
        b_mono = np.polynomial.chebyshev.cheb2poly(coeffs["b"])
    except (TypeError, ValueError):
        return np.full(2 * (3 * order + 2), np.nan)
    poles = sorted_roots(d_mono, order)
    zeros11 = sorted_roots(a_mono, order + 1)
    zeros21 = sorted_roots(b_mono, order + 1)
    vec = np.concatenate([poles, zeros11, zeros21])
    return np.concatenate([vec.real, vec.imag]).astype(np.float64)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float) -> np.ndarray:
    return numerator * np.conj(denominator) / (np.abs(denominator) ** 2 + eps)


def complex_algebra_features(
    s11: np.ndarray,
    s21: np.ndarray,
    eps: float,
    kind: str,
) -> np.ndarray:
    inv_s21 = _safe_ratio(np.ones_like(s21), s21, eps)
    ratio = _safe_ratio(s11, s21, eps)
    if kind == "inverse":
        parts = [np.real(inv_s21), np.imag(inv_s21)]
    elif kind == "ratio":
        parts = [np.real(ratio), np.imag(ratio)]
    elif kind == "combined":
        parts = [
            np.real(s11), np.imag(s11),
            np.real(s21), np.imag(s21),
            np.real(inv_s21), np.imag(inv_s21),
            np.real(ratio), np.imag(ratio),
        ]
    else:
        raise ValueError(f"Unknown algebra feature kind '{kind}'")
    return np.concatenate(parts).astype(np.float64)


def log_derivative_features(
    s11: np.ndarray,
    s21: np.ndarray,
    eps: float,
) -> np.ndarray:
    def one(signal: np.ndarray) -> np.ndarray:
        deriv = np.gradient(signal)
        return deriv * np.conj(signal) / (np.abs(signal) ** 2 + eps)

    l11 = one(s11)
    l21 = one(s21)
    return np.concatenate(
        [np.real(l11), np.imag(l11), np.real(l21), np.imag(l21)]
    ).astype(np.float64)
