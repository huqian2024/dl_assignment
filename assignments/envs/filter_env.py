"""Numerical CM19 response calculation and reproducible state sampling."""

import numpy as np

from .filter_designs import get_design_tuning_data, load_cm19_design, symmetrize_nominal


class CM19Filter:
    """Internal numerical backend for the public CM19Simulator interface."""

    def __init__(self):
        self.design = load_cm19_design()
        self.matrix_size = self.design.matrix_size
        self.num_freq_points = self.design.num_freq_points
        self.freqs = self.design.freqs
        self.f1 = self.design.f1
        self.f2 = self.design.f2
        self.q_values = self.design.q_values
        self.source_port = self.design.source_port
        self.load_port = self.design.load_port
        self.detune_noise_std = self.design.detune_noise_std
        self.specs = self.design.specs
        self.nominal_matrix = symmetrize_nominal(self.design.nominal_matrix)
        self.tunable_positions, self.tunable_lower, self.tunable_upper = get_design_tuning_data(self.design)
        self.num_tunable = len(self.tunable_positions)
        self.current_matrix = None
        self._rng = None

    def reset(self, seed=None):
        """Sample independent Gaussian detuning and clip each tunable entry."""
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        self.current_matrix = np.copy(self.nominal_matrix)
        noise = self._rng.normal(0, self.detune_noise_std, size=self.num_tunable)
        for idx, (i, j) in enumerate(self.tunable_positions):
            value = self.current_matrix[i, j] + noise[idx]
            value = np.clip(value, self.tunable_lower[idx], self.tunable_upper[idx])
            self.current_matrix[i, j] = value
            if i != j:
                self.current_matrix[j, i] = value

    def compute_complex_sparams(self, matrix=None):
        """Return complex S11 and S21 on the fixed assignment frequency grid."""
        if matrix is None:
            if self.current_matrix is None:
                raise ValueError("Call reset() or pass an explicit matrix first")
            matrix = self.current_matrix
        matrix = np.array(matrix, dtype=np.complex128, copy=True)
        f0 = np.sqrt(self.f1 * self.f2)
        bw = self.f2 - self.f1
        for i, q_value in enumerate(self.q_values):
            matrix[i + 2, i + 2] = matrix[i + 2, i + 2] - 1.0j * f0 / bw / q_value

        n = self.matrix_size - 2
        cf = np.sqrt(self.f1 * self.f2)
        fbw = (self.f2 - self.f1) / cf
        r_matrix = np.zeros((n + 2, n + 2))
        r_matrix[self.source_port, self.source_port] = 1.0
        r_matrix[self.load_port, self.load_port] = 1.0
        w_matrix = np.eye(n + 2)
        w_matrix[self.source_port, self.source_port] = 0.0
        w_matrix[self.load_port, self.load_port] = 0.0
        s11 = np.zeros(self.num_freq_points, dtype=np.complex128)
        s21 = np.zeros(self.num_freq_points, dtype=np.complex128)
        for idx, freq in enumerate(self.freqs):
            wp = ((freq / cf) - (cf / freq)) / fbw * 1j
            a_matrix = r_matrix + w_matrix * wp + 1j * matrix
            a_inv = np.linalg.inv(a_matrix)
            s11[idx] = -1 + 2.0 * a_inv[self.source_port, self.source_port]
            s21[idx] = 2.0 * a_inv[self.load_port, self.source_port]
        return s11, s21

    def check_observation(self, observation):
        """Evaluate the seven requirements using S11/S21 magnitudes in dB."""
        traces = {
            "s11": observation[:self.num_freq_points],
            "s21": observation[self.num_freq_points:],
        }
        info = {}
        hard_passes = []
        for spec in self.specs:
            mask = spec.mask(self.freqs)
            values = traces[spec.trace][mask] if np.any(mask) else np.array([])
            summary = spec.summary_value(values)
            passed = spec.passed(summary)
            info[f"{spec.name}_value"] = summary
            info[f"{spec.name}_pass"] = passed
            info[f"{spec.name}_miss"] = self._spec_miss(spec, summary)
            if spec.hard:
                hard_passes.append(passed)
        info["all_specs_pass"] = bool(all(hard_passes)) if hard_passes else False
        return info

    @staticmethod
    def _spec_miss(spec, summary):
        if np.isnan(summary):
            return float("inf")
        if spec.op == "<=":
            return max(0.0, summary - spec.threshold)
        return max(0.0, spec.threshold - summary)
