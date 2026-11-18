"""Conversion between CM19 matrices and ordered tunable parameter vectors."""

import numpy as np


def get_tunable_vector(matrix, positions):
    """Extract tunable entries in the parameter order used by the simulator."""
    return np.array([matrix[i, j] for i, j in positions], dtype=np.float64)


def set_tunable_vector(env, vector, clip_to_bounds=True):
    """Set tunable entries in a fresh nominal matrix and preserve symmetry."""
    if len(vector) != env.num_tunable:
        raise ValueError(f"Expected {env.num_tunable} tunable values, got {len(vector)}")
    matrix = np.copy(env.nominal_matrix)
    for idx, (value, (i, j)) in enumerate(zip(vector, env.tunable_positions)):
        if clip_to_bounds:
            value = np.clip(value, env.tunable_lower[idx], env.tunable_upper[idx])
        matrix[i, j] = value
        if i != j:
            matrix[j, i] = value
    env.current_matrix = matrix
    return matrix
