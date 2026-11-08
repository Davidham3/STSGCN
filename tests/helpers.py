"""Test-only data generators (not part of the stsgcn_torch package)."""

from __future__ import annotations

import numpy as np

from stsgcn_torch import construct_adj


def chain_adjacency(num_of_vertices: int) -> np.ndarray:
    """Deterministic localized adjacency of a chain graph."""
    A = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    for i in range(num_of_vertices - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    return construct_adj(A, 3)


def random_adjacency(num_of_vertices: int, seed: int = 0) -> np.ndarray:
    """Random symmetric localized adjacency matrix."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=(num_of_vertices, num_of_vertices)).astype(np.float32)
    A = np.triu(A, 1)
    A = A + A.T
    return construct_adj(A, 3)
