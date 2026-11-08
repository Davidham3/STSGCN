"""Data utilities for STSGCN (adjacency construction, sequence generation, metrics).

These functions mirror the behaviour of the original MXNet ``utils.py`` but are
written on top of numpy only.
"""

from __future__ import annotations

import numpy as np


def construct_adj(A: np.ndarray, steps: int) -> np.ndarray:
    """Construct the localized ``(N*steps, N*steps)`` adjacency matrix."""
    N = len(A)
    adj = np.zeros([N * steps] * 2, dtype=np.float32)

    for i in range(steps):
        adj[i * N : (i + 1) * N, i * N : (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def adjacency_from_edges(edges: np.ndarray, num_of_vertices: int) -> np.ndarray:
    """Build a symmetric connectivity adjacency from an ``(M, 3)`` edge array."""
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    i = edges[:, 0].astype(np.int64)
    j = edges[:, 1].astype(np.int64)
    A[i, j] = 1
    A[j, i] = 1
    return A


def generate_seq(
    data: np.ndarray, train_length: int, pred_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window sequence generation -> ``(x, y)``.

    ``x`` shape ``(num_samples, train_length, N, 1)``,
    ``y`` shape ``(num_samples, pred_length, N, 1)``.
    """
    seq = np.concatenate(
        [
            np.expand_dims(data[i : i + train_length + pred_length], 0)
            for i in range(data.shape[0] - train_length - pred_length + 1)
        ],
        axis=0,
    )[:, :, :, 0:1]
    return np.split(seq, 2, axis=1)


def split_series(
    data: np.ndarray, train_length: int = 12, pred_length: int = 12
) -> list[tuple[np.ndarray, np.ndarray]]:
    """60 / 20 / 20 temporal split + sliding windows -> ``[train, val, test]``.

    Only ``x`` is standardized (mean/std of the training split); targets ``y``
    stay in the original scale, matching the original MXNet implementation.
    """
    length = data.shape[0]
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    splits = []
    mean = std = None
    for start, end in ((0, train_line), (train_line, val_line), (val_line, length)):
        x, y = generate_seq(data[start:end], train_length, pred_length)
        if mean is None:
            mean, std = x.mean(), x.std()
        splits.append(((x - mean) / std, y))
    return splits


def mask_np(array: np.ndarray, null_val) -> np.ndarray:
    if np.isnan(null_val):
        return (~np.isnan(array)).astype("float32")
    return np.not_equal(array, null_val).astype("float32")


def masked_mape_np(y_true, y_pred, null_val=np.nan) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return float(np.mean(mape) * 100)


def masked_mse_np(y_true, y_pred, null_val=np.nan) -> float:
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return float(np.mean(np.nan_to_num(mask * mse)))


def masked_mae_np(y_true, y_pred, null_val=np.nan) -> float:
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return float(np.mean(np.nan_to_num(mask * mae)))
