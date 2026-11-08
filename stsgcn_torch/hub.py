"""Load the PEMS-STSGCN benchmark dataset from ModelScope / HuggingFace / a local dir.

The dataset layout is described in ``scripts/README.md``: per dataset a
``<DS>_series.parquet`` tensor, ``<DS>_edges.parquet`` and (PEMS03 only)
``<DS>_sensors.parquet``. Three sources are supported:

- ``"local"``: an already-materialised dataset directory (``root=<dir>``).
  Works fully offline — used for testing before/without uploading.
- ``"hf"``: HuggingFace Hub via ``huggingface_hub.snapshot_download``
  (``uv sync --extra hf``, or ``pip install "stsgcn[hf]"``).
- ``"ms"``: ModelScope via ``modelscope.dataset_snapshot_download``
  (``uv sync --extra ms``, or ``pip install "stsgcn[ms]"``).

Repo ids default to the public dataset ``Davidham3/pems-stsgcn`` and can be
overridden per call or with the ``STSGCN_HF_REPO`` / ``STSGCN_MS_REPO``
environment variables. ``root`` doubles as the local dataset directory
(``local``) or the download cache directory (``hf`` / ``ms``).
HuggingFace requests honour ``HF_ENDPOINT`` (e.g. ``https://hf-mirror.com``)
and ``HF_HOME`` / ``HF_HUB_CACHE``; they are read by ``huggingface_hub`` when it
is imported, so export them before starting the run.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np

SOURCES = ("hf", "ms", "local")
CHANNELS = ("flow", "occupancy", "speed")
DEFAULT_REPO = "Davidham3/pems-stsgcn"


def _default_repo(source: str) -> str:
    env = "STSGCN_HF_REPO" if source == "hf" else "STSGCN_MS_REPO"
    return os.environ.get(env, "").strip() or DEFAULT_REPO


@cache
def _snapshot_dir(dataset: str, source: str, repo: str, root: str | None) -> Path:
    """Download (once per process) ``dataset`` from ``repo`` and return its directory."""
    if source == "hf":
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            cache_dir=root,
            allow_patterns=[f"{dataset}/*"],
        )
    else:
        try:
            from modelscope import dataset_snapshot_download
        except ImportError:  # older modelscope layout
            from modelscope.hub.snapshot_download import dataset_snapshot_download
        snap = dataset_snapshot_download(
            repo, cache_dir=root, allow_file_pattern=[f"{dataset}/*"]
        )
    return Path(snap) / dataset


def resolve_dataset_dir(
    dataset: str,
    source: str = "hf",
    repo: str | None = None,
    root: str | None = None,
) -> Path:
    """Return the local directory containing ``<dataset>/<dataset>_*.parquet``.

    For ``hf`` / ``ms`` this downloads (with caching) the dataset repo's files
    belonging to ``dataset``; for ``local`` it resolves ``root/<dataset>``.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")
    if source == "local":
        if root is None:
            raise ValueError("source='local' requires root=<dataset directory>")
        d = Path(root).expanduser() / dataset
        if not (d / f"{dataset}_series.parquet").is_file():
            raise FileNotFoundError(f"{d / f'{dataset}_series.parquet'} not found")
        return d
    repo = repo or _default_repo(source)
    return _snapshot_dir(dataset, source, repo, None if root is None else str(root))


def _read_series(path: Path) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    T = table.num_rows
    if not np.array_equal(
        table["timestep"].to_numpy(), np.arange(T, dtype=np.int32)
    ):
        raise ValueError(f"{path}: timestep column is not 0..T-1")
    channels = []
    for name in CHANNELS:
        if name not in table.column_names:
            continue
        flat = pa.array(pc.list_flatten(table[name].combine_chunks()))
        values = flat.to_numpy(zero_copy_only=False)
        channels.append(values.reshape(T, len(values) // T))
    return np.stack(channels, axis=-1).astype(np.float64)


def load_series(
    dataset: str,
    source: str = "hf",
    repo: str | None = None,
    root: str | None = None,
) -> np.ndarray:
    """Return the original ``(T, N, C)`` graph-signal tensor (float64)."""
    return _read_series(resolve_dataset_dir(dataset, source, repo, root) / f"{dataset}_series.parquet")


def load_edges(
    dataset: str,
    source: str = "hf",
    repo: str | None = None,
    root: str | None = None,
) -> np.ndarray:
    """Return the spatial-network edge list as an ``(M, 3)`` float64 array.

    Columns mirror the original csv: ``from_id``, ``to_id``, ``cost``.
    """
    import pyarrow.parquet as pq

    t = pq.read_table(
        resolve_dataset_dir(dataset, source, repo, root) / f"{dataset}_edges.parquet"
    )
    return np.stack(
        [t["from_id"].to_numpy(), t["to_id"].to_numpy(), t["cost"].to_numpy()], axis=1
    ).astype(np.float64)


def load_sensors(
    dataset: str,
    source: str = "hf",
    repo: str | None = None,
    root: str | None = None,
) -> np.ndarray | None:
    """Return original sensor ids per node, or None if not shipped."""
    import pyarrow.parquet as pq

    path = resolve_dataset_dir(dataset, source, repo, root) / f"{dataset}_sensors.parquet"
    if not path.is_file():
        return None
    return pq.read_table(path)["sensor_id"].to_numpy(zero_copy_only=False).astype(np.int64)


def load_adjacency(
    dataset: str,
    num_of_vertices: int,
    source: str = "hf",
    repo: str | None = None,
    root: str | None = None,
) -> np.ndarray:
    """Return the ``(N, N)`` symmetric connectivity adjacency.

    PEMS03 edge labels are the original sensor ids, so they are remapped to the
    sensor-axis order via ``<DS>_sensors.parquet`` (the row order of the sensor
    list is the node order). Other datasets already use 0-based node indices and
    ship no sensor table.
    """
    from .utils import adjacency_from_edges

    edges = load_edges(dataset, source=source, repo=repo, root=root).copy()
    sensors = load_sensors(dataset, source=source, repo=repo, root=root)
    if sensors is not None:
        id_map = {int(sid): idx for idx, sid in enumerate(sensors)}
        edges[:, 0] = [id_map[int(i)] for i in edges[:, 0]]
        edges[:, 1] = [id_map[int(i)] for i in edges[:, 1]]
    return adjacency_from_edges(edges, num_of_vertices)
