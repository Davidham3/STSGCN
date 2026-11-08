"""Tests for stsgcn_torch.hub: loading the parquet dataset (local simulation).

These run against a locally materialised dataset directory
(``dist/pems-stsgcn``), which has the same layout as the ModelScope repo — the
``local`` source therefore exercises the Hub code paths without network access.
"""

import itertools
import os
import sys
import types

import numpy as np
import pytest

from stsgcn_torch import hub
from stsgcn_torch.utils import adjacency_from_edges, construct_adj, split_series

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(REPO_ROOT, "dist", "pems-stsgcn")

requires_dist = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DIST_DIR, "PEMS08")),
    reason="dataset directory dist/pems-stsgcn is missing",
)


def _local() -> dict:
    return {"source": "local", "root": DIST_DIR}


def _local_kwargs() -> dict:
    # constructor-argument spelling used by STSGCN / STSGCNDataModule
    return {"hub_source": "local", "hub_root": DIST_DIR}


@requires_dist
def test_load_series():
    for ds, shape in [
        ("PEMS03", (26208, 358, 1)),
        ("PEMS04", (16992, 307, 3)),
        ("PEMS07", (28224, 883, 1)),
        ("PEMS08", (17856, 170, 3)),
    ]:
        arr = hub.load_series(ds, **_local())
        assert arr.shape == shape
        assert arr.dtype == np.float64


@requires_dist
def test_load_adjacency_matches_edges():
    ds = "PEMS08"
    edges = hub.load_edges(ds, **_local())
    assert edges.ndim == 2 and edges.shape[1] == 3
    A = hub.load_adjacency(ds, 170, **_local())
    assert np.array_equal(A, adjacency_from_edges(edges, 170))
    assert np.array_equal(A, A.T)


@requires_dist
def test_load_adjacency_remaps_pems03_sensor_ids():
    """PEMS03 edges use original sensor ids; the sensors table remaps them."""
    ds = "PEMS03"
    edges = hub.load_edges(ds, **_local()).copy()
    sensors = hub.load_sensors(ds, **_local())
    id_map = {int(sid): idx for idx, sid in enumerate(sensors)}
    edges[:, 0] = [id_map[int(i)] for i in edges[:, 0]]
    edges[:, 1] = [id_map[int(i)] for i in edges[:, 1]]

    A = hub.load_adjacency(ds, 358, **_local())
    assert np.array_equal(A, adjacency_from_edges(edges, 358))


@requires_dist
def test_load_sensors():
    sensors = hub.load_sensors("PEMS03", **_local())
    assert sensors.shape == (358,)
    assert len(set(sensors.tolist())) == 358
    assert hub.load_sensors("PEMS04", **_local()) is None


@requires_dist
def test_model_adjacency_and_mask_from_hub():
    import torch

    from stsgcn_torch import STSGCN

    for ds, n in [("PEMS08", 170), ("PEMS03", 358)]:
        adj = construct_adj(hub.load_adjacency(ds, n, **_local()), 3)
        model = STSGCN(
            num_of_vertices=n,
            hub_dataset=ds,
            filter_list=[[8, 8]],
            first_layer_embedding_size=8,
            **_local_kwargs(),
        )
        assert torch.equal(model.adj, torch.from_numpy(adj))
        assert torch.equal(model.mask, torch.from_numpy((adj != 0).astype(np.float32)))


@requires_dist
def test_split_series():
    arr = hub.load_series("PEMS08", **_local())
    splits = split_series(arr)
    assert len(splits) == 3

    length = arr.shape[0]
    bounds = (0, int(length * 0.6), int(length * 0.8), length)
    for (x, y), (start, end) in zip(splits, itertools.pairwise(bounds)):
        n = end - start - 12 - 12 + 1
        assert x.shape == (n, 12, 170, 1) and y.shape == (n, 12, 170, 1)

    train_x, train_y = splits[0]
    # only x is standardized, with training statistics
    assert abs(float(train_x.mean())) < 1e-6
    assert abs(float(train_x.std()) - 1.0) < 1e-6
    # targets stay raw: first window's y is the unstandardised next 12 steps
    assert np.array_equal(train_y[0], arr[12:24, :, :1])


def test_source_validation(tmp_path):
    with pytest.raises(ValueError, match="unknown source"):
        hub.resolve_dataset_dir("PEMS08", source="s3")
    with pytest.raises(ValueError, match="requires root"):
        hub.resolve_dataset_dir("PEMS08", source="local")
    with pytest.raises(FileNotFoundError):
        hub.resolve_dataset_dir("PEMS08", source="local", root=str(tmp_path))


def test_resolve_hf_calls_huggingface_hub(tmp_path, monkeypatch):
    calls = {}
    mod = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(repo_id, repo_type=None, cache_dir=None, allow_patterns=None):
        calls.update(
            repo_id=repo_id,
            repo_type=repo_type,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
        )
        (tmp_path / "snap" / "PEMS04").mkdir(parents=True)
        return str(tmp_path / "snap")

    mod.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.delenv("STSGCN_HF_REPO", raising=False)
    cache = str(tmp_path / "cache")

    # No env override: the public dataset repo is used by default.
    d = hub.resolve_dataset_dir("PEMS04", source="hf", root=cache)
    assert d == tmp_path / "snap" / "PEMS04"
    assert calls == {
        "repo_id": hub.DEFAULT_REPO,
        "repo_type": "dataset",
        "cache_dir": cache,
        "allow_patterns": ["PEMS04/*"],
    }


def test_resolve_ms_calls_modelscope(tmp_path, monkeypatch):
    calls = {}
    mod = types.ModuleType("modelscope")

    def fake_download(repo, cache_dir=None, allow_file_pattern=None):
        calls.update(repo=repo, cache_dir=cache_dir, pattern=allow_file_pattern)
        (tmp_path / "snap" / "PEMS08").mkdir(parents=True)
        return str(tmp_path / "snap")

    mod.dataset_snapshot_download = fake_download
    monkeypatch.setitem(sys.modules, "modelscope", mod)
    monkeypatch.setenv("STSGCN_MS_REPO", "org/pems-stsgcn")

    d = hub.resolve_dataset_dir("PEMS08", source="ms")
    assert d == tmp_path / "snap" / "PEMS08"
    assert calls == {
        "repo": "org/pems-stsgcn",
        "cache_dir": None,
        "pattern": ["PEMS08/*"],
    }


def test_resolve_ms_uses_default_repo(tmp_path, monkeypatch):
    calls = {}
    mod = types.ModuleType("modelscope")

    def fake_download(repo, cache_dir=None, allow_file_pattern=None):
        calls.update(repo=repo)
        (tmp_path / "snap" / "PEMS07").mkdir(parents=True)
        return str(tmp_path / "snap")

    mod.dataset_snapshot_download = fake_download
    monkeypatch.setitem(sys.modules, "modelscope", mod)
    monkeypatch.delenv("STSGCN_MS_REPO", raising=False)

    assert hub.resolve_dataset_dir("PEMS07", source="ms") == tmp_path / "snap" / "PEMS07"
    assert calls == {"repo": hub.DEFAULT_REPO}
