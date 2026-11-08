"""Validate the Lightning CLI YAML configs."""

import glob
import os

import numpy as np
import pytest
import torch
import yaml

from stsgcn_torch import STSGCN

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

MODEL_REQUIRED = [
    "num_of_vertices", "input_length", "filter_list",
    "module_type", "activation", "learning_rate",
]
DATA_REQUIRED = ["batch_size"]


def _yaml_files():
    return sorted(glob.glob(os.path.join(CONFIG_DIR, "**", "*.yaml"), recursive=True))


def test_config_count():
    assert len(_yaml_files()) == 12


def test_yaml_configs_structure():
    for path in _yaml_files():
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert {"model", "data", "trainer"} <= set(cfg), path
        model, data = cfg["model"], cfg["data"]
        for k in MODEL_REQUIRED:
            assert k in model, (path, k)
        for k in DATA_REQUIRED:
            assert k in data, (path, k)
        # the graph and series both come from the PEMS-STSGCN hub dataset
        assert "hub_dataset" in model and "hub_dataset" in data, path
        assert model["hub_dataset"] == data["hub_dataset"], path
        assert model["hub_source"] == data["hub_source"] == "ms", path
        assert model["module_type"] in {"individual", "sharing"}
        assert model["activation"] in {"GLU", "relu"}
        assert len(model["filter_list"]) > 0 and all(len(f) > 0 for f in model["filter_list"])


@pytest.mark.parametrize("path", _yaml_files(), ids=lambda p: os.path.relpath(p, CONFIG_DIR))
def test_yaml_configs_parse_with_lightning_cli(path, monkeypatch):
    """The full config must be parseable by Lightning CLI (arg names, types, callbacks).

    ``--trainer.devices 1`` overrides the config so parsing does not depend on
    how many GPUs the current machine has. Instantiating the model would hit
    ModelScope, so the adjacency download is stubbed out.
    """
    from pytorch_lightning.cli import LightningCLI

    from stsgcn_torch import STSGCNDataModule, hub

    monkeypatch.setattr(
        hub, "load_adjacency", lambda ds, n, **kw: np.eye(int(n), dtype=np.float32)
    )
    LightningCLI(
        STSGCN,
        STSGCNDataModule,
        run=False,
        args=["--config", path, "--trainer.devices", "1"],
    )


def test_config_builds_model_and_forwards():
    path = os.path.join(CONFIG_DIR, "PEMS04", "individual_GLU_mask_emb.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Shrink N for a fast smoke check; everything else comes from the real config.
    from helpers import chain_adjacency

    model_cfg = dict(cfg["model"])
    small_n = 16
    model_cfg["num_of_vertices"] = small_n
    model_cfg.pop("hub_dataset")
    model_cfg.pop("hub_source")

    model = STSGCN(adj_mx=chain_adjacency(small_n), **model_cfg)
    x = torch.randn(2, model_cfg["input_length"], small_n, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 12, small_n)
