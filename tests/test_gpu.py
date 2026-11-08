"""GPU smoke test: forward/backward + a few optimizer steps on random samples."""

import pytest
import torch
from helpers import random_adjacency

from stsgcn_torch import STSGCN

NUM_OF_VERTICES = 32
INPUT_LENGTH = 12
NUM_FOR_PREDICT = 12
FILTERS = [[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]]
BATCH_SIZE = 8


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def adj():
    return random_adjacency(NUM_OF_VERTICES)


def build_model(module_type: str, activation: str, adj) -> STSGCN:
    return STSGCN(
        adj_mx=adj,
        num_of_vertices=NUM_OF_VERTICES,
        input_length=INPUT_LENGTH,
        num_of_features=1,
        filter_list=FILTERS,
        module_type=module_type,
        activation=activation,
        use_mask=True,
        temporal_emb=True,
        spatial_emb=True,
        first_layer_embedding_size=64,
        num_for_predict=NUM_FOR_PREDICT,
    )


def test_forward_backward_and_steps(device, adj):
    model = build_model("individual", "GLU", adj).to(device)
    x = torch.randn(BATCH_SIZE, INPUT_LENGTH, NUM_OF_VERTICES, 1, device=device)

    y = model(x)
    assert y.shape == (BATCH_SIZE, NUM_FOR_PREDICT, NUM_OF_VERTICES)

    label = torch.randn(BATCH_SIZE, NUM_FOR_PREDICT, NUM_OF_VERTICES, device=device)
    loss = torch.abs(y - label).mean()
    loss.backward()
    assert model.mask.grad is not None, "mask should receive gradient"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        l = torch.abs(model(x) - label).mean()
        l.backward()
        optimizer.step()


@pytest.mark.parametrize(
    "module_type,activation",
    [("sharing", "GLU"), ("individual", "relu"), ("sharing", "relu")],
)
def test_variant_output_shapes(device, adj, module_type, activation):
    model = build_model(module_type, activation, adj).to(device)
    x = torch.randn(BATCH_SIZE, INPUT_LENGTH, NUM_OF_VERTICES, 1, device=device)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_FOR_PREDICT, NUM_OF_VERTICES)
