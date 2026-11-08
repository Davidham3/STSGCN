"""Unit tests mirroring the original ``STSGCN/test/test_stsgcn.py`` shape checks."""

import numpy as np
import pytest
import torch

from stsgcn_torch import STSGCL, STSGCM, STSGCN, GCNOperation, OutputLayer, huber_loss


def _adj(shape, seed=0):
    return torch.randn(*shape)


def test_position_embedding_shapes():
    from stsgcn_torch.model import PositionEmbedding

    B, T, N, C = 2, 12, 16, 32
    data = torch.randn(B, T, N, C)
    for temporal in (True, False):
        for spatial in (True, False):
            layer = PositionEmbedding(T, N, C, temporal, spatial)
            out = layer(data)
            assert out.shape == (B, T, N, C)


def test_gcn_operation_shapes():
    B, N, C = 2, 16, 32
    data = torch.randn(3 * N, B, C)
    adj = torch.randn(3 * N, 3 * N)
    for activation in ("GLU", "relu"):
        layer = GCNOperation(64, C, activation)
        assert layer(data, adj).shape == (3 * N, B, 64)


def test_stsgcm_shapes():
    B, N, C = 2, 16, 32
    data = torch.randn(3 * N, B, C)
    adj = torch.randn(3 * N, 3 * N)
    for activation in ("GLU", "relu"):
        layer = STSGCM([3, 3, 3], C, N, activation)
        assert layer(data, adj).shape == (N, B, 3)


def test_stsgcl_shapes():
    B, T, N, C = 2, 12, 16, 32
    data = torch.randn(B, T, N, C)
    adj = torch.randn(3 * N, 3 * N)
    for module_type in ("sharing", "individual"):
        for activation in ("GLU", "relu"):
            for temporal in (True, False):
                for spatial in (True, False):
                    layer = STSGCL(
                        T, N, C, [3, 3, 3], module_type, activation, temporal, spatial
                    )
                    assert layer(data, adj).shape == (B, 10, N, 3)


def test_output_layer_shapes():
    B, T, N, C = 2, 12, 16, 32
    data = torch.randn(B, T, N, C)
    layer = OutputLayer(N, T, C)
    assert layer(data).shape == (B, 1, N)


def test_huber_loss_shape():
    B, T, N = 2, 12, 16
    data = torch.randn(B, T, N)
    label = torch.randn(B, T, N)
    assert huber_loss(data, label).shape == (B, T, N)


@pytest.mark.parametrize("module_type", ["sharing", "individual"])
@pytest.mark.parametrize("activation", ["GLU", "relu"])
@pytest.mark.parametrize("temporal_emb", [True, False])
@pytest.mark.parametrize("spatial_emb", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_stsgcn_shapes(module_type, activation, temporal_emb, spatial_emb, use_mask):
    B, T, N, C = 2, 12, 32, 32
    adj = np.random.rand(3 * N, 3 * N).astype(np.float32)
    model = STSGCN(
        adj_mx=adj,
        num_of_vertices=N,
        input_length=T,
        num_of_features=C,
        filter_list=[[3, 3, 3], [6, 6, 6], [9, 9, 9]],
        module_type=module_type,
        activation=activation,
        use_mask=use_mask,
        temporal_emb=temporal_emb,
        spatial_emb=spatial_emb,
        first_layer_embedding_size=None,
        num_for_predict=12,
    )
    x = torch.randn(B, T, N, C)
    assert model(x).shape == (B, 12, N)


def test_masked_metrics_exclude_zero_labels():
    from stsgcn_torch.model import _masked_metrics_np

    # (0.0, 2.0) and (0.0, 4.0): only the two non-zero labels are scored.
    pred = np.array([[1.0, 2.0], [3.0, 2.0]])
    label = np.array([[0.0, 2.0], [0.0, 4.0]])
    mae, mape, rmse = _masked_metrics_np(pred, label)
    assert mae == pytest.approx(1.0)  # (|2-2| + |2-4|) / 2
    assert mape == pytest.approx(25.0)  # mean(0, |2-4|/4)
    assert rmse == pytest.approx(2.0**0.5)  # sqrt((0 + 4) / 2)


def test_validation_epoch_end_logs_masked_mae(monkeypatch):
    model = _small_model()
    model._val_preds = [torch.tensor([[[1.0, 2.0]]]), torch.tensor([[[3.0, 2.0]]])]
    model._val_labels = [torch.tensor([[[0.0, 2.0]]]), torch.tensor([[[0.0, 4.0]]])]
    logged = {}
    monkeypatch.setattr(model, "log", lambda name, value, **kw: logged.setdefault(name, value))
    model.on_validation_epoch_end()
    assert logged["val/loss"] == pytest.approx(1.0)


def test_padded_batch_sampler_matches_mxnet():
    from stsgcn_torch.data import _PaddedBatchSampler

    # ndarray_iter pads the last batch with the first samples of the epoch
    assert list(_PaddedBatchSampler(5, 4, shuffle=False)) == [[0, 1, 2, 3], [4, 0, 1, 2]]
    assert len(_PaddedBatchSampler(5, 4, shuffle=False)) == 2

    batches = list(_PaddedBatchSampler(5, 4, shuffle=True))
    assert len(batches) == 2 and all(len(b) == 4 for b in batches)
    # the last batch is the leftover row followed by the epoch's first samples
    assert batches[1][1:] == batches[0][:3]
    assert batches[1][0] not in batches[0]
    assert set(batches[0]) | set(batches[1]) == set(range(5))


def test_train_epoch_end_logs_unmasked_metrics(monkeypatch):
    model = _small_model()
    # MAE is global; RMSE is the mean of per-batch RMSEs (mx.metric semantics).
    model._train_abs_sum, model._train_count = 3.0, 2
    model._train_rmse_sum, model._train_batches = 2.0, 2
    logged = {}
    monkeypatch.setattr(model, "log", lambda name, value, **kw: logged.setdefault(name, value))
    model.on_train_epoch_end()
    assert logged["train/mae"] == pytest.approx(1.5)
    assert logged["train/rmse"] == pytest.approx(1.0)


def test_test_epoch_end_per_horizon_metrics(monkeypatch):
    model = _small_model()
    model.num_for_predict = 2
    model._test_preds = [torch.tensor([[[5.0, 2.0], [5.0, 2.0]]])]
    model._test_labels = [torch.tensor([[[1.0, 2.0], [1.0, 4.0]]])]
    logged = {}
    monkeypatch.setattr(model, "log", lambda name, value, **kw: logged.setdefault(name, value))
    model.on_test_epoch_end()
    # horizon 1: errors 4 and 0 -> MAE 2, MAPE 200%, RMSE sqrt(8)
    assert logged["test/mae_h1"] == pytest.approx(2.0)
    assert logged["test/mape_h1"] == pytest.approx(200.0)
    assert logged["test/rmse_h1"] == pytest.approx(8.0**0.5)
    # horizon 2 (cumulative): errors 4, 0, 4, 2
    assert logged["test/mae_h2"] == pytest.approx(2.5)
    assert logged["test/mae"] == pytest.approx(2.5)  # reported = last horizon
    assert logged["test/mape"] == pytest.approx(212.5)
    assert logged["test/rmse"] == pytest.approx(3.0)
    assert len(model.test_horizon_metrics) == 2


def _small_model():
    from stsgcn_torch import STSGCN

    N = 8
    adj = np.random.rand(3 * N, 3 * N).astype(np.float32)
    return STSGCN(
        adj_mx=adj,
        num_of_vertices=N,
        input_length=12,
        filter_list=[[4, 4, 4]],
        learning_rate=1e-3,
    )


def test_poly_lr_schedule_matches_mxnet():
    model = _small_model()
    # 100 samples, batch 32 -> steps/epoch = 100//32+1 = 4; epochs = 3
    model.set_scheduler_params(num_train_samples=100, batch_size=32, max_epochs=3)
    assert model._scheduler_warmup_steps == 4
    assert model._scheduler_max_update == 12

    opt = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    # MXNet increments num_update before reading the LR, so step i uses sched(i+1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: model._poly_lr_lambda(step + 1)
    )
    lrs = []
    for _ in range(12):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    # linear warmup over the first epoch (lr at update n is base*n/warmup)
    assert lrs[0] == pytest.approx(1e-3 / 4)
    assert lrs[1] == pytest.approx(1e-3 * 2 / 4)
    assert lrs[2] == pytest.approx(1e-3 * 3 / 4)
    assert lrs[3] == pytest.approx(1e-3)  # warmup end
    # poly(pwr=2) decay to 0 at max_update
    assert lrs[4] == pytest.approx(1e-3 * (1 - 1 / 8) ** 2)
    assert lrs[11] == pytest.approx(0.0)
