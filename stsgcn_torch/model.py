"""PyTorch Lightning reimplementation of STSGCN (AAAI 2020).

The forward pass mirrors the original MXNet symbol graph in
``STSGCN/models/stsgcn.py`` one operator at a time:

* Input layout is kept identical: ``(B, T, N, C)``.
* The graph "convolution" is the MXNet ``dot(adj(3N,3N), data(3N,B,C))``
  contraction (sum over the *first* axis of the data), implemented here as an
  explicit matmul against a flattened ``(3N, B*C)`` view.
* ``module_type`` supports ``individual`` and ``sharing`` STSGCMs.
* ``act_type`` supports ``GLU`` and ``relu``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .utils import (
    construct_adj,
    masked_mae_np,
    masked_mape_np,
    masked_mse_np,
)


def mxnet_xavier_(tensor: torch.Tensor, magnitude: float = 0.0003) -> None:
    """Replicate ``mx.init.Xavier(magnitude=0.0003)`` (uniform, factor_type='avg').

    MXNet computes ``scale = sqrt(magnitude / ((fan_in + fan_out) / 2))`` with
    ``fan_in = shape[1] * prod(shape[2:])`` and ``fan_out = shape[0] * prod(shape[2:])``.
    """
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Xavier initializer cannot be applied to 1-D tensors")
    hw_scale = float(np.prod(shape[2:])) if len(shape) > 2 else 1.0
    fan_in = shape[1] * hw_scale
    fan_out = shape[0] * hw_scale
    scale = float(np.sqrt(magnitude / ((fan_in + fan_out) / 2.0)))
    with torch.no_grad():
        tensor.uniform_(-scale, scale)


def _init_linear(layer: nn.Linear) -> None:
    """MXNet defaults for a ``FullyConnected``: Xavier weight, zero bias."""
    mxnet_xavier_(layer.weight)
    nn.init.zeros_(layer.bias)


def _gcn_dot(adj: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """MXNet ``dot(adj(3N,3N), data(3N,B,C))`` -> ``(3N, B, C)``."""
    B, C = data.shape[1], data.shape[2]
    return torch.matmul(adj, data.reshape(data.shape[0], -1)).reshape(-1, B, C)


class PositionEmbedding(nn.Module):
    """Broadcast-add learnable temporal / spatial embeddings."""

    def __init__(self, T: int, N: int, C: int, temporal: bool, spatial: bool):
        super().__init__()
        self.temporal = temporal
        self.spatial = spatial
        if temporal:
            self.t_emb = nn.Parameter(torch.empty(1, T, 1, C))
        if spatial:
            self.v_emb = nn.Parameter(torch.empty(1, 1, N, C))
        self.reset_parameters()

    def reset_parameters(self):
        if self.temporal:
            mxnet_xavier_(self.t_emb)
        if self.spatial:
            mxnet_xavier_(self.v_emb)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.temporal:
            data = data + self.t_emb
        if self.spatial:
            data = data + self.v_emb
        return data


class GCNOperation(nn.Module):
    """Single GCN step: ``dot(adj, data)`` followed by GLU or relu."""

    def __init__(self, num_of_filter: int, num_of_features: int, activation: str):
        super().__init__()
        assert activation in {"GLU", "relu"}
        self.activation = activation
        out_features = 2 * num_of_filter if activation == "GLU" else num_of_filter
        self.linear = nn.Linear(num_of_features, out_features)
        _init_linear(self.linear)

    def forward(self, data: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        data = _gcn_dot(adj, data)
        if self.activation == "GLU":
            data = self.linear(data)
            lhs, rhs = torch.chunk(data, 2, dim=-1)
            return lhs * torch.sigmoid(rhs)
        return F.relu(self.linear(data))


class STSGCM(nn.Module):
    """Stacked GCN layers with middle-crop + max pooling."""

    def __init__(
        self,
        filters: Sequence[int],
        num_of_features: int,
        num_of_vertices: int,
        activation: str,
    ):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.gcn_layers = nn.ModuleList()
        in_features = num_of_features
        for f in filters:
            self.gcn_layers.append(GCNOperation(f, in_features, activation))
            in_features = f

    def forward(self, data: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = self.num_of_vertices
        outputs = []
        for gcn in self.gcn_layers:
            data = gcn(data, adj)  # (3N, B, C')
            outputs.append(data[N : 2 * N].unsqueeze(0))  # (1, N, B, C')
        return torch.max(torch.cat(outputs, dim=0), dim=0).values  # (N, B, C')


class STSGCL(nn.Module):
    """One STSGCL layer (individual or sharing)."""

    def __init__(
        self,
        T: int,
        num_of_vertices: int,
        num_of_features: int,
        filters: Sequence[int],
        module_type: str,
        activation: str,
        temporal_emb: bool,
        spatial_emb: bool,
    ):
        super().__init__()
        assert module_type in {"sharing", "individual"}
        self.module_type = module_type
        self.T = T
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = list(filters)
        self.position_embedding = PositionEmbedding(
            T, num_of_vertices, num_of_features, temporal_emb, spatial_emb
        )
        if module_type == "individual":
            self.stsgcms = nn.ModuleList(
                [
                    STSGCM(filters, num_of_features, num_of_vertices, activation)
                    for _ in range(T - 2)
                ]
            )
        else:
            self.stsgcm = STSGCM(filters, num_of_features, num_of_vertices, activation)

    def forward(self, data: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B = data.shape[0]
        N = self.num_of_vertices
        C = self.num_of_features
        data = self.position_embedding(data)
        T = self.T

        # Each window of 3 consecutive steps is flattened to (3N, B, C).
        windows = [
            data[:, i : i + 3].reshape(B, 3 * N, C).transpose(0, 1) for i in range(T - 2)
        ]

        if self.module_type == "individual":
            outputs = [
                self.stsgcms[i](t, adj).transpose(0, 1).unsqueeze(1)
                for i, t in enumerate(windows)
            ]
            return torch.cat(outputs, dim=1)  # (B, T-2, N, C')

        t = self.stsgcm(torch.cat(windows, dim=1), adj)  # (N, (T-2)*B, C')
        t = t.reshape(N, T - 2, B, self.filters[-1])  # (N, T-2, B, C')
        return t.transpose(0, 2)  # (B, T-2, N, C')


class OutputLayer(nn.Module):
    """Per-horizon output head: ``(B, T, N, C) -> (B, 1, N)``."""

    def __init__(self, num_of_vertices: int, input_length: int, num_of_features: int):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        in_features = input_length * num_of_features
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 1)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        B = data.shape[0]
        N = self.num_of_vertices
        x = data.transpose(1, 2)  # (B, N, T, C)
        x = x.reshape(B, N, self.input_length * self.num_of_features)  # (B, N, T*C)
        x = F.relu(self.fc1(x))  # (B, N, 128)
        x = self.fc2(x)  # (B, N, 1)
        return x.transpose(1, 2)  # (B, 1, N)


def huber_loss(data: torch.Tensor, label: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
    """Element-wise smooth L1-style Huber loss matching the MXNet implementation."""
    loss = torch.abs(data - label)
    loss = torch.where(loss > rho, loss - 0.5 * rho, (0.5 / rho) * loss.square())
    return loss


def _masked_metrics_np(pred: np.ndarray, label: np.ndarray) -> tuple[float, float, float]:
    """MAE, MAPE (%), RMSE with labels equal to 0 excluded.

    Reuses the MXNet-mirroring ``masked_*_np(..., null_val=0)`` helpers, so the
    numbers match the original ``STSGCN/main.py`` validation / test evaluation.
    """
    mae = masked_mae_np(label, pred, 0)
    mape = masked_mape_np(label, pred, 0)
    rmse = masked_mse_np(label, pred, 0) ** 0.5
    return mae, mape, rmse


class STSGCN(pl.LightningModule):
    """Full STSGCN model.

    Parameters mirror the MXNet ``config/*.json`` keys.
    """

    def __init__(
        self,
        num_of_vertices: int,
        adj_mx=None,
        hub_dataset: str | None = None,
        hub_source: str = "ms",
        hub_repo: str | None = None,
        hub_root: str | None = None,
        input_length: int = 12,
        num_of_features: int = 1,
        filter_list: list[list[int]] | None = None,
        module_type: str = "individual",
        activation: str = "GLU",
        use_mask: bool = True,
        temporal_emb: bool = True,
        spatial_emb: bool = True,
        first_layer_embedding_size: int | None = 64,
        num_for_predict: int = 12,
        learning_rate: float = 1e-3,
        rho: float = 1.0,
        max_update_factor: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["adj_mx"])

        self.num_for_predict = num_for_predict
        self.learning_rate = learning_rate
        self.rho = rho
        self.max_update_factor = max_update_factor

        # Set via set_scheduler_params before training.
        self._scheduler_warmup_steps: int | None = None
        self._scheduler_max_update: int | None = None

        # Predictions are accumulated over the whole split, so metrics are
        # computed on the full arrays (and zero labels masked) like MXNet.
        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._test_preds: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

        # Unmasked training metrics (mirrors mx.metric ['RMSE', 'MAE']).
        self._train_abs_sum = 0.0
        self._train_rmse_sum = 0.0
        self._train_count = 0
        self._train_batches = 0
        # Cumulative 1..H test metrics (mirrors main.py's tmp_info per horizon).
        self.test_horizon_metrics: list[tuple[float, float, float]] = []

        if adj_mx is None:
            if hub_dataset is not None:
                from . import hub

                adj = hub.load_adjacency(
                    hub_dataset,
                    num_of_vertices,
                    source=hub_source,
                    repo=hub_repo,
                    root=hub_root,
                )
                adj_mx = construct_adj(adj, 3)
            else:
                raise ValueError("either adj_mx or hub_dataset is required")

        # Fixed localized adjacency matrix (mirrors mx.sym.BlockGrad(adj)).
        adj = torch.from_numpy(np.asarray(adj_mx, dtype=np.float32))
        self.register_buffer("adj", adj)

        # Learnable mask (mirrors mx.sym.var(..., init=Constant(adj != 0))).
        self.use_mask = use_mask
        if use_mask:
            mask_init = torch.from_numpy(
                (np.asarray(adj_mx) != 0).astype(np.float32)
            )
            self.mask = nn.Parameter(mask_init)
        else:
            self.register_parameter("mask", None)

        # First-layer embedding (a 1x1 conv / FC on the feature axis).
        self.first_layer_embedding_size = first_layer_embedding_size
        if first_layer_embedding_size:
            self.first_embedding = nn.Linear(num_of_features, first_layer_embedding_size)
            in_features = first_layer_embedding_size
        else:
            self.first_embedding = None
            in_features = num_of_features

        # STSGCL stack.
        filter_list = filter_list or [[64, 64, 64]] * 4
        self.filter_list = [list(f) for f in filter_list]
        T = input_length
        self.stsgcl_layers = nn.ModuleList()
        for filters in self.filter_list:
            self.stsgcl_layers.append(
                STSGCL(
                    T=T,
                    num_of_vertices=num_of_vertices,
                    num_of_features=in_features,
                    filters=filters,
                    module_type=module_type,
                    activation=activation,
                    temporal_emb=temporal_emb,
                    spatial_emb=spatial_emb,
                )
            )
            T -= 2
            in_features = filters[-1]

        # Output heads: one independent head per predicted horizon.
        self.output_heads = nn.ModuleList(
            [OutputLayer(num_of_vertices, T, in_features) for _ in range(num_for_predict)]
        )

        if self.first_embedding is not None:
            _init_linear(self.first_embedding)

    def _effective_adj(self) -> torch.Tensor:
        if self.use_mask:
            return self.mask * self.adj
        return self.adj

    def set_scheduler_params(self, num_train_samples: int, batch_size: int, max_epochs: int) -> None:
        """Replicate ``STSGCN/main.py``: warmup = 1 epoch, poly decay after."""
        global_train_steps = num_train_samples // batch_size + 1
        self._scheduler_warmup_steps = global_train_steps
        self._scheduler_max_update = int(global_train_steps * max_epochs * self.max_update_factor)

    def _poly_lr_lambda(self, step: int) -> float:
        """mx.lr_scheduler.PolyScheduler(pwr=2, linear warmup from 0, final_lr=0)."""
        warmup = self._scheduler_warmup_steps or 0
        max_update = self._scheduler_max_update or 1
        if step < warmup:
            return step / warmup if warmup > 0 else 1.0
        if step <= max_update:
            return (1.0 - (step - warmup) / max(max_update - warmup, 1)) ** 2
        return 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C)
        if self.first_embedding is not None:
            x = F.relu(self.first_embedding(x))
        adj = self._effective_adj()
        for layer in self.stsgcl_layers:
            x = layer(x, adj)
        outputs = [head(x) for head in self.output_heads]
        return torch.cat(outputs, dim=1)  # (B, num_for_predict, N)

    def _forward_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        return self(x), y.squeeze(-1)  # pred, label: (B, T', N)

    def on_train_epoch_start(self) -> None:
        self._train_abs_sum = 0.0
        self._train_rmse_sum = 0.0
        self._train_count = 0
        self._train_batches = 0

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        pred, y = self._forward_batch(batch)
        # MXNet MakeLoss defaults to normalization='null': the objective is the
        # *sum* over all elements, not the mean, and it is not zero-masked.
        loss = huber_loss(pred, y.to(pred.dtype), self.rho).sum()
        self.log("train/loss", loss, prog_bar=True)
        with torch.no_grad():
            diff = pred.detach().double() - y.double()
            self._train_abs_sum += diff.abs().sum().item()
            self._train_count += diff.numel()
            # mx.metric.RMSE averages per-batch RMSEs (see metric.py update()).
            self._train_rmse_sum += diff.square().mean().sqrt().item()
            self._train_batches += 1
        optimizer = self.optimizers()
        if optimizer is not None:
            self.log("train/lr", optimizer.param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        # mx.metric ['RMSE', 'MAE'] over the epoch, unmasked (padded samples included).
        if self._train_count == 0:
            return
        self.log("train/mae", self._train_abs_sum / self._train_count, prog_bar=True)
        self.log("train/rmse", self._train_rmse_sum / self._train_batches, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_labels.clear()

    def validation_step(self, batch, batch_idx: int) -> None:
        pred, y = self._forward_batch(batch)
        self._val_preds.append(pred.detach().cpu())
        self._val_labels.append(y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        # MXNet: ``masked_mae_np(val_y, prediction, 0)`` drives the checkpoint.
        pred = torch.cat(self._val_preds).numpy()
        label = torch.cat(self._val_labels).numpy()
        mae, _, _ = _masked_metrics_np(pred, label)
        self.log("val/loss", mae, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self._test_preds.clear()
        self._test_labels.clear()

    def test_step(self, batch, batch_idx: int) -> None:
        pred, y = self._forward_batch(batch)
        self._test_preds.append(pred.detach().cpu())
        self._test_labels.append(y.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        pred = torch.cat(self._test_preds).numpy()
        label = torch.cat(self._test_labels).numpy()
        # main.py evaluates cumulative horizons 1..H and reports the H-step values
        # (tmp_info[-1]); the intermediate per-horizon values are kept as well.
        self.test_horizon_metrics = [
            _masked_metrics_np(pred[:, :h], label[:, :h])
            for h in range(1, self.num_for_predict + 1)
        ]
        for h, (mae, mape, rmse) in enumerate(self.test_horizon_metrics, start=1):
            self.log(f"test/mae_h{h}", mae)
            self.log(f"test/mape_h{h}", mape)
            self.log(f"test/rmse_h{h}", rmse)
        mae, mape, rmse = self.test_horizon_metrics[-1]
        # main.py reports masked MAE as the test "loss"; huber is training-only.
        self.log("test/loss", mae)
        self.log("test/mae", mae)
        self.log("test/mape", mape)
        self.log("test/rmse", rmse)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self._scheduler_max_update is None:
            return optimizer
        # MXNet increments num_update before reading the LR, so optimizer step i
        # uses PolyScheduler(i + 1); LambdaLR evaluates lambda(step) with step=i.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: self._poly_lr_lambda(step + 1)
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
