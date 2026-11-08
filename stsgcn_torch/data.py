"""LightningDataModule for STSGCN."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset

from .utils import split_series


def _to_tensor(arr: np.ndarray, dtype=None) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=dtype))


class _PaddedBatchSampler(Sampler[list[int]]):
    """Batch sampler replicating ``mx.io.NDArrayIter(last_batch_handle="pad")``.

    Like the MXNet iterator the indices are reshuffled every epoch, and an
    incomplete last batch is completed with the first samples of that epoch
    (duplicated), so every optimizer step sees exactly ``batch_size`` rows.
    """

    def __init__(self, num_samples: int, batch_size: int, shuffle: bool = True):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        order = (
            torch.randperm(self.num_samples)
            if self.shuffle
            else torch.arange(self.num_samples)
        )
        for start in range(0, self.num_samples, self.batch_size):
            batch = order[start : start + self.batch_size]
            if len(batch) < self.batch_size:
                batch = torch.cat([batch, order[: self.batch_size - len(batch)]])
            yield batch.tolist()

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class STSGCNDataModule(pl.LightningDataModule):
    """Feeds ``(x, y)`` pairs from the PEMS-STSGCN parquet dataset."""

    def __init__(
        self,
        hub_dataset: str,
        hub_source: str = "ms",
        hub_repo: str | None = None,
        hub_root: str | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.hub_dataset = hub_dataset
        self.hub_source = hub_source
        self.hub_repo = hub_repo
        self.hub_root = hub_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "train_x"):  # idempotent: data generation is expensive
            return
        from . import hub

        data = hub.load_series(
            self.hub_dataset, source=self.hub_source, repo=self.hub_repo, root=self.hub_root
        )
        train, val, test = split_series(data)
        # x is float32 for the model; keep the targets' original dtype (the
        # parquet stores float64) so evaluation matches MXNet's val_y / test_y.
        self.train_x = _to_tensor(train[0], np.float32)
        self.train_y = _to_tensor(train[1])
        self.val_x = _to_tensor(val[0], np.float32)
        self.val_y = _to_tensor(val[1])
        self.test_x = _to_tensor(test[0], np.float32)
        self.test_y = _to_tensor(test[1])

    def _loader(self, x: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
        common = {
            "num_workers": self.num_workers,
            "persistent_workers": self.num_workers > 0,
        }
        dataset = TensorDataset(x, y)
        if shuffle:
            return DataLoader(
                dataset,
                batch_sampler=_PaddedBatchSampler(len(x), self.batch_size, shuffle=True),
                **common,
            )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, **common)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_x, self.train_y, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_x, self.val_y, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_x, self.test_y, shuffle=False)

