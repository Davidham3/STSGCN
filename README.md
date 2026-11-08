# STSGCN (PyTorch)

> **Update**
>
> - The previous MXNet implementation of the paper now lives in
>   [`archive/mxnet/`](archive/mxnet), unchanged. Everything else in this
>   repository is a new **PyTorch** (Lightning) implementation.
> - The dataset (PEMS03/04/07/08, Parquet) is public on
>   [HuggingFace](https://huggingface.co/datasets/Davidham3/pems-stsgcn) and
>   [ModelScope](https://modelscope.cn/datasets/Davidham3/pems-stsgcn); the code
>   downloads it automatically on first use.
> - Prerequisites: [`uv`](https://docs.astral.sh/uv/) and a GPU driver with
>   CUDA 12 support.
>
> ```bash
> # ModelScope (the default in every config)
> uv sync --extra ms
> uv run python scripts/train.py fit --config config/PEMS04/individual_GLU_mask_emb.yaml
>
> # HuggingFace
> uv sync --extra hf
> uv run python scripts/train.py fit --config config/PEMS04/individual_GLU_mask_emb.yaml \
>     --model.hub_source hf --data.hub_source hf
> ```

AAAI 2020. Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting

url: paper/AAAI2020-STSGCN.pdf

**The original MXNet sources now live in [`archive/mxnet/`](archive/mxnet); the code at the repository root is a PyTorch reimplementation that follows them exactly (same model, training loop and evaluator) and reproduces the paper's results.**

# Usage

Requires [uv](https://docs.astral.sh/uv/) and a GPU driver with CUDA 12 support. The dataset is downloaded automatically on first use.

1. install [uv](https://docs.astral.sh/uv/)
2. install the dependencies using `uv sync --extra ms`
3. run code using `uv run python scripts/train.py fit --config config/PEMS04/individual_GLU_mask_emb.yaml`

Use `uv sync --extra hf` and append `--model.hub_source hf --data.hub_source hf` to load the dataset from HuggingFace instead of ModelScope. The docker workflow of the MXNet version is kept in [`archive/mxnet/README.md`](archive/mxnet/README.md).

# repo structure
name|description
-|-
archive/mxnet|the original MXNet implementation (reference, unmodified)
compare|MXNet-vs-PyTorch forward parity harness
config|configurations of STSGCN
scripts|code of training STSGCN
stsgcn_torch|core of STSGCN
tests|pytest files
pyproject.toml|python packages requirements
