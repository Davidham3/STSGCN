# scripts/

`train.py` is the training entry point (Lightning CLI). The PEMS-STSGCN Parquet
dataset is published on
[HuggingFace](https://huggingface.co/datasets/Davidham3/pems-stsgcn) and
[ModelScope](https://modelscope.cn/datasets/Davidham3/pems-stsgcn), and the code
downloads it automatically: every config loads it directly
(`hub_dataset: PEMS0x`, `hub_source: ms`); the legacy `.npz` / `.csv` files are
not supported anymore.

```bash
uv sync --extra ms

# ModelScope (default; repo Davidham3/pems-stsgcn)
uv run python scripts/train.py fit --config config/PEMS08/individual_GLU_mask_emb.yaml

# fully offline against an already materialised dataset directory
uv run python scripts/train.py fit --config config/PEMS08/individual_GLU_mask_emb.yaml \
    --model.hub_source local --model.hub_root /path/to/pems-stsgcn \
    --data.hub_source local --data.hub_root /path/to/pems-stsgcn

# HuggingFace (mirror via HF_ENDPOINT, e.g. https://hf-mirror.com)
uv sync --extra hf
export HF_ENDPOINT=https://hf-mirror.com
uv run python scripts/train.py fit --config config/PEMS08/individual_GLU_mask_emb.yaml \
    --model.hub_source hf --data.hub_source hf \
    --model.hub_root ~/.cache/pems-stsgcn --data.hub_root ~/.cache/pems-stsgcn
```

`huggingface_hub` reads `HF_ENDPOINT` / `HF_HOME` when it is imported, so export
them before launching. `hub_root` is the dataset directory for `local` and the
download cache for `hf` / `ms` (default: the respective Hub cache). The repo id
defaults to `Davidham3/pems-stsgcn` and can be overridden with
`STSGCN_MS_REPO` / `STSGCN_HF_REPO` or the `hub_repo` constructor argument.

The dataset directory layout is:

```
pems-stsgcn/
├── PEMS03/{PEMS03_series, PEMS03_edges, PEMS03_sensors}.parquet
├── PEMS04/{PEMS04_series, PEMS04_edges}.parquet
├── PEMS07/{PEMS07_series, PEMS07_edges}.parquet
└── PEMS08/{PEMS08_series, PEMS08_edges}.parquet
```
