"""Dump MXNet STSGCN weights + a reference forward pass for cross-framework comparison.

Run with the mxnet 1.4.1 CPU interpreter (see compare/run_compare.sh).

This script does NOT modify the original STSGCN source; it only imports it.
"""

from __future__ import annotations

import json
import os
import sys

import mxnet as mx
import numpy as np

# Import the original (unmodified) STSGCN source: archive/mxnet/ inside this
# repository, or STSGCN_ORIG_DIR=/path/to/STSGCN for an external checkout.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = (
    os.environ.get("STSGCN_ORIG_DIR"),
    os.path.join(REPO, "archive", "mxnet"),
    os.path.join(os.path.dirname(REPO), "STSGCN"),
)
STSGCN_DIR = next(
    (p for p in _CANDIDATES if p and os.path.isdir(os.path.join(p, "models"))), None
)
if STSGCN_DIR is None:
    raise SystemExit(
        "original STSGCN source not found (expected archive/mxnet/); "
        "set STSGCN_ORIG_DIR=/path/to/STSGCN"
    )
sys.path.insert(0, STSGCN_DIR)

from models.stsgcn import stsgcn
from utils import construct_adj


def build_symbol(config, adj_mx):
    N = config["num_of_vertices"]
    T = config["points_per_hour"]
    P = config["num_for_predict"]
    data = mx.sym.var("data")
    label = mx.sym.var("label")
    adj = mx.sym.Variable(
        "adj", shape=adj_mx.shape, init=mx.init.Constant(value=adj_mx.tolist())
    )
    adj = mx.sym.BlockGrad(adj)
    mask_init_value = mx.init.Constant(
        value=(adj_mx != 0).astype(adj_mx.dtype).tolist()
    )

    fes = config.get("first_layer_embedding_size")
    if fes:
        data = mx.sym.Activation(
            mx.sym.FullyConnected(data, flatten=False, num_hidden=fes),
            act_type="relu",
        )
    else:
        fes = config.get("num_of_features", 1)

    net = stsgcn(
        data,
        adj,
        label,
        T,
        N,
        fes,
        config["filters"],
        config["module_type"],
        config["act_type"],
        config.get("use_mask", True),
        mask_init_value,
        config.get("temporal_emb", True),
        config.get("spatial_emb", True),
        prefix="",
        rho=1,
        predict_length=P,
    )
    return net


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "mxnet_dump.npz"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    randomize = len(sys.argv) > 3 and sys.argv[3] == "--randomize"
    dtype = np.float64 if (len(sys.argv) > 4 and sys.argv[4] == "float64") else np.float32
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    module_type = sys.argv[6] if len(sys.argv) > 6 else "individual"
    activation = sys.argv[7] if len(sys.argv) > 7 else "GLU"
    use_mask = (sys.argv[8] if len(sys.argv) > 8 else "1") == "1"
    N = int(sys.argv[9]) if len(sys.argv) > 9 else 8
    config = {
        "num_of_vertices": N,
        "points_per_hour": 12,
        "num_for_predict": 12,
        "num_of_features": 1,
        "first_layer_embedding_size": 64,
        "filters": [[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]],
        "module_type": module_type,
        "act_type": activation,
        "use_mask": use_mask,
        "temporal_emb": True,
        "spatial_emb": True,
        "batch_size": 4,
    }

    rng = np.random.default_rng(0)
    A = rng.integers(0, 2, size=(N, N)).astype(dtype)
    A = np.triu(A, 1)
    A = A + A.T
    adj_mx = construct_adj(A, 3)

    net = build_symbol(config, adj_mx)
    batch_size = config["batch_size"]

    mod = mx.mod.Module(net, data_names=["data"], label_names=["label"], context=mx.cpu())
    mod.bind(
        data_shapes=[mx.io.DataDesc("data", (batch_size, 12, N, 1), dtype=dtype)],
        label_shapes=[mx.io.DataDesc("label", (batch_size, 12, N), dtype=dtype)],
    )
    mx.random.seed(seed)
    mod.init_params(initializer=mx.init.Xavier(magnitude=0.0003))

    arg_params, aux_params = mod.get_params()

    if randomize:
        rng3 = np.random.default_rng(999)
        for k, v in arg_params.items():
            if k == "adj":
                continue
            if k == "_mask":
                v[:] = rng3.uniform(0.0, 1.0, size=v.shape).astype(v.dtype)
            else:
                v[:] = (scale * rng3.normal(0.0, 1.0, size=v.shape)).astype(v.dtype)
        mod.set_params(arg_params, aux_params)

    # Fixed random input / label.
    rng2 = np.random.default_rng(42)
    x = rng2.normal(0.0, 1.0, size=(batch_size, 12, N, 1)).astype(dtype)
    label = rng2.normal(0.0, 1.0, size=(batch_size, 12, N)).astype(dtype)

    batch = mx.io.DataBatch(data=[mx.nd.array(x)], label=[mx.nd.array(label)])
    mod.forward(batch, is_train=False)
    outputs = mod.get_outputs()
    print("num outputs:", len(outputs))
    for i, o in enumerate(outputs):
        print("output", i, o.shape)

    pred = outputs[1].asnumpy()

    # Save everything.
    np.savez(
        out_path,
        x=x,
        label=label,
        pred=pred,
        adj_mx=adj_mx,
        config=json.dumps(config),
        **{f"arg_{k}": v.asnumpy() for k, v in arg_params.items()},
    )
    with open(out_path + ".names.json", "w") as f:
        json.dump(
            {"arg_params": list(arg_params.keys()), "aux_params": list(aux_params.keys())},
            f,
            indent=2,
        )
    print("saved", out_path)
    print("pred shape", pred.shape)


if __name__ == "__main__":
    main()
