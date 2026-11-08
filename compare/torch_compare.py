"""Load MXNet-dumped weights into the PyTorch model and compare forward outputs.

Run with the torch interpreter (see compare/run_compare.sh).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from stsgcn_torch import STSGCN


def load_mxnet_params(model: STSGCN, arg_params: dict) -> int:
    """Copy MXNet arg_params into the torch model; returns number of FC layers mapped."""
    fc_idx = 0
    if model.first_embedding is not None:
        model.first_embedding.weight.data.copy_(
            torch.from_numpy(arg_params[f"fullyconnected{fc_idx}_weight"])
        )
        model.first_embedding.bias.data.copy_(
            torch.from_numpy(arg_params[f"fullyconnected{fc_idx}_bias"])
        )
        fc_idx += 1

    for i, stsgcl in enumerate(model.stsgcl_layers):
        pe = stsgcl.position_embedding
        if pe.temporal:
            pe.t_emb.data.copy_(torch.from_numpy(arg_params[f"_stsgcl_{i}_emb_t_emb"]))
        if pe.spatial:
            pe.v_emb.data.copy_(torch.from_numpy(arg_params[f"_stsgcl_{i}_emb_v_emb"]))
        stsgcms = stsgcl.stsgcms if stsgcl.module_type == "individual" else [stsgcl.stsgcm]
        for stsgcm in stsgcms:
            for gcn in stsgcm.gcn_layers:
                gcn.linear.weight.data.copy_(
                    torch.from_numpy(arg_params[f"fullyconnected{fc_idx}_weight"])
                )
                gcn.linear.bias.data.copy_(
                    torch.from_numpy(arg_params[f"fullyconnected{fc_idx}_bias"])
                )
                fc_idx += 1

    for head in model.output_heads:
        head.fc1.weight.data.copy_(
            torch.from_numpy(arg_params[f"fullyconnected{fc_idx}_weight"])
        )
        head.fc1.bias.data.copy_(
            torch.from_numpy(arg_params[f"fullyconnected{fc_idx}_bias"])
        )
        fc_idx += 1
        head.fc2.weight.data.copy_(
            torch.from_numpy(arg_params[f"fullyconnected{fc_idx}_weight"])
        )
        head.fc2.bias.data.copy_(
            torch.from_numpy(arg_params[f"fullyconnected{fc_idx}_bias"])
        )
        fc_idx += 1

    if model.use_mask:
        model.mask.data.copy_(torch.from_numpy(arg_params["_mask"]))
    return fc_idx


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)

    dump_path = sys.argv[1] if len(sys.argv) > 1 else "mxnet_dump.npz"
    data = np.load(dump_path, allow_pickle=False)
    config = json.loads(data["config"].item())
    adj_mx = data["adj_mx"]
    x_np = data["x"]
    mx_pred = data["pred"]

    use_double = x_np.dtype == np.float64

    model = STSGCN(
        adj_mx=adj_mx,
        num_of_vertices=config["num_of_vertices"],
        input_length=config["points_per_hour"],
        num_of_features=config.get("num_of_features", 1),
        filter_list=config["filters"],
        module_type=config["module_type"],
        activation=config["act_type"],
        use_mask=config.get("use_mask", True),
        temporal_emb=config.get("temporal_emb", True),
        spatial_emb=config.get("spatial_emb", True),
        first_layer_embedding_size=config.get("first_layer_embedding_size"),
        num_for_predict=config["num_for_predict"],
    )
    if use_double:
        model = model.double()

    arg_params = {k[4:]: data[k] for k in data.files if k.startswith("arg_")}
    n_fc = load_mxnet_params(model, arg_params)
    print(f"mapped {n_fc} FC layers")

    # Verify adjacency matches.
    adj_torch = model.adj.detach().numpy()
    adj_mxnet = arg_params["adj"]
    print("adj max abs diff:", float(np.abs(adj_torch - adj_mxnet).max()))

    x = torch.from_numpy(x_np)
    if use_double:
        x = x.double()
    model.eval()
    with torch.no_grad():
        torch_pred = model(x).numpy()

    diff = np.abs(torch_pred - mx_pred)
    max_abs = float(diff.max())
    max_rel = float((diff / (np.abs(mx_pred) + 1e-6)).max())
    print("pred shape:", torch_pred.shape)
    print(f"pred magnitude: max={float(np.abs(mx_pred).max()):.3e} mean={float(np.abs(mx_pred).mean()):.3e}")
    print("max abs diff:", max_abs)
    print("mean abs diff:", float(diff.mean()))
    print("rel diff (max):", max_rel)
    print("mxnet pred sample:", mx_pred[0, 0, :5])
    print("torch pred sample:", torch_pred[0, 0, :5])

    # Magnitude-aware verdict: pass if either the absolute error (near-zero
    # outputs) or the relative error (large outputs) is within float32 noise.
    atol, rtol = 1e-4, 1e-4
    if max_abs < atol or max_rel < rtol:
        print(f"RESULT: MATCH (max_abs={max_abs:.3e} < atol={atol} "
              f"or max_rel={max_rel:.3e} < rtol={rtol})")
    else:
        print(f"RESULT: MISMATCH (max_abs={max_abs:.3e} >= atol={atol} "
              f"and max_rel={max_rel:.3e} >= rtol={rtol})")


if __name__ == "__main__":
    main()
