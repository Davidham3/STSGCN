#!/usr/bin/env bash
# Cross-framework comparison: MXNet 1.4.1 (CPU) vs PyTorch Lightning.
#
# Requires a Python 3.8 interpreter with mxnet==1.4.1. The original STSGCN
# source is taken from archive/mxnet/ (or STSGCN_ORIG_DIR, or a sibling STSGCN/
# directory):
#
#   MXNET_PY=/path/to/mxnet-py38/bin/python bash compare/run_compare.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MXNET_PY="${MXNET_PY:-python3}"

echo "==> MXNet python: ${MXNET_PY}"
"${MXNET_PY}" -c "import mxnet; print('mxnet', mxnet.__version__)"

ART_DIR="${SCRIPT_DIR}/parity_artifacts"
mkdir -p "${ART_DIR}"
echo "==> artifacts -> ${ART_DIR}"

# (label, module_type, activation, use_mask, scale, dtype)
COMBOS=(
  "individual_GLU_mask   individual GLU 1 0.05 float32"
  "individual_relu_mask  individual relu 1 0.05 float32"
  "sharing_GLU_mask      sharing   GLU 1 0.05 float32"
  "sharing_relu_mask     sharing   relu 1 0.05 float32"
  "individual_GLU_nomask individual GLU 0 0.05 float32"
  "individual_GLU_mask64 individual GLU 1 1.0  float64"
)

for combo in "${COMBOS[@]}"; do
  read -r name module_type activation use_mask scale dtype <<<"$combo"
  fname="${ART_DIR}/mxnet_dump_${name}.npz"
  echo ""
  echo "===== ${name} (${module_type}/${activation}, mask=${use_mask}, scale=${scale}, dtype=${dtype}) ====="
  "${MXNET_PY}" "${SCRIPT_DIR}/mxnet_dump.py" "$fname" 1234 --randomize "$dtype" "$scale" \
      "$module_type" "$activation" "$use_mask" 2>&1 | grep -v "SyntaxWarning\|num_device" | tail -1
  (cd "${PROJECT_DIR}" && uv run python compare/torch_compare.py "compare/parity_artifacts/mxnet_dump_${name}.npz")
done

echo ""
echo "Done."
