#!/usr/bin/env bash
# Inference benchmark for Mistral-7B-Instruct-v0.3 (no zkLLM).
# Usage:
#   ./scripts/run_mistral_inference_benchmark.sh [H20|CSV+4090|CSV+5090|all]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${COVE_PYTHON:-/home/baijiaoyang/anaconda3/bin/python}"
MODEL_REF="${MISTRAL_MODEL_REF:-/home/data/models/Mistral-7B-Instruct-v0.3}"
DEVICE="${COVE_DEVICE:-cuda:0}"
DTYPE="${COVE_DTYPE:-fp16}"
CONFIG_ARG="${1:-H20}"
export COVE_LOW_RAM_LOAD="${COVE_LOW_RAM_LOAD:-1}"

require_model() {
  if [[ ! -f "$MODEL_REF/config.json" ]]; then
    echo "Missing model at $MODEL_REF (need config.json)."
    echo "Download on an online machine, then rsync to this path:"
    echo "  ./scripts/download_mistral_7b_instruct.sh /path/to/Mistral-7B-Instruct-v0.3"
    exit 1
  fi
  if ! compgen -G "$MODEL_REF/model-*.safetensors" >/dev/null; then
    echo "Missing weight shards under $MODEL_REF (model-*.safetensors)."
    exit 1
  fi
}

require_cuda() {
  if ! "$PYTHON" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(1)
print(torch.cuda.get_device_name(0))
PY
  then
    echo "CUDA unavailable (torch.cuda.is_available() is False)."
    echo "On H20, check: sudo systemctl start nvidia-fabricmanager"
    exit 1
  fi
}

run_one() {
  local cfg="$1"
  echo "=== Mistral inference benchmark: config=$cfg model=$MODEL_REF ==="
  if [[ "$cfg" == "H20" ]]; then
    COVE_FORCE_MATH_SDP=1 "$PYTHON" "$REPO_ROOT/evaluation.py" \
      --model_ref "$MODEL_REF" \
      --config "$cfg" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --save-format both
  else
    "$PYTHON" "$REPO_ROOT/evaluation.py" \
      --model_ref "$MODEL_REF" \
      --config "$cfg" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --save-format both
  fi
}

require_model
require_cuda

case "$CONFIG_ARG" in
  all)
    run_one "H20"
    run_one "CSV+4090"
    run_one "CSV+5090"
    ;;
  H20|CSV+4090|CSV+5090)
    run_one "$CONFIG_ARG"
    ;;
  *)
    echo "Unknown config: $CONFIG_ARG (use H20, CSV+4090, CSV+5090, or all)"
    exit 2
    ;;
esac

echo "Results written under $REPO_ROOT/eval-results/$(basename "$MODEL_REF" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._+-]/-/g')/"
