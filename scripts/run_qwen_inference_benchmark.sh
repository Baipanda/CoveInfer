#!/usr/bin/env bash
# Inference-only benchmark for Qwen2.5-7B-Instruct (no zkLLM).
# Usage:
#   ./scripts/run_qwen_inference_benchmark.sh [H20|CSV+4090|CSV+5090|all]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${COVE_PYTHON:-/home/baijiaoyang/anaconda3/bin/python}"
MODEL_REF="${QWEN_MODEL_REF:-/home/data/models/Qwen2.5-7B-Instruct}"
DEVICE="${COVE_DEVICE:-cuda:0}"
# H20 + Qwen fp16/bf16 SIGFPE; evaluation.py auto-upgrades to fp32 on H20.
DTYPE="${COVE_DTYPE:-fp16}"
CONFIG_ARG="${1:-all}"
# Host has ~14GB RAM; Qwen7B default HF load OOMs — stream safetensors to GPU.
export COVE_LOW_RAM_LOAD="${COVE_LOW_RAM_LOAD:-1}"

require_model() {
  if [[ ! -f "$MODEL_REF/config.json" ]]; then
    echo "Missing model at $MODEL_REF (need config.json)."
    echo "Download on an online machine, then rsync to this path."
    exit 1
  fi
  if ! compgen -G "$MODEL_REF/*.safetensors" >/dev/null; then
    echo "Missing weight shards under $MODEL_REF (*.safetensors)."
    exit 1
  fi
}

run_one() {
  local cfg="$1"
  echo "=== Qwen inference benchmark: config=$cfg model=$MODEL_REF ==="
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

echo "Results written under $REPO_ROOT/eval-results/qwen2.5-7b-instruct/"
