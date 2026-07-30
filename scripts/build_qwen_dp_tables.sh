#!/usr/bin/env bash
# Build Qwen2.5-7B-Instruct DP lookup tables (same algorithm as Llama; model-specific vocab).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${COVE_PYTHON:-/home/baijiaoyang/anaconda3/bin/python}"
MODEL_PATH="${QWEN_MODEL_PATH:-/home/data/models/Qwen2.5-7B-Instruct}"
CACHE_DIR="${COVE_CACHE_DIR:-$REPO_ROOT/model-storage}"
CORPUS="${COVE_DP_CORPUS:-$REPO_ROOT/PRISM/corpus_sample.txt}"
QWEN_DP_DIR="$REPO_ROOT/PRISM/qwen2.5-7b-instruct"
OUT_LOW="${COVE_QWEN_LOW_FREQ:-$QWEN_DP_DIR/low_freq_words.txt}"
OUT_NPZ="${COVE_QWEN_NEAREST_NPZ:-$QWEN_DP_DIR/nearest_tokens_30.npz}"

mkdir -p "$QWEN_DP_DIR"

cd "$REPO_ROOT"

echo "[1/2] low_freq_words -> $OUT_LOW"
"$PYTHON" PRISM/build_low_freq_words.py 7 \
  --model_path "$MODEL_PATH" \
  --cache_dir "$CACHE_DIR" \
  --corpus_txt "$CORPUS" \
  --output_txt "$OUT_LOW"

echo "[2/2] nearest_tokens_30 -> $OUT_NPZ"
"$PYTHON" scripts/build_qwen_nearest_tokens_npz.py \
  --model_path "$MODEL_PATH" \
  --cache_dir "$CACHE_DIR" \
  --k 30 \
  --output_npz "$OUT_NPZ"

echo "Done."
