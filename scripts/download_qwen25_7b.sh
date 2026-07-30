#!/usr/bin/env bash
# Download Qwen2.5-7B-Instruct on a machine WITH outbound internet.
# Transfer the output directory to the Cove server:
#   rsync -avP ./Qwen2.5-7B-Instruct/ user@server:/home/data/models/Qwen2.5-7B-Instruct/
set -euo pipefail

TARGET="${1:-./Qwen2.5-7B-Instruct}"
mkdir -p "$TARGET"

echo "Target directory: $(realpath "$TARGET")"

download_with_modelscope() {
  python - <<'PY' "$TARGET"
import sys
from modelscope import snapshot_download

target = sys.argv[1]
path = snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct",
    local_dir=target,
)
print("ModelScope download OK:", path)
PY
}

download_with_hf_mirror() {
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  python - <<'PY' "$TARGET"
import os
import sys
from huggingface_hub import snapshot_download

target = sys.argv[1]
path = snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir=target,
)
print("HF mirror download OK:", path)
print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))
PY
}

if python -c "import modelscope" 2>/dev/null; then
  echo "[1/2] Trying ModelScope..."
  download_with_modelscope && exit 0
else
  echo "modelscope not installed; trying: pip install modelscope"
  pip install modelscope -q
  echo "[1/2] Trying ModelScope..."
  download_with_modelscope && exit 0
fi

echo "[2/2] ModelScope failed; trying HF mirror (hf-mirror.com)..."
download_with_hf_mirror

echo "Verifying required files..."
test -f "$TARGET/config.json"
test -f "$TARGET/tokenizer_config.json"
ls "$TARGET"/*.safetensors >/dev/null

echo "Done. Transfer to Cove server:"
echo "  rsync -avP $(realpath "$TARGET")/ USER@HOST:/home/data/models/Qwen2.5-7B-Instruct/"
