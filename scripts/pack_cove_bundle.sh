#!/usr/bin/env bash
# Pack Cove for copying to another machine (school server / laptop).
# Run ON the H20 worker (10.10.11.162):
#   ./scripts/pack_cove_bundle.sh [inference|full] [output.tar.gz]
#
# inference = benchmark + DP scripts (+ optional small DP tables)
# full      = inference + zk-PIM CUDA sources + LSAGE sources
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${1:-inference}"
OUT="${2:-$HOME/cove-${PROFILE}-$(date +%Y%m%d).tar.gz}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

STAGE="$TMP/Cove"
mkdir -p "$STAGE"

echo "=== Packing Cove profile=$PROFILE -> $OUT ==="

# Core Python (always)
CORE=(
  chat.py
  evaluation.py
  cove_paths.py
  cove_ui_theme.py
  requirements.txt
  README.md
  .gitignore
)
for f in "${CORE[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] && cp "$REPO_ROOT/$f" "$STAGE/"
done

# DP: scripts + sample corpus (not large npz unless INCLUDE_DP_TABLES=1)
mkdir -p "$STAGE/PRISM"
rsync -a --exclude='llama-2-7b-hf/' --exclude='qwen2.5-7b-instruct/' \
  "$REPO_ROOT/PRISM/" "$STAGE/PRISM/"

if [[ "${INCLUDE_DP_TABLES:-1}" == "1" ]]; then
  echo "Including DP lookup tables (low_freq + nearest_tokens npz)..."
  mkdir -p "$STAGE/PRISM/llama-2-7b-hf" "$STAGE/PRISM/qwen2.5-7b-instruct"
  for slug in llama-2-7b-hf qwen2.5-7b-instruct; do
    src="$REPO_ROOT/PRISM/$slug"
    if [[ -d "$src" ]]; then
      cp -a "$src/"* "$STAGE/PRISM/$slug/" 2>/dev/null || true
    fi
  done
fi

# Scripts + plotting code
cp -a "$REPO_ROOT/scripts" "$STAGE/"
mkdir -p "$STAGE/plotting"
rsync -a --exclude='llama-2-7b-hf/' --exclude='qwen2.5-7b-instruct/' --exclude='__pycache__/' \
  "$REPO_ROOT/plotting/" "$STAGE/plotting/"

if [[ "$PROFILE" == "full" ]]; then
  echo "Including zk-PIM + LSAGE sources..."
  rsync -a \
    --exclude='zkllm-workdir/' --exclude='*.bin' --exclude='*.o' \
    --exclude='main' --exclude='ppgen' --exclude='commit-param' \
    --exclude='self-attn' --exclude='ffn' --exclude='rmsnorm' --exclude='skip-connection' \
    --exclude='__pycache__/' \
    "$REPO_ROOT/zk-PIM/" "$STAGE/zk-PIM/"
  rsync -a \
    --exclude='gpu_attest_40xx' --exclude='.gpu_attest_build.json' \
    --exclude='attestation_profile.json' --exclude='__pycache__/' \
    "$REPO_ROOT/LSAGE/" "$STAGE/LSAGE/"
fi

# Empty dirs for local runtime outputs
mkdir -p "$STAGE/eval-results" "$STAGE/plotting/llama-2-7b-hf" "$STAGE/plotting/qwen2.5-7b-instruct"
mkdir -p "$STAGE/zkllm-chat"

cat > "$STAGE/SYNC_README.txt" <<EOF
Cove bundle (profile=$PROFILE)
Packed from: $(hostname) $(date -Iseconds)

On your school server / laptop (after EasyConnect VPN):

  1. Models are NOT included. Rsync weights separately, e.g.:
     rsync -avP user@10.10.11.162:/home/data/models/Llama-2-7b-hf/ /path/to/models/Llama-2-7b-hf/

  2. Install deps:
     pip install -r requirements.txt

  3. Run inference benchmark (edit model path if needed):
     export COVE_PYTHON=\$(which python)
     ./scripts/run_qwen_inference_benchmark.sh CSV+4090
     python evaluation.py --model_ref /path/to/Llama-2-7b-hf --config CSV+4090

  4. School GPU may use fp16; Qwen on H20 needs fp32 (evaluation.py auto-detects H20).

See scripts/pull_from_h20.sh for live rsync instead of this tarball.
EOF

tar -czf "$OUT" -C "$TMP" Cove
ls -lh "$OUT"
echo "Done. Copy to local machine with (run ON 192.168.52.207 after VPN):"
echo "  scp baijiaoyang@10.10.11.162:$OUT ~/"
