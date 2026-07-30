#!/usr/bin/env bash
# Pull Cove source FROM H20 worker TO your local / school machine.
#
# Run ON your laptop/school server (192.168.52.207) AFTER EasyConnect VPN is connected.
#
#   chmod +x scripts/pull_from_h20.sh
#   ./scripts/pull_from_h20.sh
#   ./scripts/pull_from_h20.sh --profile full --dest ~/work/Cove
#   ./scripts/pull_from_h20.sh --with-dp-tables
#
# Env: H20_HOST=10.10.11.162  H20_USER=baijiaoyang  H20_REMOTE=/home/baijiaoyang/Cove
set -euo pipefail

PROFILE="inference"
LOCAL_DIR="${HOME}/Cove"
WITH_DP_TABLES=0

H20_HOST="${H20_HOST:-10.10.11.162}"
H20_USER="${H20_USER:-baijiaoyang}"
H20_REMOTE="${H20_REMOTE:-/home/baijiaoyang/Cove}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXCLUDES="$SCRIPT_DIR/rsync-excludes.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --dest) LOCAL_DIR="${2:?}"; shift 2 ;;
    --with-dp-tables) WITH_DP_TABLES=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

echo "=== Pull Cove: ${H20_USER}@${H20_HOST}:${H20_REMOTE} -> ${LOCAL_DIR} ==="
echo "Profile=$PROFILE  with_dp_tables=$WITH_DP_TABLES"
echo "(Requires EasyConnect / VPN to 10.10.11.x)"

mkdir -p "$LOCAL_DIR"

RSYNC_OPTS=(-avz --progress)
if [[ -f "$EXCLUDES" ]]; then
  RSYNC_OPTS+=(--exclude-from="$EXCLUDES")
fi
if [[ "$WITH_DP_TABLES" != "1" ]]; then
  RSYNC_OPTS+=(--exclude='PRISM/llama-2-7b-hf/' --exclude='PRISM/qwen2.5-7b-instruct/')
fi
if [[ "$PROFILE" == "full" ]]; then
  RSYNC_OPTS+=(
    --exclude='zk-PIM/zkllm-workdir/'
    --exclude='LSAGE/gpu_attest_40xx'
    --exclude='LSAGE/attestation_profile.json'
  )
else
  RSYNC_OPTS+=(--exclude='zk-PIM/' --exclude='LSAGE/')
fi

rsync "${RSYNC_OPTS[@]}" "${H20_USER}@${H20_HOST}:${H20_REMOTE}/" "${LOCAL_DIR}/"

mkdir -p "$LOCAL_DIR/eval-results" "$LOCAL_DIR/zkllm-chat"

echo ""
echo "Done -> $LOCAL_DIR"
echo "  pip install -r $LOCAL_DIR/requirements.txt"
echo "  # Models (separate, large):"
echo "  rsync -avP ${H20_USER}@${H20_HOST}:/home/data/models/Llama-2-7b-hf/ /your/models/Llama-2-7b-hf/"
