#!/usr/bin/env bash
# Push a model from LOCAL HF cache to AutoDL via scp.
# This is the *recommended* path for gated models when you already have them cached
# locally — bypasses unreliable AutoDL proxy entirely. ~5 MB/s sustained (FTTH bottleneck).
#
# Run this on YOUR LOCAL DEV MACHINE, not on AutoDL.
#
# Usage:
#   SSHPASS=xxx bash scp_upload_model.sh nvidia/Cosmos-Reason2-2B [autodl_user@host] [port]
#
# Examples:
#   SSHPASS=qV+/abc bash scp_upload_model.sh nvidia/Cosmos-Reason2-2B \
#       root@connect.westd.seetacloud.com 13330

set -euo pipefail

REPO_ID="${1:?usage: SSHPASS=xxx bash $0 <repo_id> [user@host] [port]}"
USER_HOST="${2:-root@connect.westd.seetacloud.com}"
PORT="${3:-22}"
REPO_NAME="${REPO_ID##*/}"
TARGET_DIR="/root/autodl-tmp/${REPO_NAME,,}_raw"

REPO_CACHE_SLUG="${REPO_ID//\//--}"
LOCAL_SNAPSHOTS="${HOME}/.cache/huggingface/hub/models--${REPO_CACHE_SLUG}/snapshots"
if [[ ! -d "$LOCAL_SNAPSHOTS" ]]; then
    echo "[scp_upload] ERROR: no local HF cache for $REPO_ID at $LOCAL_SNAPSHOTS" >&2
    echo "  → first run: hf download $REPO_ID  (on local dev box)" >&2
    exit 1
fi

# pick latest snapshot
SNAPSHOT_DIR=$(ls -dt "$LOCAL_SNAPSHOTS"/*/ | head -1)
echo "[scp_upload] local source: $SNAPSHOT_DIR"
echo "[scp_upload] remote target: $USER_HOST:$TARGET_DIR (port $PORT)"

if ! command -v sshpass >/dev/null 2>&1; then
    echo "[scp_upload] ERROR: sshpass not installed (apt install sshpass)" >&2
    exit 1
fi

# mkdir remote
sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p "$PORT" "$USER_HOST" "mkdir -p $TARGET_DIR"

# scp -r resolves symlinks (HF cache files are symlinks to blobs), so we get real content
echo "[scp_upload] starting scp (4.6 GB Cosmos-Reason2-2B ≈ 15 min @ 5 MB/s)"
time sshpass -e scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -P "$PORT" -r "$SNAPSHOT_DIR"* "$USER_HOST:$TARGET_DIR/"

# verify md5 of largest file (model.safetensors) matches
LARGEST=$(find "$SNAPSHOT_DIR" -type l -name "*.safetensors" | head -1)
if [[ -n "$LARGEST" ]]; then
    LOCAL_MD5=$(md5sum "$LARGEST" | awk '{print $1}')
    FN=$(basename "$LARGEST")
    REMOTE_MD5=$(sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p "$PORT" "$USER_HOST" "md5sum $TARGET_DIR/$FN" | awk '{print $1}')
    if [[ "$LOCAL_MD5" == "$REMOTE_MD5" ]]; then
        echo "[scp_upload] ✓ md5 match: $LOCAL_MD5"
    else
        echo "[scp_upload] ✗ md5 MISMATCH local=$LOCAL_MD5 remote=$REMOTE_MD5" >&2
        exit 2
    fi
fi
echo "[scp_upload] DONE"
