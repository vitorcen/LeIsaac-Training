#!/usr/bin/env bash
# scp a pre-built uv cache bundle (from prefetch_uv_cache.sh) to AutoDL.
# Runs on LOCAL DEV BOX.
#
# Usage:
#   SSHPASS=xxxxx bash scp_upload_bundle.sh /tmp/n17_uv_cache_bundle_*.tar.gz \
#       root@connect.westd.seetacloud.com 13330

set -euo pipefail

BUNDLE="${1:?usage: SSHPASS=xxx bash $0 <bundle.tar.gz> [user@host] [port]}"
USER_HOST="${2:-root@connect.westd.seetacloud.com}"
PORT="${3:-22}"

if [[ ! -f "$BUNDLE" ]]; then
    echo "ERROR: $BUNDLE not found" >&2; exit 1
fi
if ! command -v sshpass >/dev/null 2>&1; then
    echo "ERROR: sshpass not installed (apt install sshpass)" >&2; exit 1
fi

BUNDLE_SIZE=$(du -h "$BUNDLE" | cut -f1)
echo "[scp_upload_bundle] uploading $BUNDLE ($BUNDLE_SIZE) to $USER_HOST:/root/autodl-tmp/ (port $PORT)"
echo "  expected: ~$(du -B 1M "$BUNDLE" | awk '{print int($1/5)}') sec @ 5 MB/s upload bandwidth"

time sshpass -e scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -P "$PORT" "$BUNDLE" "$USER_HOST:/root/autodl-tmp/"

# verify
REMOTE_NAME=$(basename "$BUNDLE")
LOCAL_MD5=$(md5sum "$BUNDLE" | awk '{print $1}')
REMOTE_MD5=$(sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p "$PORT" "$USER_HOST" "md5sum /root/autodl-tmp/$REMOTE_NAME" | awk '{print $1}')

if [[ "$LOCAL_MD5" == "$REMOTE_MD5" ]]; then
    echo "[scp_upload_bundle] ✓ md5 match: $LOCAL_MD5"
else
    echo "[scp_upload_bundle] ✗ md5 MISMATCH local=$LOCAL_MD5 remote=$REMOTE_MD5" >&2
    exit 2
fi

cat <<EOF
[scp_upload_bundle] DONE
=== next step on AutoDL (no-card mode is fine) ===
  ssh -p $PORT $USER_HOST
  cd /root/autodl-tmp
  tar xzf $REMOTE_NAME
  bash isaaclab-experience/LeIsaac/scripts/cloud/autodl/uv_sync_offline.sh /root/autodl-tmp
EOF
