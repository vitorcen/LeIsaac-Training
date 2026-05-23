#!/usr/bin/env bash
# Download a gated HF model via curl single-stream (the only path that survives
# AutoDL proxy disconnects). 多 worker / aria2 / hf python client 全部被代理掐。
#
# Usage:
#   HF_TOKEN=hf_xxxx bash download_gated_model.sh nvidia/Cosmos-Reason2-2B [output_dir]
#
# Output:
#   /root/autodl-tmp/<modelname>_raw/  ←  all files flat
#
# WARNING: this can take 2-3 hours for 4.6 GB at ~600 KB/s.
# If you have the model in local HF cache on your dev machine, prefer scp_upload_model.sh
# (runs on local dev box, 15 min @ 5 MB/s).

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[download_gated] ERROR: HF_TOKEN env var required" >&2
    exit 1
fi

REPO_ID="${1:?usage: HF_TOKEN=xxx bash $0 <repo_id> [output_dir]}"
REPO_NAME="${REPO_ID##*/}"
OUTPUT_DIR="${2:-/root/autodl-tmp/${REPO_NAME,,}_raw}"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"
source /etc/network_turbo >/dev/null 2>&1

# 1. fetch file list from HF API (canonical endpoint, with auth)
echo "[download_gated] fetching file list for $REPO_ID ..."
FILES=$(curl -s -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/models/$REPO_ID" | \
    grep -oE '"rfilename":"[^"]*"' | sed 's/"rfilename":"//;s/"//g' | \
    grep -v '^\.gitattributes$' | grep -v '^[A-Za-z0-9]*$' )    # skip top-level non-files
echo "$FILES" | head

# 2. download each file with curl single-stream + auto-retry on disconnect
for f in $FILES; do
    dir=$(dirname "$f")
    [[ "$dir" != "." ]] && mkdir -p "$dir"
    # already complete?
    EXPECTED_SIZE=$(curl -sI -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/$REPO_ID/resolve/main/$f" 2>/dev/null | \
        grep -iE '^(x-linked-size|content-length):' | head -1 | awk '{print $2}' | tr -d '\r')
    if [[ -f "$f" ]]; then
        ACTUAL=$(stat -c %s "$f" 2>/dev/null || echo 0)
        if [[ "$ACTUAL" == "$EXPECTED_SIZE" && -n "$EXPECTED_SIZE" ]]; then
            echo "[$(date +%H:%M:%S)] $f already complete ($ACTUAL bytes), skipping"
            continue
        fi
        echo "[$(date +%H:%M:%S)] $f partial ($ACTUAL/$EXPECTED_SIZE), DELETING and restarting (sparse-hole risk)"
        rm -f "$f"
    fi
    echo "[$(date +%H:%M:%S)] downloading $f (expected $EXPECTED_SIZE bytes)"
    # critical flags:
    #   --speed-time 60 --speed-limit 10240  → abort + retry if speed < 10 KB/s for 60s
    #   --retry 50 --retry-delay 15          → up to 50 retries with 15s gap
    #   NO -C - (resume)                     → cross-session resume creates sparse holes
    curl -L --retry 50 --retry-delay 15 --retry-max-time 14400 \
         --max-time 14400 \
         --speed-time 60 --speed-limit 10240 \
         -o "$f" \
         -H "Authorization: Bearer $HF_TOKEN" \
         "https://huggingface.co/$REPO_ID/resolve/main/$f"
    # verify
    ACTUAL=$(stat -c %s "$f" 2>/dev/null || echo 0)
    BLOCKS=$(stat -c %b "$f" 2>/dev/null || echo 0)
    DENSE_BYTES=$((BLOCKS * 512))
    echo "[$(date +%H:%M:%S)] done $f: size=$ACTUAL dense=$DENSE_BYTES"
    if [[ -n "$EXPECTED_SIZE" && "$ACTUAL" != "$EXPECTED_SIZE" ]]; then
        echo "[$(date +%H:%M:%S)] ⚠️ SIZE MISMATCH for $f" >&2
    fi
done

echo "[download_gated] DONE — files at $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
