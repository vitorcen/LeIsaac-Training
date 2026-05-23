#!/usr/bin/env bash
# Download a public HF dataset via hf-mirror.com (fastest path, ~12 MB/s sustained).
# Default = LeIsaac PickOrange dataset.
#
# Usage:
#   bash download_dataset.sh [HF_DATASET_ID]
#
# Examples:
#   bash download_dataset.sh                                    # → LightwheelAI/leisaac-pick-orange
#   bash download_dataset.sh LightwheelAI/leisaac-cleanup-table

set -euo pipefail

DATASET_ID="${1:-LightwheelAI/leisaac-pick-orange}"
export PATH=/root/.local/bin:/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
unset http_proxy https_proxy   # public mirror, no proxy needed

echo "[download_dataset] downloading $DATASET_ID via $HF_ENDPOINT (8 workers)"
time hf download "$DATASET_ID" --repo-type dataset --max-workers 8
echo "[download_dataset] DONE"
du -sh "$HF_HOME/hub/datasets--${DATASET_ID//\//--}" 2>/dev/null
