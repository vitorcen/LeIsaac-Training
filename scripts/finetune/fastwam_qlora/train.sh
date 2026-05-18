#!/usr/bin/env bash
# Launch FastWAM QLoRA finetune for LeIsaac SO-101 PickOrange.
#
# Steps:
#   1. Pre-compute UMT5 text embedding for the PickOrange prompt (once).
#   2. Run train.py — builds FastWAM, applies NF4 + LoRA, hands off to Wan22Trainer.
#
# Knobs (env vars):
#   CONDA_ENV          fastwam env                       (default: fastwam)
#   FASTWAM_REPO_ROOT  upstream fastwam repo             (default: ~/work/fastwam-repo)
#   SKIP_TEXT_CACHE    set to 1 to skip text precompute  (default: 0)
#
# Usage:
#   bash LeIsaac/scripts/finetune/fastwam/train.sh [extra hydra overrides]

set -euo pipefail

LEISAAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-fastwam}"
FASTWAM_REPO_ROOT="${FASTWAM_REPO_ROOT:-$HOME/work/fastwam-repo}"
SKIP_TEXT_CACHE="${SKIP_TEXT_CACHE:-0}"

DATASET_DIR="$LEISAAC_ROOT/datasets/raw/leisaac-pick-orange_old"
CACHE_DIR="$FASTWAM_REPO_ROOT/data/text_embeds_cache/leisaac_pickorange"
PROMPT="Grab orange and place into plate"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[fastwam-qlora] ERROR: dataset not found: $DATASET_DIR" >&2
    exit 1
fi

# Importable as `fastwam_qlora` once $PARENT_OF_PKG is on PYTHONPATH.
PARENT_OF_PKG="$(dirname "$SCRIPT_DIR")"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PARENT_OF_PKG:$FASTWAM_REPO_ROOT/src:${PYTHONPATH:-}"

cd "$FASTWAM_REPO_ROOT"

# --- Step 1: precompute text embedding cache for the one PickOrange prompt ---
# UMT5-xxl 5.5B model.  Default init is fp32 → ~22GB transient on CUDA,
# which OOMs alongside the eventual DiT load.  Force CPU encoding (~30s)
# so the GPU stays free for training.
if [[ "$SKIP_TEXT_CACHE" != "1" ]]; then
    echo "[fastwam-qlora] Pre-caching UMT5 embedding (CPU) for prompt: \"$PROMPT\""
    mkdir -p "$CACHE_DIR"
    CUDA_VISIBLE_DEVICES="" conda run -n "$CONDA_ENV" --no-capture-output \
        python "$FASTWAM_REPO_ROOT/scripts/precompute_text_embeds.py" \
        --config-path "$SCRIPT_DIR/configs" \
        --config-name train \
        +override_instruction="$PROMPT" \
        +overwrite=false
fi

# --- Step 2: train ---
echo "[fastwam-qlora] Launching training..."
exec conda run -n "$CONDA_ENV" --no-capture-output \
    python -m fastwam_qlora.train \
    --config-path "$SCRIPT_DIR/configs" \
    --config-name train \
    "$@"
