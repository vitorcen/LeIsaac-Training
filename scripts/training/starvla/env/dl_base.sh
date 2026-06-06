#!/bin/bash
# Download a VLM backbone for StarVLA (runs on the AutoDL cloud box, via proxy).
# Parameterized by REPO id so every variant reuses it:
#   REPO=Qwen/Qwen3-VL-4B-Instruct         bash dl_base.sh   # default (QwenGR00T)
#   REPO=google/gemma-3-4b-it              bash dl_base.sh   # Gemma4 variant
#   REPO=nvidia/Cosmos-Reason1-7B          bash dl_base.sh   # Cosmos variant
REPO=${REPO:-Qwen/Qwen3-VL-4B-Instruct}
DEST=${DEST:-/root/autodl-tmp/models/$(basename "$REPO")}
exec > "/root/dl_$(basename "$REPO").log" 2>&1
set -o pipefail
export https_proxy=${https_proxy:-http://127.0.0.1:7890} http_proxy=${http_proxy:-http://127.0.0.1:7890}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
HF=${HF:-/root/miniconda3/envs/wallx/bin/hf}
mkdir -p "$DEST"
echo "=== download $REPO -> $DEST ==="; date
$HF download "$REPO" --local-dir "$DEST" 2>&1 | tail -30
echo "=== done ==="; date
du -sh "$DEST"
ls "$DEST"
