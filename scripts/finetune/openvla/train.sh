#!/usr/bin/env bash
# OpenVLA-7B QLoRA finetune launcher — LeIsaac SO-101 PickOrange.
#
# Env knobs:
#   CONDA_ENV       conda env (default: openvla)
#   DATASET_DIR     LeRobot v3.0 dataset root
#   OUTPUT_DIR      ckpt + logs output dir
#   MAX_STEPS       training steps (default 10000)
#   LORA_TARGETS    LoRA target modules (default q_proj,v_proj)
#   RESUME          checkpoint dir to resume from (optional)
#
# Usage:
#   bash LeIsaac/scripts/finetune/openvla/train.sh

set -euo pipefail

LEISAAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "$LEISAAC_ROOT/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-openvla}"
DATASET_DIR="${DATASET_DIR:-$LEISAAC_ROOT/datasets/raw/leisaac-pick-orange}"
OUTPUT_DIR="${OUTPUT_DIR:-$LEISAAC_ROOT/outputs/openvla-leisaac-pick-orange}"
MAX_STEPS="${MAX_STEPS:-10000}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"
RESUME="${RESUME:-}"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[openvla-train] ERROR: dataset not found: $DATASET_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Lean allocator — Isaac Sim may co-tenant the GPU later.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# `openvla.dataset` / `openvla.train` are importable once parent is on PYTHONPATH.
export PYTHONPATH="$LEISAAC_ROOT/scripts/finetune:${PYTHONPATH:-}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/openvla_train_$(date +%Y%m%d_%H%M%S).log"

EXTRA=()
if [[ -n "$RESUME" ]]; then
    EXTRA+=(--resume "$RESUME")
fi

echo "[openvla-train] launching:"
echo "  env=$CONDA_ENV  dataset=$DATASET_DIR  output=$OUTPUT_DIR"
echo "  max_steps=$MAX_STEPS  lora_targets=$LORA_TARGETS  log=$LOG_FILE"

exec conda run -n "$CONDA_ENV" --no-capture-output \
    python -u -m openvla.train \
        --dataset "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --max_steps "$MAX_STEPS" \
        --lora_targets "$LORA_TARGETS" \
        "${EXTRA[@]}" "$@" \
        2>&1 | tee "$LOG_FILE"
