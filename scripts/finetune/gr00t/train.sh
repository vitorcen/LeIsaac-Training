#!/usr/bin/env bash
# GR00T-N1.6 fine-tune launcher — LeIsaac SO-101 PickOrange.
#
# Native Isaac-GR00T path (NOT lerobot wrapper — wrapper hardcodes N1.5).
# Single 4090, bf16 full fine-tune (projector + DiT + top4 LLM layers ≈ 600M trainable).
#
# Env knobs:
#   GR00T_ROOT        path to Isaac-GR00T repo (default: ~/work/Isaac-GR00T)
#   DATASET_DIR       LeRobot v3.0 dataset root (default: LeIsaac/datasets/raw/leisaac-pick-orange)
#   OUTPUT_DIR        ckpt + logs output dir
#   BASE_MODEL        HF model id (default: nvidia/GR00T-N1.6-3B)
#   MAX_STEPS         total steps (default: 10000)
#   SAVE_STEPS        ckpt cadence (default: 500 — X-VLA sweet spot for auto-eval)
#   GLOBAL_BATCH      effective batch (default: 32)
#   GRAD_ACCUM        grad accumulation (default: 4 → per-step batch = 32/4 = 8)
#   USE_WANDB         enable wandb (default: 0)
#
# Note: launch_finetune.py uses tyro CLI; kebab-case OR snake_case both work.
# We use snake_case to match SO100 finetune_so100.sh example.

set -euo pipefail

LEISAAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "$LEISAAC_ROOT/.." && pwd)"

GR00T_ROOT="${GR00T_ROOT:-$HOME/work/Isaac-GR00T}"
DATASET_DIR="${DATASET_DIR:-$LEISAAC_ROOT/datasets/v2-gr00t/leisaac-pick-orange}"
OUTPUT_DIR="${OUTPUT_DIR:-$LEISAAC_ROOT/outputs/gr00t-n16-leisaac-pick-orange}"
MODALITY_CFG="${MODALITY_CFG:-$LEISAAC_ROOT/scripts/finetune/gr00t/leisaac_config.py}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-25}"
USE_WANDB="${USE_WANDB:-0}"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[gr00t-train] ERROR: dataset not found: $DATASET_DIR" >&2
    exit 1
fi
if [[ ! -f "$DATASET_DIR/meta/modality.json" ]]; then
    echo "[gr00t-train] ERROR: dataset missing meta/modality.json (run scaffold step)" >&2
    exit 1
fi
if [[ ! -d "$GR00T_ROOT" ]]; then
    echo "[gr00t-train] ERROR: Isaac-GR00T repo not found: $GR00T_ROOT" >&2
    exit 1
fi
if [[ ! -f "$MODALITY_CFG" ]]; then
    echo "[gr00t-train] ERROR: modality config not found: $MODALITY_CFG" >&2
    exit 1
fi

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gr00t_train_$(date +%Y%m%d_%H%M%S).log"

WANDB_FLAG=()
if [[ "$USE_WANDB" == "1" ]]; then
    WANDB_FLAG+=(--use_wandb)
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Disable cublasLt fast path: triggered CUBLAS_STATUS_NOT_SUPPORTED on bf16 matmul
# with non-contiguous tensors produced by use_reentrant=False checkpoint recompute
# (torch 2.7.1 + cuBLAS bug on certain bf16 shapes).
export DISABLE_ADDMM_CUDA_LT="${DISABLE_ADDMM_CUDA_LT:-1}"

echo "[gr00t-train] launching:"
echo "  gr00t_root=$GR00T_ROOT  dataset=$DATASET_DIR  output=$OUTPUT_DIR"
echo "  base=$BASE_MODEL  steps=$MAX_STEPS  save_steps=$SAVE_STEPS"
echo "  global_batch=$GLOBAL_BATCH  grad_accum=$GRAD_ACCUM  (per-step ≈ $((GLOBAL_BATCH / GRAD_ACCUM)))"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  log=$LOG_FILE"

cd "$GR00T_ROOT"

WRAPPER="$LEISAAC_ROOT/scripts/finetune/gr00t/launch_finetune_ckpt.py"
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
GR00T_ROOT="$GR00T_ROOT" \
exec uv run --extra=gpu python \
    "$WRAPPER" \
        --base_model_path "$BASE_MODEL" \
        --dataset_path "$DATASET_DIR" \
        --modality_config_path "$MODALITY_CFG" \
        --embodiment_tag NEW_EMBODIMENT \
        --num_gpus 1 \
        --output_dir "$OUTPUT_DIR" \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit "$SAVE_TOTAL_LIMIT" \
        --max_steps "$MAX_STEPS" \
        --warmup_ratio 0.05 \
        --weight_decay 1e-5 \
        --learning_rate 1e-4 \
        --global_batch_size "$GLOBAL_BATCH" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
        --shard_size 1024 \
        --num_shards_per_epoch 100000 \
        --episode_sampling_rate 0.1 \
        "${WANDB_FLAG[@]}" \
        2>&1 | tee "$LOG_FILE"
