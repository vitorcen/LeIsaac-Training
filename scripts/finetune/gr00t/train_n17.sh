#!/usr/bin/env bash
# GR00T-N1.7 fine-tune launcher — LeIsaac SO-101 PickOrange.
#
# Replicate hi-space/GR00T-N1.7-3B-Pick-Orange recipe (current 14/15 SOTA).
#  - VLM backbone : nvidia/Cosmos-Reason2-2B (4.6 GB, already cached)
#  - Action head  : Gr00tN1d7, action_horizon=40, 4 diffusion steps
#  - Trainable    : projector + DiT + linear heads + VL-LN  (~600M)
#  - Frozen       : Cosmos vision encoder + LLM backbone
#
# Single 4090 24GB, bf16; adafactor + grad-ckpt squeeze. If OOM, flip
# `backbone_trainable_params_fp32 = False` inside launch_finetune_ckpt_n17.py.
#
# Env knobs:
#   GR00T_ROOT          Isaac-GR00T repo (default: REPO_ROOT/dependencies/Isaac-GR00T)
#   DATASET_DIR         LeRobot v3.0 dataset (default: LeIsaac v2-gr00t leisaac-pick-orange)
#   OUTPUT_DIR          ckpt + logs (default: LeIsaac/outputs/gr00t-n17-leisaac-pick-orange)
#   BASE_MODEL          Path 1 (cold, default on AutoDL) = /root/autodl-tmp/cosmos_raw
#                       Path 1 (HF download)             = nvidia/Cosmos-Reason2-2B
#                       Path 2 (warm)                    = hi-space/GR00T-N1.7-3B-Pick-Orange
#   MAX_STEPS           default 10000 (hi-space converged at 6000; we go longer + auto-eval-on-save)
#   SAVE_STEPS          default 1000   (10k step / 10 ckpts; fits 140GB autodl-tmp)
#   SAVE_ONLY_MODEL     default 1      (skip optimizer state, single ckpt 25→12 GB)
#   LOSS_PRUNE_TOP_K    default 5      (keep best-5 ckpts by train_loss + last 1 = 6 max)
#   GPU_PROFILE         "auto" (default) | "small24" | "big48" | "big96"
#                       auto detect via nvidia-smi:
#                         <30 GB  → small24  (4090 24GB squeeze: grad-ckpt + adafactor + per-step=2)
#                         30-60GB → big48    (no grad-ckpt + adamw + per-step=4)
#                         >60 GB  → big96    (no grad-ckpt + adamw + per-step=8; e.g. RTX PRO 6000 96GB)
#   GLOBAL_BATCH        auto from GPU_PROFILE (override to force)
#   GRAD_ACCUM          auto from GPU_PROFILE
#   OPTIM               auto from GPU_PROFILE: adafactor (small24) vs adamw_torch (big48/big96)
#   GRADIENT_CKPT       auto from GPU_PROFILE: 1 (small24) vs 0 (big48/big96)
#   USE_WANDB           default 0
#
# Don't `set -x` here — Adafactor prints fewer things than Adam, log stays readable.

set -euo pipefail

LEISAAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "$LEISAAC_ROOT/.." && pwd)"

GR00T_ROOT="${GR00T_ROOT:-$REPO_ROOT/dependencies/Isaac-GR00T}"
DATASET_DIR="${DATASET_DIR:-$LEISAAC_ROOT/datasets/v2-gr00t/leisaac-pick-orange}"
OUTPUT_DIR="${OUTPUT_DIR:-$LEISAAC_ROOT/outputs/gr00t-n17-leisaac-pick-orange}"
MODALITY_CFG="${MODALITY_CFG:-$LEISAAC_ROOT/scripts/finetune/gr00t/leisaac_config_n17.py}"
# BASE_MODEL: AutoDL workflow pre-stages Cosmos at /root/autodl-tmp/cosmos_raw (scp from local)
# to avoid the unstable AutoDL proxy. If that dir exists we use it; otherwise fall back to HF id.
_DEFAULT_BASE_MODEL="nvidia/Cosmos-Reason2-2B"
if [[ -d "/root/autodl-tmp/cosmos_raw" && -f "/root/autodl-tmp/cosmos_raw/config.json" ]]; then
    _DEFAULT_BASE_MODEL="/root/autodl-tmp/cosmos_raw"
fi
BASE_MODEL="${BASE_MODEL:-$_DEFAULT_BASE_MODEL}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1200}"
# save_only_model=True dropped 332/1030 weight keys interacting with tune_top_llm_layers — keep full ckpt.
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-0}"
LOSS_PRUNE_TOP_K="${LOSS_PRUNE_TOP_K:-2}"

# Auto-detect GPU profile from VRAM (resolves GPU_PROFILE=auto)
GPU_PROFILE="${GPU_PROFILE:-auto}"
if [[ "$GPU_PROFILE" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        _VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
        if   [[ $_VRAM_GB -gt 60 ]]; then GPU_PROFILE="big96"
        elif [[ $_VRAM_GB -gt 30 ]]; then GPU_PROFILE="big48"
        else                              GPU_PROFILE="small24"; fi
        echo "[gr00t-n17-train] auto-detected GPU_PROFILE=$GPU_PROFILE (VRAM ${_VRAM_GB} GB)"
    else
        GPU_PROFILE="small24"
        echo "[gr00t-n17-train] nvidia-smi unavailable, defaulting to GPU_PROFILE=small24" >&2
    fi
fi

case "$GPU_PROFILE" in
    small24)  # RTX 4090 24 GB: every memory squeeze trick
        _G=${GLOBAL_BATCH:-8};  _A=${GRAD_ACCUM:-4};  _O=${OPTIM:-adafactor};      _C=${GRADIENT_CKPT:-1} ;;
    big48)    # A100 40GB / L40 48GB: moderate
        _G=${GLOBAL_BATCH:-16}; _A=${GRAD_ACCUM:-4};  _O=${OPTIM:-adamw_torch};   _C=${GRADIENT_CKPT:-0} ;;
    big96)    # RTX PRO 6000 96 GB / H100 80GB: efficient
        _G=${GLOBAL_BATCH:-32}; _A=${GRAD_ACCUM:-4};  _O=${OPTIM:-adamw_torch};   _C=${GRADIENT_CKPT:-0} ;;
    *)
        echo "[gr00t-n17-train] ERROR: unknown GPU_PROFILE=$GPU_PROFILE (small24|big48|big96|auto)" >&2
        exit 1 ;;
esac
GLOBAL_BATCH=$_G; GRAD_ACCUM=$_A; OPTIM=$_O; GRADIENT_CKPT=$_C
export OPTIM GRADIENT_CKPT

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
USE_WANDB="${USE_WANDB:-0}"
export LOSS_PRUNE_TOP_K

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[gr00t-n17-train] ERROR: dataset not found: $DATASET_DIR" >&2
    exit 1
fi
if [[ ! -f "$DATASET_DIR/meta/modality.json" ]]; then
    echo "[gr00t-n17-train] ERROR: dataset missing meta/modality.json" >&2
    exit 1
fi
if [[ ! -d "$GR00T_ROOT" ]]; then
    echo "[gr00t-n17-train] ERROR: Isaac-GR00T repo not found: $GR00T_ROOT" >&2
    exit 1
fi
if [[ ! -f "$MODALITY_CFG" ]]; then
    echo "[gr00t-n17-train] ERROR: modality config not found: $MODALITY_CFG" >&2
    exit 1
fi

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gr00t_n17_train_$(date +%Y%m%d_%H%M%S).log"

WANDB_FLAG=()
if [[ "$USE_WANDB" == "1" ]]; then
    WANDB_FLAG+=(--use_wandb)
fi
SAVE_ONLY_MODEL_FLAG=()
if [[ "$SAVE_ONLY_MODEL" == "1" ]]; then
    SAVE_ONLY_MODEL_FLAG+=(--save_only_model)
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Same cuBLAS workaround as N1.6 wrapper (torch 2.7.1 bf16 non-contig matmul bug).
export DISABLE_ADDMM_CUDA_LT="${DISABLE_ADDMM_CUDA_LT:-1}"

echo "[gr00t-n17-train] launching:"
echo "  gr00t_root=$GR00T_ROOT"
echo "  dataset=$DATASET_DIR"
echo "  output=$OUTPUT_DIR"
echo "  base=$BASE_MODEL  steps=$MAX_STEPS  save_steps=$SAVE_STEPS"
echo "  save_only_model=$SAVE_ONLY_MODEL  loss_prune_top_k=$LOSS_PRUNE_TOP_K  save_total_limit=$SAVE_TOTAL_LIMIT"
echo "  gpu_profile=$GPU_PROFILE  optim=$OPTIM  grad_ckpt=$GRADIENT_CKPT"
echo "  global_batch=$GLOBAL_BATCH  grad_accum=$GRAD_ACCUM  (per-step ≈ $((GLOBAL_BATCH / GRAD_ACCUM)))"
echo "  modality_cfg=$MODALITY_CFG"
echo "  log=$LOG_FILE"

cd "$GR00T_ROOT"

WRAPPER="$LEISAAC_ROOT/scripts/finetune/gr00t/launch_finetune_ckpt_n17.py"
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
GR00T_ROOT="$GR00T_ROOT" \
exec uv run --no-sync python \
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
        "${SAVE_ONLY_MODEL_FLAG[@]}" \
        "${WANDB_FLAG[@]}" \
        2>&1 | tee "$LOG_FILE"
