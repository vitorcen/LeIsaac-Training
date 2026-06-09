#!/usr/bin/env bash
# π0.5 PyTorch LoRA fine-tune on a LeRobot v3.0 dataset.
#
# Thin wrapper around the pi05_leisaac.train entry point installed by
# pip install -e ~/work/isaaclab-experience/LeIsaac/server.
#
# Usage:
#   bash scripts/training/openpi/pytorch/train.sh                 # 3000-step smoke
#   STEPS=10000 bash scripts/training/openpi/pytorch/train.sh     # full 10k
#
# Knobs (env vars):
#   CONDA_ENV        lerobot env                 (default: lerobot)
#   DATASET_REPO_ID  HF id                       (default: LightwheelAI/leisaac-pick-orange)
#   DATASET_ROOT     local v3.0 path             (default: ~/work/LeIsaac/datasets/raw/leisaac-pick-orange)
#   OUTPUT_DIR       LoRA output dir             (default: ~/work/LeIsaac/outputs/pi05-leisaac-pt)
#   STEPS            total training steps        (default: 3000)
#   BATCH_SIZE       per-step batch              (default: 16)
#   LR / WARMUP      AdamW lr + warmup           (defaults: 5e-5 / 500)
#   LORA_R/ALPHA     LoRA rank/alpha             (defaults: 16/16)
#   SAVE_FREQ        ckpt save interval          (default: 1000)
#   GRADIENT_CKPT    fit batch=16 in 24GB        (default: 1)
#   LEROBOT_SRC      editable lerobot src dir    (default: ~/work/lerobot-experience/lerobot/src)

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-LightwheelAI/leisaac-pick-orange}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/work/LeIsaac/datasets/raw/leisaac-pick-orange}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/work/LeIsaac/outputs/pi05-leisaac-pt}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-500}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
GRADIENT_CKPT="${GRADIENT_CKPT:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEROBOT_SRC="${LEROBOT_SRC:-${HOME}/work/lerobot-experience/lerobot/src}"

GC_FLAG=""
[[ "${GRADIENT_CKPT}" == "1" ]] && GC_FLAG="--gradient-checkpointing"

INIT_FLAG=""
if [[ -n "${INIT_LORA:-}" ]]; then
    if [[ ! -f "${INIT_LORA}" ]]; then
        echo "[train] ERROR: INIT_LORA not found: ${INIT_LORA}" >&2; exit 1
    fi
    INIT_FLAG="--init-lora-npz ${INIT_LORA}"
fi

PHASED_FLAG=""
if [[ "${PHASED_SAMPLER:-0}" == "1" ]]; then
    PHASED_FLAG="--phased-sampler"
    PHASED_HEAD_FRAMES="${PHASED_HEAD_FRAMES:-100}"
    PHASED_HEAD_WEIGHT="${PHASED_HEAD_WEIGHT:-8.0}"
    PHASED_MID_FRAMES="${PHASED_MID_FRAMES:-100}"
    PHASED_MID_WEIGHT="${PHASED_MID_WEIGHT:-4.0}"
    PHASED_FLAG="${PHASED_FLAG} --phased-head-frames ${PHASED_HEAD_FRAMES} \
        --phased-head-weight ${PHASED_HEAD_WEIGHT} \
        --phased-mid-frames ${PHASED_MID_FRAMES} \
        --phased-mid-weight ${PHASED_MID_WEIGHT}"
fi

export LEROBOT_SRC
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="$(dirname "${OUTPUT_DIR}")/.logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(basename "${OUTPUT_DIR}")-train.log"

echo "[train] dataset:    ${DATASET_REPO_ID}  (${DATASET_ROOT})"
echo "[train] output:     ${OUTPUT_DIR}"
echo "[train] steps:      ${STEPS}  batch: ${BATCH_SIZE}  lr: ${LR}  warmup: ${WARMUP}"
echo "[train] LoRA:       r=${LORA_R} α=${LORA_ALPHA}  save_freq: ${SAVE_FREQ}"
echo "[train] gradient checkpointing: ${GRADIENT_CKPT}"
echo "[train] log:        ${LOG_FILE}"

exec conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u -m pi05_leisaac.train \
        --dataset-repo-id "${DATASET_REPO_ID}" \
        --dataset-root "${DATASET_ROOT}" \
        --output-dir "${OUTPUT_DIR}" \
        --steps "${STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" --warmup-steps "${WARMUP}" \
        --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" \
        --save-freq "${SAVE_FREQ}" \
        --num-workers "${NUM_WORKERS}" \
        --dtype bfloat16 \
        ${GC_FLAG} ${INIT_FLAG} ${PHASED_FLAG} 2>&1 | tee "${LOG_FILE}"
