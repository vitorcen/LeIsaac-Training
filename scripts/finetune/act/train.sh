#!/usr/bin/env bash
# ACT fine-tune (from scratch) on LeIsaac PickOrange v3.0 dataset.
#
# Recipe ported verbatim from shadowHokage/act_policy (the only public 1/1
# success on this dataset, 36k frames / 60 episodes). Key deltas vs the
# lerobot ACT default:
#   - chunk_size 50 → 100 (2x larger action chunk)
#   - batch_size 64 → 8 (small batches, more gradient steps per epoch)
#   - lr 1e-4 → 1e-5 (10x smaller, both head and backbone)
#   - steps 100000 → 10000 (short training, ~2.2 epochs over 36k frames)
#   - image_transforms disabled (no ColorJitter / RandomAffine)
#   - normalization MEAN_STD (not QUANTILES)
#   - use_imagenet_stats=true (vision_backbone ResNet18 with ImageNet pretrain)
#
# Usage:
#   bash scripts/finetune/act/train.sh                     # 10k step shadowHokage parity
#   STEPS=20000 BATCH_SIZE=16 \
#     bash scripts/finetune/act/train.sh                   # extended schedule
#
# Knobs (env vars):
#   OUTPUT_NAME      output dir name              (default act-leisaac-pick-orange)
#   STEPS            total training steps         (default 10000)
#   BATCH_SIZE       per-step batch               (default 8)
#   SAVE_FREQ        ckpt save interval           (default 2000 — 5 ckpts over 10k)
#   LR               adamw lr (head + backbone)   (default 1e-5)
#   CHUNK_SIZE       action chunk length          (default 100)
#   NUM_WORKERS      dataloader workers           (default 4)
#   VIDEO_BACKEND    pyav | torchcodec            (default pyav — torchcodec
#                                                  segfaulted on smolvla v1)
#   CONDA_ENV        conda env name               (default lerobot)
#   DATASET_REPO_ID  HF repo id                   (default LightwheelAI/leisaac-pick-orange)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

OUTPUT_NAME="${OUTPUT_NAME:-act-leisaac-pick-orange}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LR="${LR:-1e-5}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-LightwheelAI/leisaac-pick-orange}"
DATASET_BASENAME="$(basename "${DATASET_REPO_ID}")"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/raw/${DATASET_BASENAME}}"
OUTPUT_DIR="${REPO_ROOT}/outputs/${OUTPUT_NAME}"

# -------- preflight --------
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "[act] dataset not found at ${DATASET_ROOT}" >&2
    echo "[act] hint: bash datasets/download.sh ${DATASET_REPO_ID}" >&2
    exit 1
fi
CV="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version',''))" "${DATASET_ROOT}/meta/info.json")"
if [[ "${CV}" != "v3.0" ]]; then
    echo "[act] dataset is ${CV}, lerobot ≥0.5 requires v3.0" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "[act] ERROR: ${OUTPUT_DIR} exists; lerobot-train refuses to overwrite." >&2
    echo "[act] either delete it, or rerun with OUTPUT_NAME=<new>" >&2
    exit 1
fi

LOG_DIR="${REPO_ROOT}/outputs/.logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${OUTPUT_NAME}.log"

cat <<EOF | tee "${LOG_FILE}"
[act] dataset:        ${DATASET_REPO_ID}  (${DATASET_ROOT})
[act] output:         ${OUTPUT_DIR}
[act] steps:          ${STEPS}
[act] batch:          ${BATCH_SIZE}
[act] lr:             ${LR}  (head + backbone same)
[act] chunk_size:     ${CHUNK_SIZE}  (n_action_steps = chunk_size)
[act] save_freq:      ${SAVE_FREQ}
[act] num_workers:    ${NUM_WORKERS}
[act] video_backend:  ${VIDEO_BACKEND}
[act] conda env:      ${CONDA_ENV}
EOF

# shadowHokage parity: act + chunk=100, ResNet18+ImageNet, no aug, MEAN_STD norm
ARGS=(
    --policy.type=act
    --policy.push_to_hub=false
    --policy.device=cuda
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${CHUNK_SIZE}"
    --policy.vision_backbone=resnet18
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
    --policy.dim_model=512
    --policy.n_heads=8
    --policy.dim_feedforward=3200
    --policy.n_encoder_layers=4
    --policy.n_decoder_layers=1
    --policy.use_vae=true
    --policy.latent_dim=32
    --policy.n_vae_encoder_layers=4
    --policy.dropout=0.1
    --policy.kl_weight=10.0
    --policy.optimizer_lr="${LR}"
    --policy.optimizer_lr_backbone="${LR}"
    --policy.optimizer_weight_decay=1e-4
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend="${VIDEO_BACKEND}"
    --dataset.use_imagenet_stats=true
    --dataset.image_transforms.enable=false
    --output_dir="${OUTPUT_DIR}"
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --num_workers="${NUM_WORKERS}"
    --wandb.enable=false
    --job_name="${OUTPUT_NAME}"
)

echo "[act] launching lerobot-train; full log: ${LOG_FILE}"
set -o pipefail
conda run -n "${CONDA_ENV}" --no-capture-output lerobot-train "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
TRAIN_RC=${PIPESTATUS[0]}
if [[ ${TRAIN_RC} -ne 0 ]]; then
    echo "[act] FAILED with rc=${TRAIN_RC}; see ${LOG_FILE}" >&2
    exit "${TRAIN_RC}"
fi

echo "[act] done; final ckpt at ${OUTPUT_DIR}/checkpoints/last/pretrained_model"
