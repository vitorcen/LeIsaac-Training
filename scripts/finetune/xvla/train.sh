#!/usr/bin/env bash
# X-VLA fine-tune launcher — LeIsaac SO-101 PickOrange.
#
# Two modes (selected by RESUME env var):
#   RESUME=false → fresh start from lerobot/xvla-base via --policy.path
#                  All policy CLI overrides apply.
#   RESUME=true  → continue from $OUTPUT_DIR/checkpoints/last/pretrained_model
#                  Passes --config_path so lerobot loads full TrainPipelineConfig
#                  (incl. optimizer/scheduler) from the saved train_config.json.
#                  --policy.path MUST NOT be passed (mutually exclusive in lerobot).
#                  Override knobs come via --steps / --save_freq (CLI overrides
#                  on top of the loaded config).
#
# Why the split: lerobot's `validate()` takes the `--policy.path` branch if
# present, skipping the resume branch that loads optimizer/scheduler from disk;
# the result is `cfg.optimizer is None` → ValueError at make_optimizer_and_scheduler.
#
# Env knobs:
#   CONDA_ENV    conda env w/ lerobot xvla (default: lerobot)
#   DATASET_DIR  LeRobot v3.0 dataset root
#   OUTPUT_DIR   ckpt + logs output dir
#   MAX_STEPS    training steps (default 10000)
#   BATCH_SIZE   per-step batch (default 4 for 24GB)
#   SAVE_FREQ    ckpt save cadence (default 500)
#   DOMAIN_ID    soft_prompt slot 0..29 (default 0)
#   RESUME       false (fresh) | true (continue from last ckpt)

set -euo pipefail

LEISAAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "$LEISAAC_ROOT/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot}"
DATASET_DIR="${DATASET_DIR:-$LEISAAC_ROOT/datasets/raw/leisaac-pick-orange}"
OUTPUT_DIR="${OUTPUT_DIR:-$LEISAAC_ROOT/outputs/xvla-leisaac-pick-orange}"
BASE_CKPT="${BASE_CKPT:-lerobot/xvla-base}"
MAX_STEPS="${MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-500}"
DOMAIN_ID="${DOMAIN_ID:-0}"
RESUME="${RESUME:-false}"
# Episode subset for training.  Default: '[0..49]' so 50-59 stay as a true held-out
# val set for offline_action_mse.  Pass empty string to use all 60 episodes (old behavior).
EPISODES_TRAIN="${EPISODES_TRAIN:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]}"
# Toggle image augmentation (lerobot built-in: ColorJitter / SharpnessJitter / RandomAffine).
# Set IMAGE_AUG=1 to enable; default off.  Visuomotor-aug recipe (ALOHA, DP).
IMAGE_AUG="${IMAGE_AUG:-0}"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[xvla-train] ERROR: dataset not found: $DATASET_DIR" >&2
    exit 1
fi
# Do NOT mkdir OUTPUT_DIR: lerobot rejects pre-existing dir unless --resume is set.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/xvla_train_$(date +%Y%m%d_%H%M%S).log"

EXTRA_ARGS=()
if [[ "$IMAGE_AUG" == "1" ]]; then
    EXTRA_ARGS+=(--dataset.image_transforms.enable=true)
fi
# Weak aug: only brightness + contrast (no saturation/hue/sharpness/affine).
# Use ONLY brightness with mild range; turn off all others by weight=0.
# Tested because default aug was harmful (4/18=22% vs baseline ~33%) on
# 50-demo dataset — over-regularizes.
if [[ "${WEAK_IMAGE_AUG:-0}" == "1" ]]; then
    EXTRA_ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.max_num_transforms=1
        '--dataset.image_transforms.tfs={"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.95,1.05]}}}'
    )
fi

echo "[xvla-train] launching:"
echo "  env=$CONDA_ENV  dataset=$DATASET_DIR  output=$OUTPUT_DIR"
echo "  base=$BASE_CKPT  steps=$MAX_STEPS  batch=$BATCH_SIZE  save_freq=$SAVE_FREQ"
echo "  resume=$RESUME  image_aug=$IMAGE_AUG  log=$LOG_FILE"

if [[ "$RESUME" == "true" ]]; then
    CFG_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
    if [[ ! -f "$CFG_PATH" ]]; then
        echo "[xvla-train] ERROR: --resume=true but $CFG_PATH not found" >&2
        exit 1
    fi
    # On resume, lerobot loads the FULL TrainPipelineConfig from disk
    # (incl. optimizer, scheduler, policy, dataset).  We only override
    # the few knobs we want to change on the next segment.
    # NOTE: train_entry.py is our thin wrapper that registers SingleArmSO101
    # action_space into lerobot before invoking lerobot.scripts.lerobot_train,
    # so the lerobot submodule stays patch-free.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
    exec conda run -n "$CONDA_ENV" --no-capture-output \
        python -u -m train_entry \
            --config_path="$CFG_PATH" \
            --resume=true \
            --steps="$MAX_STEPS" \
            --save_freq="$SAVE_FREQ" \
            --batch_size="$BATCH_SIZE" \
            --num_workers="$NUM_WORKERS" \
            2>&1 | tee "$LOG_FILE"
else
    # Fresh start.  Note: --policy.type is auto-inferred from the pretrained
    # config when --policy.path is given; passing both raises an error.
    # train_entry wraps lerobot_train and registers SingleArmSO101 first.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
    exec conda run -n "$CONDA_ENV" --no-capture-output \
        python -u -m train_entry \
            --policy.path="$BASE_CKPT" \
            --policy.action_mode=so101_single \
            --policy.max_state_dim=20 \
            --policy.max_action_dim=20 \
            --policy.empty_cameras=1 \
            --policy.chunk_size=32 \
            --policy.n_action_steps=8 \
            --policy.n_obs_steps=2 \
            --policy.use_proprio=true \
            --policy.freeze_vision_encoder=true \
            --policy.freeze_language_encoder=true \
            --policy.train_policy_transformer=true \
            --policy.train_soft_prompts=true \
            --policy.num_denoising_steps=10 \
            --policy.resize_imgs_with_padding="[224,224]" \
            '--policy.normalization_mapping={"STATE":"MEAN_STD","ACTION":"MIN_MAX"}' \
            --policy.optimizer_lr=1e-4 \
            --policy.scheduler_warmup_steps=200 \
            --policy.optimizer_grad_clip_norm=10.0 \
            --policy.device=cuda \
            --policy.dtype=bfloat16 \
            --policy.push_to_hub=false \
            --dataset.repo_id=leisaac/pick-orange \
            --dataset.root="$DATASET_DIR" \
            --dataset.episodes="$EPISODES_TRAIN" \
            '--rename_map={"observation.images.front":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
            --output_dir="$OUTPUT_DIR" \
            --resume=false \
            --batch_size="$BATCH_SIZE" \
            --num_workers="$NUM_WORKERS" \
            --steps="$MAX_STEPS" \
            --save_freq="$SAVE_FREQ" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "$LOG_FILE"
fi
