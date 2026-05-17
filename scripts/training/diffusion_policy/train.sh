#!/usr/bin/env bash
# Diffusion Policy from-scratch training on LeRobot v3.0 datasets.
#
# Thin wrapper around scripts/training/lerobot_finetune.sh (the generic
# LeRobot launcher) that pins Diffusion-Policy-specific defaults:
#   - POLICY_TYPE=diffusion (no pretrained base)
#   - resize_shape=240×320 (avoids OOM at native 480×640 on batch=32)
#   - pyav video backend (torchcodec long-runs segfault)
#
# Usage:
#   bash scripts/train/diffusion_policy/train.sh
#
# All knobs from lerobot_finetune.sh still apply (STEPS, BATCH_SIZE, etc.).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

POLICY_TYPE="${POLICY_TYPE:-diffusion}"
DATASET_REPO_ID="${DATASET_REPO_ID:-LightwheelAI/leisaac-pick-orange}"
OUTPUT_NAME="${OUTPUT_NAME:-dp-leisaac-pick-orange}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
EXTRA_ARGS="${EXTRA_ARGS:---dataset.video_backend=pyav --policy.resize_shape=[240,320]}"

export POLICY_TYPE DATASET_REPO_ID OUTPUT_NAME STEPS BATCH_SIZE NUM_WORKERS SAVE_FREQ EXTRA_ARGS

exec bash "${REPO_ROOT}/scripts/training/lerobot_finetune.sh"
