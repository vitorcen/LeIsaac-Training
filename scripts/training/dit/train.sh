#!/usr/bin/env bash
# DiT Policy from-scratch training (multi_task_dit policy type).
#
# Thin wrapper around scripts/training/lerobot_finetune.sh.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

POLICY_TYPE="${POLICY_TYPE:-multi_task_dit}"
DATASET_REPO_ID="${DATASET_REPO_ID:-LightwheelAI/leisaac-pick-orange}"
OUTPUT_NAME="${OUTPUT_NAME:-dit-leisaac-pick-orange}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
EXTRA_ARGS="${EXTRA_ARGS:---dataset.video_backend=pyav}"

export POLICY_TYPE DATASET_REPO_ID OUTPUT_NAME STEPS BATCH_SIZE SAVE_FREQ EXTRA_ARGS

exec bash "${REPO_ROOT}/scripts/training/lerobot_finetune.sh"
