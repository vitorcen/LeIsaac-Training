#!/usr/bin/env bash
# ACT (Action Chunking Transformer) from-scratch training.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

POLICY_TYPE="${POLICY_TYPE:-act}"
DATASET_REPO_ID="${DATASET_REPO_ID:-LightwheelAI/leisaac-pick-orange}"
OUTPUT_NAME="${OUTPUT_NAME:-act-leisaac-pick-orange}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EXTRA_ARGS="${EXTRA_ARGS:---dataset.video_backend=pyav}"

export POLICY_TYPE DATASET_REPO_ID OUTPUT_NAME STEPS BATCH_SIZE SAVE_FREQ EXTRA_ARGS

exec bash "${REPO_ROOT}/scripts/finetune/lerobot_finetune.sh"
