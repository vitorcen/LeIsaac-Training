#!/bin/bash
# Convert LeIsaac SO-101 PickOrange v2-gr00t dataset to DreamZero GEAR format.
#
# This is metadata-only conversion (no parquet/video changes); ~1 min runtime.
# Generates:
#   - meta/embodiment.json   (tag: xdof)
#   - meta/stats.json        (mean/std/min/max/q01/q99 for state+action)
#   - meta/relative_stats_dreamzero.json
#   - meta/modality.json     (re-written with joint_pos/gripper_pos split for DreamZero)
#
# Existing files preserved: episodes.jsonl, tasks.jsonl, info.json.
# Original LeIsaac modality.json backed up to modality.json.gr00t-bak.
#
# Usage (on cloud):
#   bash convert_leisaac_to_gear.sh /root/autodl-tmp/leisaac-pick-orange

set -e
DATASET_PATH=${1:-/root/autodl-tmp/leisaac-pick-orange}
DREAMZERO_REPO=${DREAMZERO_REPO:-/root/autodl-tmp/dreamzero-repo}
PYTHON=${PYTHON:-$(command -v python || command -v python3 || echo /root/miniconda3/bin/python)}

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: dataset not found at $DATASET_PATH"
    exit 1
fi

# Backup original modality.json (used by GR00T pipelines)
if [ -f "$DATASET_PATH/meta/modality.json" ] && [ ! -f "$DATASET_PATH/meta/modality.json.gr00t-bak" ]; then
    cp "$DATASET_PATH/meta/modality.json" "$DATASET_PATH/meta/modality.json.gr00t-bak"
    echo "Backed up modality.json -> modality.json.gr00t-bak"
fi

# SO-101 6-DoF schema: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
#   indices 0-4 = arm joints,  index 5 = gripper
"$PYTHON" "$DREAMZERO_REPO/scripts/data/convert_lerobot_to_gear.py" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag xdof \
    --state-keys '{"joint_pos": [0, 5], "gripper_pos": [5, 6]}' \
    --action-keys '{"joint_pos": [0, 5], "gripper_pos": [5, 6]}' \
    --relative-action-keys joint_pos \
    --task-key task \
    --force

echo "--- GEAR metadata generated ---"
ls -la "$DATASET_PATH/meta/"
