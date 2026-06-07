#!/bin/bash
# Smoke eval: wall-x policy (served on :8000) driving the LeIsaac SO-101 PickOrange
# task in the Isaac GUI. Run AFTER serve_wallx.py is up. GUI = no --headless.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> LeIsaac/
export DISPLAY="${DISPLAY:-:0}"
exec conda run -n isaaclab --no-capture-output python -u scripts/evaluation/policy_inference.py \
  --task=LeIsaac-SO101-PickOrange-v0 \
  --eval_rounds="${EVAL_ROUNDS:-1}" \
  --episode_length_s="${EPISODE_LENGTH_S:-60}" \
  --step_hz=60 \
  --policy_type=wallx \
  --policy_host=localhost \
  --policy_port="${POLICY_PORT:-8000}" \
  --policy_timeout_ms=60000 \
  --policy_action_horizon=32 \
  --policy_language_instruction="Pick three oranges and put them into the plate, then reset the arm to rest state." \
  --device=cuda \
  --enable_cameras
