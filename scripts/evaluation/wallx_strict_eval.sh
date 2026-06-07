#!/usr/bin/env bash
# Stage-2 STRICT eval for the sweep winner: 20-round on a single wall-x ckpt,
# plus a placed-count distribution P(placed=k). Run after wallx_sweep_watcher.sh
# picks a winner. Usage:
#   wallx_strict_eval.sh <ckpt_dir>            # e.g. outputs/wallx-sweep/2
#   ROUNDS=20 wallx_strict_eval.sh <ckpt_dir>
#
# Writes <ckpt_dir>/strict_eval.json (metrics) via policy_inference --metrics_out
# and echoes a one-line summary. GUI off (headless) for speed.
set -uo pipefail

CKPT="${1:?usage: wallx_strict_eval.sh <ckpt_dir>}"
CKPT="$(cd "$CKPT" && pwd)"
ROOT=/home/david/work/isaaclab-experience
BASE=/home/david/.cache/huggingface/hub/models--x-square-robot--wall-oss-0.5/snapshots/f2119fd2bc888c249ed42a4004f42dc09ed1fa84
WALLX_PY=/home/david/miniconda3/envs/wallx/bin/python
SERVE=$ROOT/LeIsaac/scripts/evaluation/serve_wallx.py
PROMPT="Pick three oranges and put them into the plate, then reset the arm to rest state."

PORT="${PORT:-8002}"
ROUNDS="${ROUNDS:-20}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-120}"
STEP_HZ="${STEP_HZ:-60}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
METRICS="$CKPT/strict_eval.json"

[ -f "$CKPT/model.safetensors" ] || { echo "no model.safetensors in $CKPT"; exit 1; }

echo "[strict] serving $CKPT on :$PORT"
rm -f /tmp/strict_serve.log
nohup env CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$WALLX_PY" -u "$SERVE" --ckpt "$CKPT" --base "$BASE" --port "$PORT" --prompt "$PROMPT" \
  > /tmp/strict_serve.log 2>&1 &
SPID=$!
up=0
for _ in $(seq 1 120); do
  grep -q "serving on ws" /tmp/strict_serve.log && { up=1; break; }
  kill -0 "$SPID" 2>/dev/null || break
  sleep 2
done
[ "$up" = 1 ] || { echo "[strict] serve FAILED (/tmp/strict_serve.log)"; kill -9 "$SPID" 2>/dev/null; exit 1; }

echo "[strict] ${ROUNDS}-round eval (episode=${EPISODE_LENGTH_S}s)..."
( cd "$ROOT/LeIsaac" && unset DISPLAY && conda run -n isaaclab --no-capture-output \
  python -u scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds="$ROUNDS" --episode_length_s="$EPISODE_LENGTH_S" --step_hz="$STEP_HZ" \
    --policy_type=wallx --policy_host=localhost --policy_port="$PORT" --policy_timeout_ms=60000 \
    --policy_action_horizon="$ACTION_HORIZON" \
    --policy_language_instruction="$PROMPT" \
    --device=cuda --headless --enable_cameras \
    --metrics_out="$METRICS" ) 2>&1 | tee /tmp/strict_eval.log | grep -iE "success rate|oranges" | tail -5

kill -9 "$SPID" 2>/dev/null
echo "[strict] metrics -> $METRICS"
grep "Final success rate" /tmp/strict_eval.log | tail -1
