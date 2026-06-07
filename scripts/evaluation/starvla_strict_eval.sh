#!/usr/bin/env bash
# STRICT 20-round eval for a single StarVLA ckpt + placed-count distribution P(placed=k).
# Mirrors wallx_strict_eval.sh, but serves via serve_starvla.py and supports VLM
# quantization (VLM_QUANT=8|4) so the same ckpt can be benchmarked bf16 vs int8/nf4 —
# the apples-to-apples way to prove "8bit ≈ bf16" instead of eyeballing 2 episodes.
#
# Usage:
#   VLM_QUANT=8 starvla_strict_eval.sh <ckpt.pt>       # 20-round, VLM int8
#   VLM_QUANT=0 ROUNDS=20 starvla_strict_eval.sh <ckpt.pt>   # bf16 baseline
#   GUI=1 VLM_QUANT=8 starvla_strict_eval.sh <ckpt.pt>  # watch the window (slower)
#
# Writes <ckpt_dir>/strict_eval_q<QUANT>.json (per-round metrics) + a distribution
# .md/.svg next to it via aggregate_distribution.py. Headless by default (speed).
set -uo pipefail

CKPT="${1:?usage: starvla_strict_eval.sh <ckpt.pt>}"
CKPT="$(readlink -f "$CKPT")"
[ -f "$CKPT" ] || { echo "no such ckpt: $CKPT"; exit 1; }
ROOT=/home/david/work/isaaclab-experience
BASE="${BASE:-/home/david/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
STARVLA_PY=/home/david/miniconda3/envs/starvla_eval/bin/python
SERVE=$ROOT/LeIsaac/scripts/evaluation/serve_starvla.py
AGG=$ROOT/scripts/benchmark/aggregate_distribution.py
PROMPT="${PROMPT:-Grab orange and place into plate}"

PORT="${PORT:-8013}"
ROUNDS="${ROUNDS:-20}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-120}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-180}"
STEP_HZ="${STEP_HZ:-30}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
STUCK_WINDOW_S="${STUCK_WINDOW_S:-30}"
STUCK_EPS_RAD="${STUCK_EPS_RAD:-0.05}"
VLM_QUANT="${VLM_QUANT:-8}"
GUI="${GUI:-0}"

CKDIR="$(dirname "$CKPT")"
METRICS="$CKDIR/strict_eval_q${VLM_QUANT}.json"
LABEL="StarVLA-4B-q${VLM_QUANT}"

# quant env for serve
QENV=()
[ "$VLM_QUANT" = 8 ] && QENV=(STARVLA_VLM_8BIT=1)
[ "$VLM_QUANT" = 4 ] && QENV=(STARVLA_VLM_4BIT=1)

echo "[strict] serving $CKPT q=${VLM_QUANT} on :$PORT"
rm -f /tmp/sv_strict_serve.log
nohup env CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${QENV[@]}" \
  "$STARVLA_PY" -u "$SERVE" --ckpt "$CKPT" --base "$BASE" --port "$PORT" --img_size 448 --prompt "$PROMPT" \
  > /tmp/sv_strict_serve.log 2>&1 &
SPID=$!
up=0
for _ in $(seq 1 150); do
  grep -q "SERVE_READY" /tmp/sv_strict_serve.log 2>/dev/null && { up=1; break; }
  ss -tln 2>/dev/null | grep -q ":$PORT " && { up=1; break; }
  kill -0 "$SPID" 2>/dev/null || { echo "[strict] serve died"; tail -15 /tmp/sv_strict_serve.log; exit 1; }
  sleep 2
done
[ "$up" = 1 ] || { echo "[strict] serve TIMEOUT"; tail -15 /tmp/sv_strict_serve.log; kill -9 "$SPID"; exit 1; }
grep -i 'quantized' /tmp/sv_strict_serve.log | tail -1
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1

RENDER=(--headless)
[ "$GUI" = 1 ] && RENDER=()
echo "[strict] ${ROUNDS}-round eval (q=${VLM_QUANT}, ep=${EPISODE_LENGTH_S}s, GUI=${GUI})..."
( cd "$ROOT/LeIsaac" && DISPLAY="${DISPLAY:-:0}" conda run -n isaaclab --no-capture-output \
  python -u scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds="$ROUNDS" --episode_length_s="$EPISODE_LENGTH_S" --step_hz="$STEP_HZ" \
    --max_round_wall_s="$MAX_ROUND_WALL_S" --stuck_window_s="$STUCK_WINDOW_S" --stuck_eps_rad="$STUCK_EPS_RAD" \
    --policy_type=starvla --policy_host=localhost --policy_port="$PORT" --policy_timeout_ms=60000 \
    --policy_action_horizon="$ACTION_HORIZON" --policy_language_instruction="$PROMPT" \
    --metrics_out="$METRICS" --metrics_label="$LABEL" \
    --device=cuda "${RENDER[@]}" --enable_cameras ) 2>&1 | tee /tmp/sv_strict_eval.log | grep -iE "success rate|oranges|Episode [0-9]" | tail -8

kill -9 "$SPID" 2>/dev/null; sleep 3
echo "[strict] === result ==="
grep "Final success rate" /tmp/sv_strict_eval.log | tail -1
if [ -f "$METRICS" ]; then
  python3 "$AGG" "$METRICS" --out "${METRICS%.json}.distribution.md" --svg "${METRICS%.json}.distribution.svg"
  echo "[strict] distribution -> ${METRICS%.json}.distribution.md"
  cat "${METRICS%.json}.distribution.md"
else
  echo "[strict] NO metrics produced — eval crashed (see /tmp/sv_strict_eval.log)"
fi
