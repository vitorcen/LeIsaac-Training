#!/usr/bin/env bash
# Quick-eval one lerobot-diffusion ckpt, then guarantee the GPU is freed.
#
# After run_one.sh returns, Isaac releases its own VRAM but two things can keep
# the 4090 occupied: a hung headless-teardown policy_inference process, and the
# persistent lerobot async server (:8080, ~2.7 GB). This wrapper kills both so
# the card returns to ~0 between checkpoints (next eval's run_one re-spawns the
# server automatically).
#
# Usage: dp_quick_eval.sh <slug> <ckpt_dir> [label]
# Env:   LEROBOT_PYTHON (default lerobot-v044), EVAL_ROUNDS=5, EPISODE_LENGTH_S=60,
#        MAX_ROUND_WALL_S=90, HORIZON=8
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLUG="${1:?slug required}"
CKPT="${2:?ckpt dir required}"
LABEL="${3:-$SLUG}"

LEROBOT_PYTHON="${LEROBOT_PYTHON:-$HOME/miniconda3/envs/lerobot-v044/bin/python}"
# lerobot-diffusion (stock DDPM) or lerobot-flowdp / lerobot-flowact (FlowHeads,
# flow Euler sampler — needs the get_policy_class/SUPPORTED_POLICIES patch in the
# lerobot-v044 env). run_one only auto-disables the stuck detector for the first
# two, so force it off here for every chunked DP/flow-family policy.
POLICY_TYPE="${POLICY_TYPE:-lerobot-diffusion}"

_run_eval() {
    env LEROBOT_PYTHON="$LEROBOT_PYTHON" \
        LEISAAC_DISABLE_RETRACT_DETECT=1 \
        EVAL_ROUNDS="${EVAL_ROUNDS:-5}" \
        EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-60}" \
        MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}" \
        STUCK_WINDOW_S="${STUCK_WINDOW_S:-99999}" \
        STUCK_EPS_RAD="${STUCK_EPS_RAD:-0}" \
        bash "$ROOT_DIR/scripts/benchmark/run_one.sh" \
            "$SLUG" "$POLICY_TYPE" "${HORIZON:-8}" "$CKPT" lerobot "$LABEL"
}
# The lerobot async server intermittently segfaults at startup on this env's
# python 3.10 build (import-torch C-stack overflow, ~non-deterministic; see
# memory wallx-env-py310-torch-segfault). If run_one fails *before any episode
# ran* (server never came up), retry — a fresh start usually succeeds.
EVAL_RC=0
for attempt in 1 2 3 4 5 6; do
    _run_eval
    EVAL_RC=$?
    BENCH_LOG="$ROOT_DIR/logs/bench-${SLUG}.log"
    if [ "$EVAL_RC" -eq 0 ]; then break; fi
    if grep -qiE 'Evaluating episode|Final success' "$BENCH_LOG" 2>/dev/null; then
        break  # the eval actually ran; rc!=0 is a real result, not a server crash
    fi
    echo "[dp_quick_eval] $SLUG attempt $attempt failed before eval started (likely server segfault); retrying"
    bash "$ROOT_DIR/scripts/policy_server.sh" stop lerobot 2>/dev/null || true
    sleep 3
done

# --- guaranteed GPU cleanup ---------------------------------------------------
# pgrep excludes its own PID, and this wrapper's argv does not contain the
# pattern, so there is no self-match (unlike an inline pkill on the shell CLI).
for pid in $(pgrep -f 'evaluation/policy_inference\.py' 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
done
bash "$ROOT_DIR/scripts/policy_server.sh" stop lerobot 2>/dev/null || true
sleep 2
echo "[dp_quick_eval] $SLUG done (rc=$EVAL_RC). GPU now:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
exit "$EVAL_RC"
