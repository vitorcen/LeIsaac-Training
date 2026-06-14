#!/usr/bin/env bash
# Auto-advance a flow/DP quick-eval chain without waiting on a patrol cycle.
#
# Walks every checkpoint dir under EVAL_DIR (sorted), and for each one that has
# no metrics yet: waits until the local GPU is free (shared with the starvla
# eval queue) and no eval is running, then runs dp_quick_eval (which itself
# retries the intermittently-segfaulting server). Skips checkpoints whose
# metrics already exist, so it's safe to (re)start at any time and it coexists
# with a manually-launched eval. Exits when every checkpoint has a result.
#
# Usage:
#   POLICY_TYPE=lerobot-flowdp SLUG_PREFIX=flowdp EVAL_DIR=outputs/flowdp-eval \
#     HORIZON=8 nohup bash scripts/evaluation/flow_sweep_watcher.sh &
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_TYPE="${POLICY_TYPE:-lerobot-flowdp}"
SLUG_PREFIX="${SLUG_PREFIX:-flowdp}"
EVAL_DIR="${EVAL_DIR:?EVAL_DIR required (e.g. outputs/flowdp-eval)}"
HORIZON="${HORIZON:-8}"
GPU_FREE_MB="${GPU_FREE_MB:-2000}"      # only launch when used < this (avoid starvla contention)
RESULTS="$ROOT_DIR/results/benchmark"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
# Busy if any eval is mid-flight — the Isaac client (policy_inference, also used
# by the starvla queue) OR a run_one wrapper still in its server-startup phase
# (GPU not yet allocated, so the gpu_used gate alone would miss it). This script
# launches evals synchronously, so neither pattern ever matches its own subtree
# while it sits in the wait loop.
eval_busy() {
    pgrep -f 'evaluation/policy_inference\.py' >/dev/null 2>&1 && return 0
    pgrep -f 'benchmark/run_one\.sh' >/dev/null 2>&1 && return 0
    return 1
}

echo "[sweep] start prefix=$SLUG_PREFIX type=$POLICY_TYPE dir=$EVAL_DIR"
while true; do
    pending=0
    for d in $(ls -1d "$ROOT_DIR/$EVAL_DIR"/*/ 2>/dev/null | sort); do
        [ -f "${d}model.safetensors" ] || continue
        step="$(basename "$d")"
        sn=$((10#$step))                        # strip leading zeros
        slug="${SLUG_PREFIX}-${sn}"
        [ -f "$RESULTS/${slug}.metrics.json" ] && continue
        pending=1
        # wait for a free GPU + idle eval slot (coarse gate against the starvla queue,
        # which exposes its eval via policy_inference/serve_starvla too)
        while [ "$(gpu_used)" -gt "$GPU_FREE_MB" ] || eval_busy; do sleep 20; done
        # Hold the shared single-GPU lock for the whole eval. Two flow watchers (dit +
        # flowditx) can both clear the GPU gate in the same instant and double-book the
        # 4090; flock serializes them on the same lock the starvla eval_queue uses. The
        # metrics re-check lives INSIDE the lock so the loser of the race skips a ckpt the
        # winner just finished instead of re-evaluating it.
        ( flock 200
          [ -f "$RESULTS/${slug}.metrics.json" ] && exit 0
          # GPU may now be busy with the lock holder we just waited behind; re-gate.
          while [ "$(gpu_used)" -gt "$GPU_FREE_MB" ] || eval_busy; do sleep 20; done
          echo "[sweep] $(date +%H:%M:%S) eval $slug"
          POLICY_TYPE="$POLICY_TYPE" HORIZON="$HORIZON" \
              bash "$ROOT_DIR/scripts/evaluation/dp_quick_eval.sh" "$slug" "${d%/}" "$slug" \
              >> "/tmp/sweep_${SLUG_PREFIX}.log" 2>&1 || true
        ) 200>/tmp/leisaac_gpu_eval.lock
        sleep 5
    done
    [ "$pending" -eq 0 ] && break
    sleep 15
done
echo "[sweep] $(date +%H:%M:%S) ${SLUG_PREFIX} ALL DONE"
