#!/usr/bin/env bash
# Strict 20-round eval of the *best* ckpt of each FlowHeads generative variant,
# to settle the conclusion that quick-eval (5-round, wall_cap 90s) only sampled
# optimistically. Strict params: 20 rounds, EPISODE_LENGTH_S=120, wall_cap 180s
# (quick-eval was truncating episodes at 90s -> skip_reason=wall_cap).
#
# Serial on one 4090, gated against the neighbour starvla eval via the shared
# flock /tmp/leisaac_gpu_eval.lock + a GPU-free + no-eval-busy wait. Safe to
# (re)start: a ckpt whose <slug>.metrics.json already exists is skipped.
#
# Usage: nohup setsid bash scripts/evaluation/strict20_best.sh >/tmp/strict20.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GPU_FREE_MB="${GPU_FREE_MB:-2000}"
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
# Busy if any eval is mid-flight: our own client, the neighbour starvla serve,
# or a run_one wrapper still in server-startup (GPU not yet allocated).
busy(){
  pgrep -f 'evaluation/policy_inference\.py' >/dev/null 2>&1 && return 0
  pgrep -f 'serve_starvla\.py'               >/dev/null 2>&1 && return 0
  pgrep -f 'benchmark/run_one\.sh'           >/dev/null 2>&1 && return 0
  return 1
}

# fam best-step policy-type ; HORIZON=8 for all (as trained/quick-evaled)
JOBS=(
  "flowdp 9800  lerobot-flowdp"
  "dit    14000 lerobot-dit"
  "flowditx 11200 lerobot-flowditx"
)

echo "[strict20] start $(date +%FT%T)"
for spec in "${JOBS[@]}"; do
  read -r fam step pt <<<"$spec"
  d=$(printf "%s/outputs/%s-eval/%06d" "$ROOT" "$fam" "$step")
  slug="${fam}-${step}-s20"
  if [ ! -f "${d}/model.safetensors" ]; then
    echo "[strict20] MISSING ckpt $d — skip $slug"; continue
  fi
  if [ -f "$ROOT/results/benchmark/${slug}.metrics.json" ]; then
    echo "[strict20] already done $slug — skip"; continue
  fi
  # hold the shared single-GPU lock for the whole 20-round run
  ( flock 200
    [ -f "$ROOT/results/benchmark/${slug}.metrics.json" ] && exit 0
    while [ "$(gpu_used)" -gt "$GPU_FREE_MB" ] || busy; do sleep 20; done
    echo "[strict20] $(date +%T) === $slug (best of $fam @ step$step) ==="
    POLICY_TYPE="$pt" HORIZON=8 \
      EVAL_ROUNDS=20 EPISODE_LENGTH_S=120 MAX_ROUND_WALL_S=180 \
      bash "$ROOT/scripts/evaluation/dp_quick_eval.sh" "$slug" "$d" "$slug"
  ) 200>/tmp/leisaac_gpu_eval.lock
  sleep 5
done
echo "[strict20] ALL DONE $(date +%FT%T)"
