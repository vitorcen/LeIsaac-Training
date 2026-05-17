#!/usr/bin/env bash
# Wrapper for policy_inference.py with user-patience-based wall-clock timeout.
#
# Philosophy: GR00T N1.5 completes 3 oranges in ~30s wall-clock. If a model
# can't finish a round in 90s wall-clock, it's not deployable. Don't expand
# the timeout to let slow models limp through — let them fail and report it.
#
# Formula:
#   total_timeout = STARTUP + n_rounds * TIMEOUT_PER_ROUND
# Defaults:
#   STARTUP=60s (Isaac sim cold start)
#   TIMEOUT_PER_ROUND=90s (GR00T 30s baseline × 3 tolerance)
#
# Optional probe step (informational only): measures recent server inference_ms
# and effective_chunk so the user sees if a model is realtime-feasible at all.
#
# Usage:
#   bash run_eval.sh [--probe-only] -- <all policy_inference.py args>
#
# Example:
#   bash run_eval.sh -- \
#       --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=3 \
#       --episode_length_s=120 --step_hz=60 \
#       --policy_type=lerobot-diffusion --policy_host=127.0.0.1 --policy_port=8080 \
#       --policy_checkpoint_path=/path/to/ckpt \
#       --policy_action_horizon=16 --device=cuda --enable_cameras

set -euo pipefail

LOG_TAIL="${LEROBOT_SERVER_LOG:-/home/david/work/isaaclab-experience/logs/lerobot_server.log}"
STARTUP="${TIMEOUT_STARTUP:-60}"             # seconds for Isaac sim cold start
TIMEOUT_PER_ROUND="${TIMEOUT_PER_ROUND:-90}" # seconds per eval round (user patience)

# Parse out our own args; pass the rest to policy_inference.py
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; PASS_ARGS+=("$@"); break;;
        *)  PASS_ARGS+=("$1"); shift;;
    esac
done

# Extract relevant args for the timeout calculation
n_rounds=3
episode_s=120
step_hz=60
horizon=16
ckpt_path=""
for ((i=0; i<${#PASS_ARGS[@]}; i++)); do
    arg="${PASS_ARGS[i]}"
    case "$arg" in
        --eval_rounds=*) n_rounds="${arg#*=}";;
        --eval_rounds)   n_rounds="${PASS_ARGS[i+1]}";;
        --episode_length_s=*) episode_s="${arg#*=}";;
        --episode_length_s)   episode_s="${PASS_ARGS[i+1]}";;
        --step_hz=*) step_hz="${arg#*=}";;
        --step_hz)   step_hz="${PASS_ARGS[i+1]}";;
        --policy_action_horizon=*) horizon="${arg#*=}";;
        --policy_action_horizon)   horizon="${PASS_ARGS[i+1]}";;
        --policy_checkpoint_path=*) ckpt_path="${arg#*=}";;
        --policy_checkpoint_path)   ckpt_path="${PASS_ARGS[i+1]}";;
    esac
done

# The effective chunk size at runtime = min(client horizon, model n_action_steps).
# Read model's n_action_steps from ckpt config when possible; fallback to horizon.
chunk_from_ckpt=""
if [[ -n "$ckpt_path" && -f "$ckpt_path/config.json" ]]; then
    chunk_from_ckpt=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(c.get('n_action_steps') or c.get('chunk_size') or '')" "$ckpt_path/config.json" 2>/dev/null || true)
fi
if [[ -n "$chunk_from_ckpt" && "$chunk_from_ckpt" -lt "$horizon" ]]; then
    effective_chunk="$chunk_from_ckpt"
else
    effective_chunk="$horizon"
fi

# Pull the most recent inference latency from server log (last 50 chunks).
# Fallback to 50ms (fast model) if no recent stats.
inference_ms=""
if [[ -f "$LOG_TAIL" ]]; then
    # `|| true` is critical: empty match under set -o pipefail would kill the script
    inference_ms=$( { grep "Total time:" "$LOG_TAIL" || true; } | tail -50 | \
        { grep -oE "[0-9]+\.[0-9]+ms" || true; } | \
        awk '{s+=$1; n++} END{if(n>0)printf "%.1f", s/n; else print ""}')
fi
[[ -z "$inference_ms" ]] && inference_ms=50.0

# Realtime feasibility check (informational only): does this model keep up with sim?
realtime_per_step_ms=$(awk "BEGIN{printf \"%.2f\", 1000.0/$step_hz}")
slowdown=$(awk "BEGIN{r = $inference_ms / ($effective_chunk * $realtime_per_step_ms); printf \"%.2f\", (r > 1.0) ? r : 1.0}")
# Wall-clock budget: user patience, not slowdown-padded.
total=$(( STARTUP + n_rounds * TIMEOUT_PER_ROUND ))

echo "[run_eval] inference probe: ${inference_ms}ms/chunk, client horizon=${horizon}, ckpt n_action_steps=${chunk_from_ckpt:-unknown}, effective_chunk=${effective_chunk}, step_hz=${step_hz}"
echo "[run_eval] realtime slowdown: ${slowdown}x (>2x = model too slow for 60Hz control)"
echo "[run_eval] timeout: STARTUP=${STARTUP}s + ${n_rounds} rounds × ${TIMEOUT_PER_ROUND}s/round = ${total}s"
if awk "BEGIN{exit !($slowdown > 2.0)}"; then
    echo "[run_eval] WARNING: slowdown ${slowdown}x means each sim step takes >2x realtime."
    echo "[run_eval] WARNING: model may not finish a meaningful trajectory within the wall-clock budget."
    echo "[run_eval] WARNING: consider DDIM sampling for diffusion, fp16 inference, or accept it as a deployment-fail signal."
fi

# Bail out early on dry-run
for a in "${PASS_ARGS[@]}"; do [[ "$a" == "--probe-only" ]] && { echo "[run_eval] probe-only mode, not launching"; exit 0; }; done

cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # → LeIsaac/
exec timeout "${total}" conda run -n "${CONDA_ENV:-isaaclab}" --no-capture-output \
    python -u scripts/evaluation/policy_inference.py "${PASS_ARGS[@]}"
