#!/usr/bin/env bash
# Auto quick-eval watcher per `LeIsaac/CLAUDE.md` training rule.
#
# Polls $OUTPUT_DIR/checkpoints/ for new <step>/pretrained_model dirs and runs
# X-VLA-style 3-round 60s quick eval on each one. Writes per-ckpt result to
# $OUTPUT_DIR/auto_eval.csv. Designed to be spawned in background by
# lerobot_finetune.sh (AUTO_EVAL=1, default on).
#
# Abort logic: after STOP_AFTER_FAIL consecutive 0-orange slices, writes
# $OUTPUT_DIR/.eval_abort marker — the training wrapper polls this and SIGTERMs
# the train process so we do not burn another N hours on a broken config.
#
# Required env:
#   OUTPUT_DIR        absolute path to training output dir (with checkpoints/)
#   POLICY_TYPE       inference-side slug: lerobot-diffusion / lerobot-act /
#                     lerobot-smolvla   (NOT the train POLICY_TYPE slug)
#
# Optional env:
#   EVAL_HORIZON      policy_action_horizon (auto:
#                     act=70 / diffusion=8 / smolvla=50, override otherwise)
#   LEROBOT_PYTHON    python for the lerobot policy_server (default: current PATH)
#   LEROBOT_REPO      lerobot repo (default: $HOME/work/lerobot-v040 for v0.4
#                     ckpts; override for v0.5+)
#   POLL_S            ckpt-dir poll interval, seconds                (default 30)
#   EVAL_ROUNDS                                                       (default 3)
#   EPISODE_LENGTH_S                                                  (default 60)
#   MAX_ROUND_WALL_S                                                  (default 90)
#   STEP_HZ                                                           (default 30)
#   STOP_AFTER_FAIL   consecutive 0-orange slices → abort             (default 3)
#   START_STEP        skip ckpts with step < this (resume mode)       (default 0)
#   PROMPT            language instruction                            (default pick-orange)
#   MAX_WAIT_S        give up if no new ckpt for this long            (default 7200)

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required (absolute training output dir)}"
POLICY_TYPE="${POLICY_TYPE:?POLICY_TYPE required, e.g. lerobot-diffusion}"

case "$POLICY_TYPE" in
    lerobot-act)        DEFAULT_H=70 ;;
    lerobot-diffusion)  DEFAULT_H=8  ;;
    lerobot-smolvla)    DEFAULT_H=50 ;;
    *)                  DEFAULT_H=16 ;;
esac
EVAL_HORIZON="${EVAL_HORIZON:-$DEFAULT_H}"

LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/david/miniconda3/envs/lerobot-v040/bin/python}"
LEROBOT_REPO="${LEROBOT_REPO:-$HOME/work/lerobot-v040}"

POLL_S="${POLL_S:-30}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
STEP_HZ="${STEP_HZ:-30}"
STOP_AFTER_FAIL="${STOP_AFTER_FAIL:-3}"
START_STEP="${START_STEP:-0}"
PROMPT="${PROMPT:-Pick up the orange and place it on the plate}"
MAX_WAIT_S="${MAX_WAIT_S:-7200}"

RESULTS_CSV="$OUTPUT_DIR/auto_eval.csv"
ABORT_MARKER="$OUTPUT_DIR/.eval_abort"
WATCHER_LOG="$OUTPUT_DIR/auto_eval.log"

mkdir -p "$OUTPUT_DIR"
[[ -f "$RESULTS_CSV" ]] || echo "step,oranges,total_oranges,rounds_success,rounds_total,avg_round_s,note" > "$RESULTS_CSV"

log() { echo "[$(date '+%H:%M:%S')] [watcher] $*" | tee -a "$WATCHER_LOG"; }

log "output_dir=$OUTPUT_DIR"
log "policy=$POLICY_TYPE horizon=$EVAL_HORIZON server_python=$LEROBOT_PYTHON"
log "eval: rounds=$EVAL_ROUNDS ep_s=$EPISODE_LENGTH_S wall=$MAX_ROUND_WALL_S step_hz=$STEP_HZ"
log "poll=$POLL_S abort_after=$STOP_AFTER_FAIL start_step=$START_STEP"
log "csv → $RESULTS_CSV"

start_lerobot_server() {
    if ss -tlnp 2>/dev/null | grep -q ':8080 '; then
        return 0
    fi
    log "starting lerobot policy_server (port 8080)"
    cd "$LEROBOT_REPO"
    nohup "$LEROBOT_PYTHON" -m lerobot.async_inference.policy_server \
        --host 0.0.0.0 --port 8080 \
        > "$OUTPUT_DIR/policy_server.log" 2>&1 &
    disown $!
    cd - >/dev/null
    for _ in $(seq 1 30); do
        sleep 2
        ss -tlnp 2>/dev/null | grep -q ':8080 ' && return 0
    done
    log "WARN: server did not bind :8080 after 60s"
    return 1
}

eval_one_ckpt() {
    local step="$1"
    local ckpt_path="$2"
    local slug
    slug="$(basename "$OUTPUT_DIR")-${step}-h${EVAL_HORIZON}"

    log "===== step=$step (slug=$slug) ====="
    start_lerobot_server || { log "skip step=$step (no server)"; return 1; }

    cd "$REPO_ROOT"
    set +e
    EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH_S="$EPISODE_LENGTH_S" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" STEP_HZ="$STEP_HZ" \
        PROMPT="$PROMPT" \
        bash scripts/benchmark/run_one.sh \
            "$slug" "$POLICY_TYPE" "$EVAL_HORIZON" \
            "$ckpt_path" "lerobot" "$slug" \
        >> "$WATCHER_LOG" 2>&1
    set -e

    local metrics_json="$REPO_ROOT/results/benchmark/${slug}.metrics.json"
    if [[ ! -f "$metrics_json" ]]; then
        log "step=$step: no metrics produced"
        echo "$step,0,$((EVAL_ROUNDS * 3)),0,$EVAL_ROUNDS,NaN,no-metrics" >> "$RESULTS_CSV"
        return 1
    fi

    local oranges total rounds_s rounds_t avg
    read -r oranges total rounds_s rounds_t avg < <(
        "$LEROBOT_PYTHON" - <<PY "$metrics_json"
import json, sys
m = json.load(open(sys.argv[1]))
oranges = m.get('oranges_placed_strict', m.get('oranges_placed_total', 0))
total = m.get('oranges_max_total', $((EVAL_ROUNDS * 3)))
r_s = m.get('rounds_success_strict', m.get('rounds_success', 0))
r_t = m.get('rounds', $EVAL_ROUNDS)
avg = m.get('avg_round_s', 'NaN')
print(oranges, total, r_s, r_t, avg)
PY
    )
    echo "$step,$oranges,$total,$rounds_s,$rounds_t,$avg,ok" >> "$RESULTS_CSV"
    log "step=$step ➜ oranges=$oranges/$total rounds=$rounds_s/$rounds_t avg=${avg}s"
    return 0
}

# --- main loop ---
declare -A seen
fail_streak=0
last_progress=$(date +%s)

while true; do
    if [[ -f "$ABORT_MARKER" ]]; then
        log "abort marker present, exiting"
        break
    fi

    new_ckpt_count=0
    for ck in "$OUTPUT_DIR"/checkpoints/*/pretrained_model; do
        [[ -d "$ck" ]] || continue
        step_str="$(basename "$(dirname "$ck")")"
        [[ "$step_str" =~ ^[0-9]+$ ]] || continue
        step=$((10#$step_str))
        [[ "$step" -lt "$START_STEP" ]] && continue
        [[ -n "${seen[$step]:-}" ]] && continue

        # skip if already in CSV (resume safety)
        if grep -qE "^${step}," "$RESULTS_CSV"; then
            seen[$step]=1
            continue
        fi

        if eval_one_ckpt "$step" "$ck"; then
            new_ckpt_count=$((new_ckpt_count + 1))
            last_oranges=$(tail -1 "$RESULTS_CSV" | cut -d, -f2)
            if [[ "$last_oranges" == "0" ]]; then
                fail_streak=$((fail_streak + 1))
                log "0-orange streak now $fail_streak/$STOP_AFTER_FAIL"
                if [[ "$fail_streak" -ge "$STOP_AFTER_FAIL" ]]; then
                    log "ABORT: $STOP_AFTER_FAIL consecutive 0-orange slices"
                    touch "$ABORT_MARKER"
                    seen[$step]=1
                    break 2
                fi
            else
                fail_streak=0
            fi
        fi
        seen[$step]=1
        last_progress=$(date +%s)
    done

    if [[ -f "$OUTPUT_DIR/checkpoints/last/pretrained_model/config.json" ]]; then
        local_done=1
    else
        local_done=0
    fi

    now=$(date +%s)
    if (( now - last_progress > MAX_WAIT_S )); then
        log "no new ckpt for ${MAX_WAIT_S}s, exiting"
        break
    fi
    sleep "$POLL_S"
done

log "watcher exit"
