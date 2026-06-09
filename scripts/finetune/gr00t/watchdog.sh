#!/usr/bin/env bash
# GR00T-N1.6 training watchdog: auto-resume on crash.
#
# Background: on 4090 24GB with bf16 + grad-ckpt + per_step=2, training crashes
# with `d.is_cuda() INTERNAL ASSERT FAILED` randomly every 500-700 step.  The
# checkpoint up to that point is intact (HF Trainer saves on save_steps).
# Resume from disk OOMs because the saved optimizer.pt (~3 GB) blows the
# transient-alloc headroom.  Workaround: drop optimizer.pt before each resume,
# HF Trainer reloads model+scheduler only and re-inits optimizer fresh.  The
# loss curve takes ~50 step to recover momentum, but it converges.
#
# Env:
#   MAX_STEPS         absolute training step target (default 5000)
#   SAVE_STEPS        ckpt cadence (default 500)
#   GLOBAL_BATCH      effective batch (default 16)
#   GRAD_ACCUM        grad accumulation (default 8 → per_step=2)
#   MAX_RETRIES       max watchdog cycles before giving up (default 30)
#   OUTPUT_DIR        train output (default LeIsaac/outputs/gr00t-n16-leisaac-pick-orange)
#
# Loop:
#   1. Pick latest checkpoint-N from OUTPUT_DIR (or fresh start if none).
#   2. If resuming: mv that ckpt's optimizer.pt → optimizer.pt.bak (drops Adam momentum).
#   3. Launch train.sh; tee to a per-cycle log.
#   4. On clean exit (train finished MAX_STEPS), break.
#   5. On crash (exit != 0): pkill stale procs, sleep 5s, increment retry, loop.
#
# Stops automatically once HF Trainer sees global_step >= MAX_STEPS in
# trainer_state.json and the resume returns success.

set -uo pipefail

LEISAAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "$LEISAAC_ROOT/.." && pwd)"

MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS_MULTIPLE="${EVAL_STEPS_MULTIPLE:-500}"  # eval ckpts at multiples of this (0 = disabled)
EVAL_FAST_THRESHOLD="${EVAL_FAST_THRESHOLD:-3000}" # step <= this → fast 3-round, else 6-round
EVAL_ROUNDS_FAST="${EVAL_ROUNDS_FAST:-3}"
EVAL_ROUNDS_FULL="${EVAL_ROUNDS_FULL:-6}"
EVAL_EPISODE_S="${EVAL_EPISODE_S:-60}"
EVAL_MAX_ROUND_WALL_S="${EVAL_MAX_ROUND_WALL_S:-90}"
EVAL_ACTION_HORIZON="${EVAL_ACTION_HORIZON:-16}"
EVAL_PROMPT="${EVAL_PROMPT:-Pick up the orange and put it in the plate}"
EVAL_POLICY_PORT="${EVAL_POLICY_PORT:-5555}"
GLOBAL_BATCH="${GLOBAL_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_RETRIES="${MAX_RETRIES:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-$LEISAAC_ROOT/outputs/gr00t-n16-leisaac-pick-orange}"
EVAL_LOG="${EVAL_LOG:-$REPO_ROOT/logs/gr00t_n16_ckpts.csv}"

WATCHDOG_LOG_DIR="$REPO_ROOT/logs/gr00t_watchdog"
mkdir -p "$WATCHDOG_LOG_DIR"
WATCHDOG_LOG="$WATCHDOG_LOG_DIR/watchdog_$(date +%Y%m%d_%H%M%S).log"

echo "[watchdog] target=$MAX_STEPS save_steps=$SAVE_STEPS global=$GLOBAL_BATCH accum=$GRAD_ACCUM"
echo "[watchdog] output=$OUTPUT_DIR" | tee -a "$WATCHDOG_LOG"
echo "[watchdog] log=$WATCHDOG_LOG"

cleanup_procs() {
    # Broad pkill first (covers most cases)
    pkill -f "launch_finetune_ckpt\|gr00t/experiment\|gr00t/eval/run_gr00t_server\|policy_inference" 2>/dev/null
    sleep 3
    pkill -9 -f "launch_finetune_ckpt\|gr00t/experiment\|gr00t/eval/run_gr00t_server\|policy_inference" 2>/dev/null
    sleep 2
    # Bulletproof orphan kill: only target known eval/inference scripts to avoid
    # nuking our own legit training process.  These all leave detached children
    # after nohup that escape process-group pkill.
    for pat in "gr00t/eval/run_gr00t_server" "policy_inference\.py" "scripts/evaluation/policy_inference"; do
        for pid in $(pgrep -f "$pat" 2>/dev/null); do
            kill -9 "$pid" 2>/dev/null && echo "  [watchdog] killed orphan PID $pid ($pat)"
        done
    done
    sleep 1
}

# Wait until GPU memory drops below threshold (MiB). Returns 0 on success,
# 1 if still busy after timeout.  Use before launching memory-hungry steps.
# Fallback: if natural drain timed out and ALLOW_GPU_RESET=1, attempts
# `sudo -n nvidia-smi --gpu-reset` (passwordless) to clear CUDA driver zombies.
wait_gpu_free() {
    local threshold_mib="${1:-3000}"
    local timeout_s="${2:-60}"
    local interval=2
    local waited=0
    while (( waited < timeout_s )); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        used=${used:-99999}
        if (( used < threshold_mib )); then
            echo "  [watchdog] GPU free (${used} MiB) after ${waited}s"
            return 0
        fi
        sleep $interval
        waited=$(( waited + interval ))
    done
    local final
    final=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  [watchdog] GPU still ${final} MiB after ${timeout_s}s — attempt gpu-reset" >&2
    if sudo -n nvidia-smi --gpu-reset >/dev/null 2>&1; then
        sleep 3
        local after
        after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "  [watchdog] post-reset GPU=${after} MiB"
        (( after < threshold_mib )) && return 0
    else
        echo "  [watchdog] WARN: sudo gpu-reset not available (no passwordless sudo?)" >&2
    fi
    return 1
}

latest_step() {
    # Latest checkpoint-N step number, or empty if none.
    ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null \
        | sed 's|.*/checkpoint-||' \
        | sort -n | tail -1
}

drop_optimizer() {
    local step="$1"
    local ckpt="$OUTPUT_DIR/checkpoint-$step"
    if [[ -f "$ckpt/optimizer.pt" ]]; then
        mv "$ckpt/optimizer.pt" "$ckpt/optimizer.pt.bak" 2>/dev/null
        echo "  [watchdog] moved optimizer.pt → .bak (free $(du -h "$ckpt/optimizer.pt.bak" | cut -f1))"
    fi
}

# HF Trainer honors trainer_state.json's save_steps over CLI args on resume.
# Patch it in place so our SAVE_STEPS env knob actually takes effect.
patch_save_steps() {
    local step="$1"
    local ts="$OUTPUT_DIR/checkpoint-$step/trainer_state.json"
    [[ -f "$ts" ]] || return 0
    python3 - "$ts" "$SAVE_STEPS" <<'PY'
import json, sys
path, save_steps = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    d = json.load(f)
old = d.get("save_steps")
d["save_steps"] = save_steps
with open(path, "w") as f:
    json.dump(d, f, indent=2)
print(f"  [watchdog] patched {path}: save_steps {old} → {save_steps}")
PY
}

trainer_state_step() {
    local step="$1"
    local ts="$OUTPUT_DIR/checkpoint-$step/trainer_state.json"
    [[ -f "$ts" ]] || { echo ""; return; }
    python3 -c "import json,sys; print(json.load(open('$ts')).get('global_step',''))" 2>/dev/null
}

# Selective ckpt pruning:
#   - Keep ALL ckpts at multiples of $KEEP_MULTIPLE (default 500)
#   - Keep only last $KEEP_TEMPORARY non-multiple ckpts
prune_checkpoints() {
    local keep_mult="${KEEP_MULTIPLE:-500}"
    local keep_temp="${KEEP_TEMPORARY:-3}"
    python3 - "$OUTPUT_DIR" "$keep_mult" "$keep_temp" <<'PY'
import os, sys, shutil, re
out, keep_mult, keep_temp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if not os.path.isdir(out):
    sys.exit(0)
ckpts = []
for d in os.listdir(out):
    m = re.match(r"^checkpoint-(\d+)$", d)
    if m:
        ckpts.append((int(m.group(1)), os.path.join(out, d)))
ckpts.sort()
permanent = {s for s, _ in ckpts if s > 0 and s % keep_mult == 0}
temporary = [(s, p) for s, p in ckpts if s not in permanent]
to_keep_temp = set(s for s, _ in temporary[-keep_temp:])
removed = []
for s, p in temporary:
    if s in to_keep_temp:
        continue
    shutil.rmtree(p, ignore_errors=True)
    removed.append(s)
if removed:
    print(f"  [watchdog] prune: removed temporary {removed}, kept permanent {sorted(permanent)}, last temp {sorted(to_keep_temp)}")
PY
}

eval_unevaluated_ckpts() {
    # Find ckpts at multiples of EVAL_STEPS_MULTIPLE that aren't yet in EVAL_LOG.
    # Run eval one by one (GPU serial), append CSV.
    (( EVAL_STEPS_MULTIPLE > 0 )) || return 0
    [[ -d "$OUTPUT_DIR" ]] || return 0
    mkdir -p "$(dirname "$EVAL_LOG")"
    [[ -f "$EVAL_LOG" ]] || echo "step,rounds,n_action_steps,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$EVAL_LOG"

    for ckpt_dir in $(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
        local step="${ckpt_dir##*/checkpoint-}"
        (( step % EVAL_STEPS_MULTIPLE == 0 )) || continue
        # Skip if already evaluated (step appears as first CSV column)
        if awk -F, -v s="$step" 'NR>1 && $1==s {found=1} END{exit !found}' "$EVAL_LOG" 2>/dev/null; then
            continue
        fi

        local rounds="$EVAL_ROUNDS_FULL"
        (( step <= EVAL_FAST_THRESHOLD )) && rounds="$EVAL_ROUNDS_FAST"

        echo "  [watchdog] EVAL ckpt-$step  rounds=$rounds  h=$EVAL_ACTION_HORIZON" | tee -a "$WATCHDOG_LOG"

        # Kill any stale procs (training already cleaned up by caller) + wait GPU drain
        pkill -f "gr00t/eval/run_gr00t_server\|policy_inference" 2>/dev/null
        sleep 2
        wait_gpu_free 3000 90 | tee -a "$WATCHDOG_LOG"

        local eval_log="$REPO_ROOT/logs/gr00t_ckpt_eval_${step}.log"
        local tmo=$(( rounds * EVAL_MAX_ROUND_WALL_S + 300 ))

        GR00T_PORT="$EVAL_POLICY_PORT" \
            bash "$LEISAAC_ROOT/scripts/policy_server.sh" start gr00t-n17 "$ckpt_dir" 2>&1 | tail -5 | tee -a "$WATCHDOG_LOG"

        if ! ss -tlnp 2>/dev/null | grep -q ":$EVAL_POLICY_PORT "; then
            echo "  [watchdog] WARN: server not up after launch — extra 15s wait" | tee -a "$WATCHDOG_LOG"
            sleep 15
        fi

        POLICY_PORT="$EVAL_POLICY_PORT" POLICY_TIMEOUT_MS=10000 \
            ACTION_HORIZON="$EVAL_ACTION_HORIZON" \
            EVAL_ROUNDS="$rounds" EPISODE_LENGTH="$EVAL_EPISODE_S" \
            MAX_ROUND_WALL_S="$EVAL_MAX_ROUND_WALL_S" PROMPT="$EVAL_PROMPT" \
            timeout $tmo bash "$REPO_ROOT/server/eval_gr00t.sh" 2>&1 | tee "$eval_log" | \
            grep -E "placed|Final|success rate" | tail -5 | tee -a "$WATCHDOG_LOG"

        # Stop server immediately (frees GPU for next train cycle)
        pkill -f "gr00t/eval/run_gr00t_server\|policy_inference" 2>/dev/null
        sleep 3

        local final_line
        final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
        if [[ -z "$final_line" ]]; then
            echo "$step,$rounds,$EVAL_ACTION_HORIZON,0,0,$((rounds * 3)),NaN,no-Final-line" >> "$EVAL_LOG"
        else
            local success rt placed total avg
            success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
            rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
            placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
            total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
            avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
            echo "$step,$rounds,$EVAL_ACTION_HORIZON,$success,$placed,$total,${avg:-NaN},\"$final_line\"" >> "$EVAL_LOG"
            echo "  [watchdog] ➜ step=$step  rounds=$success/$rt  oranges=$placed/$total" | tee -a "$WATCHDOG_LOG"
        fi
    done
}

retry=0
while (( retry < MAX_RETRIES )); do
    retry=$((retry + 1))
    cur=$(latest_step || echo "")
    actual_step=$(trainer_state_step "$cur" 2>/dev/null)

    echo | tee -a "$WATCHDOG_LOG"
    echo "===== [watchdog] cycle $retry/$MAX_RETRIES  latest_ckpt=${cur:-none}  actual_step=${actual_step:-0}  target=$MAX_STEPS =====" | tee -a "$WATCHDOG_LOG"

    # Step 1: clear any stale procs (eval server + training)
    cleanup_procs
    pkill -f "gr00t/eval/run_gr00t_server\|policy_inference" 2>/dev/null
    sleep 2

    # Step 2: eval any pending ckpts BEFORE training (so we see ckpt-N quality
    # without waiting for next cycle to crash).  Skips ckpts already in CSV.
    eval_unevaluated_ckpts

    # Step 3: cleanup eval procs + WAIT for GPU memory to actually drain
    # (CUDA driver may keep ~7-9 GB resident for ~30s after python proc dies)
    pkill -f "gr00t/eval/run_gr00t_server\|policy_inference" 2>/dev/null
    sleep 3
    wait_gpu_free 3000 90 | tee -a "$WATCHDOG_LOG"

    # Step 4: check if we already hit target after the eval pause
    if [[ -n "$actual_step" ]] && (( actual_step >= MAX_STEPS )); then
        echo "[watchdog] ✅ DONE — latest ckpt-$cur reports global_step=$actual_step >= $MAX_STEPS" | tee -a "$WATCHDOG_LOG"
        break
    fi

    # Step 5: prep ckpt for resume (drop optim, patch save_steps)
    if [[ -n "$cur" ]]; then
        drop_optimizer "$cur" | tee -a "$WATCHDOG_LOG"
        patch_save_steps "$cur" | tee -a "$WATCHDOG_LOG"
    fi

    # Step 6: train (until next crash or MAX_STEPS)
    cycle_log="$WATCHDOG_LOG_DIR/cycle_${retry}_$(date +%H%M%S).log"
    MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
        GLOBAL_BATCH="$GLOBAL_BATCH" GRAD_ACCUM="$GRAD_ACCUM" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        bash "$LEISAAC_ROOT/scripts/finetune/gr00t/train.sh" >"$cycle_log" 2>&1
    rc=$?
    last_step=$(grep -oE "checkpoint-[0-9]+" "$cycle_log" | tail -1 | sed 's/checkpoint-//' || echo "?")
    echo "  [watchdog] cycle $retry exit=$rc  ckpt-after-cycle=$last_step" | tee -a "$WATCHDOG_LOG"

    # Custom pruning: keep all multiples of KEEP_MULTIPLE + last KEEP_TEMPORARY others
    prune_checkpoints | tee -a "$WATCHDOG_LOG"

    if [[ "$rc" == "0" ]]; then
        echo "[watchdog] clean exit, training finished" | tee -a "$WATCHDOG_LOG"
        # One last eval pass to cover the final ckpts
        cleanup_procs
        pkill -f "gr00t/eval/run_gr00t_server\|policy_inference" 2>/dev/null
        sleep 2
        eval_unevaluated_ckpts
        break
    fi
    sleep 2
done

cleanup_procs

if (( retry >= MAX_RETRIES )); then
    echo "[watchdog] ⚠️  reached MAX_RETRIES=$MAX_RETRIES without finishing" | tee -a "$WATCHDOG_LOG"
fi

final=$(latest_step)
echo "[watchdog] final ckpt: checkpoint-$final" | tee -a "$WATCHDOG_LOG"
touch "$OUTPUT_DIR/.training_done"
