#!/usr/bin/env bash
# Auto-restart wrapper for xvla train.sh.
#
# Strategy:
#   - First attempt: RESUME=false, lerobot creates fresh OUTPUT_DIR
#   - Subsequent attempts: RESUME=true, lerobot picks up the latest ckpt under
#     OUTPUT_DIR/checkpoints/ automatically.
#   - Between attempts, prune all but the latest 3 numeric ckpt dirs + last/
#     symlink (each ckpt is ~2.9GB; 20 of them = 58GB, manageable but trim anyway).
#
# Hard stop: MAX_RETRIES (default 20) — 10k step / 500 save = 20 ckpt boundaries,
# enough headroom even if every save coincides with a crash.
#
# Env knobs forwarded verbatim to train.sh. OUTPUT_DIR is read here for prune.

set -uo pipefail   # NOTE: no -e — we explicitly check $?

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEISAAC_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$LEISAAC_ROOT/../logs"
mkdir -p "$LOG_DIR"

WATCHDOG_LOG="$LOG_DIR/xvla_watchdog_$(date +%Y%m%d_%H%M%S).log"
MAX_RETRIES="${MAX_RETRIES:-20}"
KEEP_LAST="${KEEP_LAST:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$LEISAAC_ROOT/outputs/xvla-leisaac-pick-orange}"

prune_ckpts() {
    local ckpt_dir="$OUTPUT_DIR/checkpoints"
    [[ -d "$ckpt_dir" ]] || return 0
    # Numeric ckpt dirs only (000005, 000500, ...); skip 'last' symlink and stray files.
    mapfile -t numeric < <(find "$ckpt_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+$' | sort -V)
    local n=${#numeric[@]}
    if (( n > KEEP_LAST )); then
        local drop=$((n - KEEP_LAST))
        echo "[watchdog] prune: $n ckpts -> keep last $KEEP_LAST (drop $drop)" | tee -a "$WATCHDOG_LOG"
        for ((i = 0; i < drop; i++)); do
            echo "[watchdog]   rm ${numeric[$i]}" | tee -a "$WATCHDOG_LOG"
            rm -rf "${numeric[$i]}"
        done
    fi
}

attempt=0
while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))
    echo "[watchdog] === attempt $attempt / $MAX_RETRIES at $(date +%H:%M:%S) ===" | tee -a "$WATCHDOG_LOG"

    if (( attempt == 1 )); then
        # First run from scratch.
        RESUME="${RESUME:-false}" OUTPUT_DIR="$OUTPUT_DIR" \
            bash "$SCRIPT_DIR/train.sh" "$@" 2>&1 | tee -a "$WATCHDOG_LOG"
    else
        # Force resume from latest ckpt under OUTPUT_DIR/checkpoints/last
        prune_ckpts
        RESUME=true OUTPUT_DIR="$OUTPUT_DIR" \
            bash "$SCRIPT_DIR/train.sh" "$@" 2>&1 | tee -a "$WATCHDOG_LOG"
    fi
    rc=${PIPESTATUS[0]}

    if (( rc == 0 )); then
        echo "[watchdog] ✅ training finished cleanly on attempt $attempt" | tee -a "$WATCHDOG_LOG"
        prune_ckpts
        exit 0
    fi

    echo "[watchdog] ⚠️  attempt $attempt crashed (exit=$rc); auto-resuming in 10s" | tee -a "$WATCHDOG_LOG"
    sleep 10
done

echo "[watchdog] ❌ exhausted $MAX_RETRIES retries" | tee -a "$WATCHDOG_LOG"
exit 1
