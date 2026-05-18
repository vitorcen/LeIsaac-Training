#!/usr/bin/env bash
# Auto-restart wrapper for openvla train.sh.
#
# Why: bnb 0.43 + accelerate hooks + 4-bit Linear4bit triggers flaky mid-training
# AttributeError crashes (we've hit two distinct ones).  With save_steps=500 and
# the new auto-resume logic in train.py, a crash loses ≤500*step_time work, then
# the next iteration of this loop picks up exactly where we left off.
#
# Hard stop: max 20 retries (10k step / 500 save = 20 ckpt boundaries — that's
# enough headroom even if every save has a crash).
#
# Env knobs forwarded verbatim to train.sh.
set -uo pipefail   # NOTE: no -e — we explicitly check $?

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEISAAC_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$LEISAAC_ROOT/../logs"
mkdir -p "$LOG_DIR"

WATCHDOG_LOG="$LOG_DIR/openvla_watchdog_$(date +%Y%m%d_%H%M%S).log"
MAX_RETRIES="${MAX_RETRIES:-20}"

attempt=0
while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))
    echo "[watchdog] === attempt $attempt / $MAX_RETRIES at $(date +%H:%M:%S) ===" | tee -a "$WATCHDOG_LOG"

    bash "$SCRIPT_DIR/train.sh" "$@" 2>&1 | tee -a "$WATCHDOG_LOG"
    rc=${PIPESTATUS[0]}

    if (( rc == 0 )); then
        echo "[watchdog] ✅ training finished cleanly on attempt $attempt" | tee -a "$WATCHDOG_LOG"
        exit 0
    fi

    echo "[watchdog] ⚠️  attempt $attempt crashed (exit=$rc); auto-resuming in 10s" | tee -a "$WATCHDOG_LOG"
    sleep 10
done

echo "[watchdog] ❌ exhausted $MAX_RETRIES retries" | tee -a "$WATCHDOG_LOG"
exit 1
