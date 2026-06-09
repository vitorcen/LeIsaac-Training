#!/usr/bin/env bash
# Local GUI sweep evaluator. The master_pull_watcher already lands ckpts under a local
# run_dir; this just evals each *fully-pulled* ckpt (size-verified, skips partials) with
# starvla_strict_eval.sh, GUI on so the user can watch, and records a one-line CSV.
# Sequential by nature (one GPU). Loops forever, picking up new ckpts as they land.
#
# Usage: CKDIR=<run/checkpoints> FULLSIZE=<bytes> [ROUNDS=3] [VLM_QUANT=0] starvla_local_sweep.sh
set -uo pipefail
ROOT=/home/david/work/isaaclab-experience
CKDIR="${CKDIR:?need CKDIR=<run/checkpoints>}"
FULLSIZE="${FULLSIZE:?need FULLSIZE=<complete ckpt bytes>}"
ROUNDS="${ROUNDS:-3}"
VLM_QUANT="${VLM_QUANT:-0}"
GUI="${GUI:-1}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLL_S="${POLL_S:-60}"
CSV="$CKDIR/local_sweep.csv"
LOG=/tmp/starvla_local_sweep.log
[ -f "$CSV" ] || echo "step,rounds,placed_per_ep,E_oranges_pct,timestamp" > "$CSV"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }

while true; do
  # newest first so the user sees the most-trained ckpt's behavior soonest
  for ck in $(ls -1 "$CKDIR"/steps_*_pytorch_model.pt 2>/dev/null | sort -t_ -k2 -n -r); do
    step=$(echo "$ck" | sed -n 's/.*steps_\([0-9]*\)_.*/\1/p')
    sz=$(stat -c %s "$ck" 2>/dev/null || echo 0)
    [ "$sz" = "$FULLSIZE" ] || { log "skip step $step (partial $((sz/1000000))MB)"; continue; }
    grep -q "^$step," "$CSV" && continue
    log "=== eval step $step (GUI=$GUI, ${ROUNDS}-round) ==="
    GUI=$GUI VLM_QUANT=$VLM_QUANT ROUNDS=$ROUNDS EPISODE_LENGTH_S=$EPISODE_LENGTH_S \
      MAX_ROUND_WALL_S=$MAX_ROUND_WALL_S DISPLAY="${DISPLAY:-:0}" \
      bash "$ROOT/LeIsaac/scripts/evaluation/starvla_strict_eval.sh" "$ck" >>"$LOG" 2>&1 || true
    # parse from the distribution .md (aggregate_distribution.py output) — the raw eval log
    # (/tmp/sv_strict_eval.log) only has per-episode lines, NOT the placed_per_ep summary.
    md="$CKDIR/strict_eval_q${VLM_QUANT}_r${ROUNDS}demo.distribution.md"
    raw=$(grep -oE 'placed_per_ep = \[[^]]*\]' "$md" 2>/dev/null | tail -1)
    pct=$(grep -oE 'E\(oranges/ep\) = [0-9.]+ / [0-9]+ = [0-9.]+%' "$md" 2>/dev/null | grep -oE '[0-9.]+%' | tail -1)
    # fallback to the raw eval log's Final line if the .md wasn't produced (eval crashed)
    [ -z "$raw" ] && raw=$(grep -oE 'oranges: [0-9]+/[0-9]+' /tmp/sv_strict_eval.log 2>/dev/null | tail -1)
    echo "$step,$ROUNDS,\"${raw:-NA}\",${pct:-NA},$(date +%FT%T)" >> "$CSV"
    log "step $step -> ${raw:-NA} ${pct:-NA}"
  done
  sleep "$POLL_S"
done
