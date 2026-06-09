#!/usr/bin/env bash
# Qwen3.5 sweep evaluator. Unlike starvla_local_sweep.sh (evals full ckpts directly), the
# Qwen3.5 frozen-VLM runs are pulled as HEADS only (~825MB) — so each sweep point is:
#   pulled head -> merge_head.py {vlm_base_qwen35_2b + head} -> full ckpt -> serve (tf5.2 env)
#   -> strict_eval (GUI) -> record CSV -> rm merged full ckpt (save disk; head is kept).
# Polls the local heads dir (master_pull_watcher lands them), evals step>=MIN_STEP every cycle,
# newest-first. Runs forever, picking up new heads as they're pulled.
#
# Env: MIN_STEP(5000) ROUNDS(3) GUI(1) VLM_QUANT(0) POLL_S(60)
set -uo pipefail
ROOT=/home/david/work/isaaclab-experience
LRD=$ROOT/LeIsaac/outputs/starvla-qwen35-2b-run
HEADS=$LRD/heads
CKDIR=$LRD/checkpoints
VLMBASE=$ROOT/LeIsaac/outputs/_head_sweep_tools/vlm_base_qwen35_2b.pt
MERGE=$ROOT/LeIsaac/outputs/_head_sweep_tools/merge_head.py
QENV=/home/david/miniconda3/envs/starvla_eval_qwen35/bin/python
BASE35=$(ls -d /home/david/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/ | head -1); BASE35=${BASE35%/}
MIN_STEP="${MIN_STEP:-5000}"
ROUNDS="${ROUNDS:-3}"; GUI="${GUI:-1}"; VLM_QUANT="${VLM_QUANT:-0}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-60}"; MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLL_S="${POLL_S:-60}"
CSV=$HEADS/qwen35_sweep.csv
LOG=/tmp/qwen35_sweep.log
mkdir -p "$CKDIR"
[ -f "$CSV" ] || echo "step,rounds,placed_per_ep,E_oranges_pct,timestamp" > "$CSV"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }

while true; do
  for hf in $(ls -1 "$HEADS"/steps_*_head.pt 2>/dev/null | sort -t_ -k2 -n -r); do
    step=$(echo "$hf" | sed -n 's/.*steps_\([0-9]*\)_.*/\1/p')
    [ "${step:-0}" -lt "$MIN_STEP" ] && continue
    grep -q "^$step," "$CSV" && continue
    # head must be fully pulled (>700MB)
    sz=$(stat -c %s "$hf" 2>/dev/null || echo 0); [ "$sz" -gt 700000000 ] || { log "skip $step (head partial $((sz/1000000))MB)"; continue; }
    full=$CKDIR/steps_${step}_pytorch_model.pt
    log "=== merge+eval step $step ==="
    $QENV "$MERGE" "$VLMBASE" "$hf" "$full" >>"$LOG" 2>&1 || { log "merge fail $step"; continue; }
    STARVLA_PY=$QENV BASE=$BASE35 GUI=$GUI VLM_QUANT=$VLM_QUANT ROUNDS=$ROUNDS \
      EPISODE_LENGTH_S=$EPISODE_LENGTH_S MAX_ROUND_WALL_S=$MAX_ROUND_WALL_S DISPLAY="${DISPLAY:-:0}" \
      bash "$ROOT/LeIsaac/scripts/evaluation/starvla_strict_eval.sh" "$full" >>"$LOG" 2>&1 || true
    md=$CKDIR/strict_eval_q${VLM_QUANT}_r${ROUNDS}demo.distribution.md
    raw=$(grep -oE 'placed_per_ep = \[[^]]*\]' "$md" 2>/dev/null | tail -1)
    pct=$(grep -oE 'E\(oranges/ep\) = [0-9.]+ / [0-9]+ = [0-9.]+%' "$md" 2>/dev/null | grep -oE '[0-9.]+%' | tail -1)
    echo "$step,$ROUNDS,\"${raw:-NA}\",${pct:-NA},$(date +%FT%T)" >> "$CSV"
    log "step $step -> ${raw:-NA} ${pct:-NA}"
    rm -f "$full"   # drop merged full ckpt; head is kept, can re-merge anytime
  done
  sleep "$POLL_S"
done
