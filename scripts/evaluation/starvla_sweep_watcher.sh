#!/usr/bin/env bash
# Auto pull + quick-eval sweep watcher for StarVLA (Qwen3-VL-4B) cloud checkpoints.
#
# Polls the westc AutoDL box for new steps_<N>_pytorch_model.pt under the run's
# checkpoints/, rsyncs each (+ the run's config.yaml + dataset_statistics.json
# once) into a local sweep dir laid out as a loadable run_dir, serves it with
# serve_starvla.py, runs a headless quick eval via policy_inference.py
# (--policy_type=starvla), parses the Final line, appends a CSV leaderboard row.
#
# StarVLA ckpts are single .pt FILES (not dirs like wall-x). from_pretrained needs
#   <run_dir>/{config.yaml, dataset_statistics.json} + checkpoints/<file>.pt
# so we mirror that layout locally and point serve_starvla at the .pt.
#
# Env overrides: POLL_S EVAL_ROUNDS EPISODE_LENGTH_S STEP_HZ ACTION_HORIZON PORT IMG_SIZE
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP=$ROOT/LeIsaac/outputs/starvla-sweep
RUNDIR=$SWEEP/run                         # local mirror of the cloud run_dir
CKDIR=$RUNDIR/checkpoints
CSV=$SWEEP/sweep.csv
LOG=$SWEEP/watcher.log
BASE=Qwen/Qwen3-VL-4B-Instruct  # HF cache repo id (model moved to ~/.cache/huggingface/hub)
STARVLA_PY=$(conda info --base)/envs/starvla_eval/bin/python
SERVE=$ROOT/LeIsaac/scripts/evaluation/serve_starvla.py
PROMPT="Grab orange and place into plate"

PORT="${PORT:-8002}"
POLL_S="${POLL_S:-180}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
# MATCH scripts/benchmark/run_one.sh authoritative params (else numbers are not
# leaderboard-comparable): ep_len=120 sim-s + wall_cap=180s, BOTH timeouts active.
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-120}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-180}"
STEP_HZ="${STEP_HZ:-30}"                   # dataset fps = 30
STUCK_WINDOW_S="${STUCK_WINDOW_S:-30}"     # run_one.sh: non-ACT/DP VLA uses 30s/0.05rad
STUCK_EPS_RAD="${STUCK_EPS_RAD:-0.05}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"     # StarVLA action_horizon
IMG_SIZE="${IMG_SIZE:-448}"

CLOUD_HOST=connect.westc.seetacloud.com
CLOUD_PORT=31709
CLOUD_RUN=/root/autodl-tmp/starvla-outputs/so101_pickorange_qwengr00t
CLOUD_CK=$CLOUD_RUN/checkpoints
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

mkdir -p "$CKDIR"
[ -f "$CSV" ] || echo "ckpt,global_step,success_rate,successes,rounds,oranges_placed,oranges_total,timestamp" > "$CSV"

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
already_done() { grep -q "^$1," "$CSV"; }

# pull run_dir metadata (config.yaml + dataset_statistics.json) once
pull_meta() {
  local pw="$1"
  [ -f "$RUNDIR/dataset_statistics.json" ] && [ -f "$RUNDIR/config.yaml" ] && return 0
  log "pulling run metadata (config.yaml + dataset_statistics.json)"
  sshpass -p "$pw" rsync -az --timeout=300 -e "ssh -p $CLOUD_PORT $SSHOPT" \
    root@"$CLOUD_HOST":"$CLOUD_RUN/config.yaml" root@"$CLOUD_HOST":"$CLOUD_RUN/dataset_statistics.json" \
    "$RUNDIR/" >> "$LOG" 2>&1
}

# eval_one <ckpt_file>  e.g. steps_500_pytorch_model.pt
eval_one() {
  local fn="$1" pt="$CKDIR/$1"
  [ -f "$pt" ] || { log "  $fn: no local .pt, skip"; return 1; }
  local gs; gs=$(echo "$fn" | sed -n 's/steps_\([0-9]*\)_.*/\1/p')

  log "  $fn: serving (gs=$gs)..."
  rm -f /tmp/sv_serve.log
  nohup env CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$STARVLA_PY" -u "$SERVE" --ckpt "$pt" --base "$BASE" --port "$PORT" --img_size "$IMG_SIZE" --prompt "$PROMPT" \
    > /tmp/sv_serve.log 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 1 150); do
    # readiness via the print signal OR the listening port (overwatch silences the module logger)
    grep -q "SERVE_READY" /tmp/sv_serve.log 2>/dev/null && { up=1; break; }
    ss -tln 2>/dev/null | grep -q ":$PORT " && { up=1; break; }
    kill -0 "$spid" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    log "  $fn: serve FAILED (see /tmp/sv_serve.log)"; kill -9 "$spid" 2>/dev/null; return 1
  fi

  log "  $fn: eval ${EVAL_ROUNDS}-round GUI (DISPLAY=:0)..."
  rm -f /tmp/sv_eval.log
  ( cd "$ROOT/LeIsaac" && DISPLAY="${DISPLAY:-:0}" conda run -n isaaclab --no-capture-output \
    python -u scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --eval_rounds="$EVAL_ROUNDS" --episode_length_s="$EPISODE_LENGTH_S" --step_hz="$STEP_HZ" \
      --max_round_wall_s="$MAX_ROUND_WALL_S" --stuck_window_s="$STUCK_WINDOW_S" --stuck_eps_rad="$STUCK_EPS_RAD" \
      --policy_type=starvla --policy_host=localhost --policy_port="$PORT" --policy_timeout_ms=60000 \
      --policy_action_horizon="$ACTION_HORIZON" \
      --policy_language_instruction="$PROMPT" \
      --device=cuda --enable_cameras ) > /tmp/sv_eval.log 2>&1

  kill -9 "$spid" 2>/dev/null; sleep 3

  local fl; fl=$(grep "Final success rate" /tmp/sv_eval.log | tail -1)
  if [ -z "$fl" ]; then log "  $fn: no Final line (crash? /tmp/sv_eval.log)"; return 1; fi
  local sr succ rnd pl tot
  sr=$(echo "$fl"   | sed -n 's/.*success rate:[[:space:]]*\([0-9.]*\).*/\1/p')
  succ=$(echo "$fl" | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\1/p')
  rnd=$(echo "$fl"  | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\2/p')
  pl=$(echo "$fl"   | sed -n 's/.*oranges:[[:space:]]*\([0-9]*\)\/\([0-9]*\).*/\1/p')
  tot=$(echo "$fl"  | sed -n 's/.*oranges:[[:space:]]*\([0-9]*\)\/\([0-9]*\).*/\2/p')
  echo "$fn,$gs,$sr,$succ,$rnd,$pl,$tot,$(date +%FT%T)" >> "$CSV"
  log "  $fn: gs=$gs success=$sr [$succ/$rnd] oranges=$pl/$tot"

  # NOTE: local prune DISABLED. Local disk has ~500GB free; the whole sweep
  # (~14 ckpts x 10GB = 140GB) fits. The old "keep newest 3" prune destroyed the
  # peak winner (steps_15000) because cloud keep_last=2 had already removed it too
  # -> lost from BOTH sides. Never auto-delete a ckpt that might be the winner.
  # Manual cleanup only, AFTER strict eval picks the winner.
  return 0
}

log "=== StarVLA sweep watcher started (poll=${POLL_S}s, eval=${EVAL_ROUNDS}-round, port=$PORT) ==="
while true; do
  PW="$(pass autodl/westd 2>/dev/null)"
  pull_meta "$PW"
  names=$(sshpass -p "$PW" ssh $SSHOPT -p "$CLOUD_PORT" root@"$CLOUD_HOST" \
            "ls $CLOUD_CK 2>/dev/null | grep -E 'steps_[0-9]+_pytorch_model.pt'" 2>/dev/null)
  if [ -n "$names" ]; then
    for fn in $names; do
      already_done "$fn" && continue
      log "new ckpt: $fn -> pulling (~8-10GB)"
      if sshpass -p "$PW" rsync -az --timeout=1800 -e "ssh -p $CLOUD_PORT $SSHOPT" \
           root@"$CLOUD_HOST":"$CLOUD_CK/$fn" "$CKDIR/" >> "$LOG" 2>&1; then
        eval_one "$fn" || log "  $fn: eval failed, retry next loop"
      else
        log "  $fn: rsync failed, retry next loop"
      fi
    done
  fi
  sleep "$POLL_S"
done
