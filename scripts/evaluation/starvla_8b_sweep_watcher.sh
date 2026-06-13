#!/usr/bin/env bash
# Auto pull + quick-eval sweep watcher for StarVLA Qwen3-VL-8B cloud checkpoints.
# 8B variant of starvla_sweep_watcher.sh: points at the westd box (the 48G-4090
# training run), pulls each steps_<N>.pt (~18GB), and serves it WITH VLM int8
# (STARVLA_VLM_8BIT=1) — 8B bf16 (~16G) will not co-locate with Isaac on a 24G
# card, int8 (~10G) does. Base weights come from the LOCAL 8B snapshot (user's
# hf download), reused across every ckpt so we only pull the 18GB ckpt each time.
#
#   <run_dir>/{config.yaml, dataset_statistics.json} + checkpoints/<file>.pt
# laid out locally; serve_starvla loads it. Password from `pass autodl/westd`.
#
# Env overrides: POLL_S EVAL_ROUNDS EPISODE_LENGTH_S STEP_HZ ACTION_HORIZON PORT IMG_SIZE GUI
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP=$ROOT/LeIsaac/outputs/starvla-8b-run    # separate from the 4B starvla-sweep
RUNDIR=$SWEEP/run
CKDIR=$RUNDIR/checkpoints
CSV=$SWEEP/sweep.csv
LOG=$SWEEP/watcher.log
# local 8B snapshot dir (abspath-safe; serve does os.path.abspath on --base)
BASE="${BASE:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
STARVLA_PY=$(conda info --base)/envs/starvla_eval/bin/python
SERVE=$ROOT/LeIsaac/scripts/evaluation/serve_starvla.py
PROMPT="Grab orange and place into plate"

PORT="${PORT:-8014}"                       # distinct from 4B watcher (8002) / strict (8013)
POLL_S="${POLL_S:-300}"                     # 6k ckpts arrive ~every 50min; poll every 5min
EVAL_ROUNDS="${EVAL_ROUNDS:-5}"            # quick-screen = 5-round (user标准 2026-06-12; 3-round variance太大)
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-120}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-180}"
STEP_HZ="${STEP_HZ:-30}"
STUCK_WINDOW_S="${STUCK_WINDOW_S:-30}"
STUCK_EPS_RAD="${STUCK_EPS_RAD:-0.05}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
IMG_SIZE="${IMG_SIZE:-448}"
VLM_QUANT="${VLM_QUANT:-8}"                 # 8bit so 8B fits 24G alongside Isaac
GUI="${GUI:-1}"                            # 1 = visible window (DISPLAY=:0), 0 = headless

CLOUD_HOST=connect.westd.seetacloud.com
CLOUD_PORT=15528
CLOUD_RUN=/root/autodl-tmp/starvla-outputs/so101_pickorange_qwen3vl8b_gr00t
CLOUD_CK=$CLOUD_RUN/checkpoints
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

mkdir -p "$CKDIR"
[ -f "$CSV" ] || echo "ckpt,global_step,success_rate,successes,rounds,oranges_placed,oranges_total,timestamp" > "$CSV"

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
already_done() { grep -q "^$1," "$CSV"; }

pull_meta() {
  local pw="$1"
  [ -f "$RUNDIR/dataset_statistics.json" ] && [ -f "$RUNDIR/config.yaml" ] && return 0
  log "pulling run metadata (config.yaml + dataset_statistics.json)"
  sshpass -p "$pw" rsync -az --timeout=300 -e "ssh -p $CLOUD_PORT $SSHOPT" \
    root@"$CLOUD_HOST":"$CLOUD_RUN/config.yaml" root@"$CLOUD_HOST":"$CLOUD_RUN/dataset_statistics.json" \
    "$RUNDIR/" >> "$LOG" 2>&1
}

# eval_one <ckpt_file>
eval_one() {
  local fn="$1" pt="$CKDIR/$1"
  [ -f "$pt" ] || { log "  $fn: no local .pt, skip"; return 1; }
  local gs; gs=$(echo "$fn" | sed -n 's/steps_\([0-9]*\)_.*/\1/p')

  local QENV=()
  [ "$VLM_QUANT" = 8 ] && QENV=(STARVLA_VLM_8BIT=1)
  [ "$VLM_QUANT" = 4 ] && QENV=(STARVLA_VLM_4BIT=1)

  # serve with retry: 8B torch-load of the 18GB ckpt intermittently SEGFAULTs
  # (C-stack, ~40% per memory wallx-env-py310-torch-segfault); just relaunch.
  local spid="" up=0 attempt
  for attempt in 1 2 3; do
    log "  $fn: serving (gs=$gs, q=$VLM_QUANT) attempt $attempt..."
    rm -f /tmp/sv8b_serve.log
    nohup env CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${QENV[@]}" \
      "$STARVLA_PY" -u "$SERVE" --ckpt "$pt" --base "$BASE" --port "$PORT" --img_size "$IMG_SIZE" --prompt "$PROMPT" \
      > /tmp/sv8b_serve.log 2>&1 &
    spid=$!
    up=0
    for _ in $(seq 1 200); do               # 8bit CPU-load of 18GB ckpt is slow -> up to 400s
      grep -q "SERVE_READY" /tmp/sv8b_serve.log 2>/dev/null && { up=1; break; }
      ss -tln 2>/dev/null | grep -q ":$PORT " && { up=1; break; }
      kill -0 "$spid" 2>/dev/null || break   # died (segfault) -> break to retry
      sleep 2
    done
    [ "$up" = 1 ] && break
    log "  $fn: serve attempt $attempt failed (segfault?), retrying"; kill -9 "$spid" 2>/dev/null; sleep 3
  done
  if [ "$up" != 1 ]; then
    log "  $fn: serve FAILED after 3 attempts (see /tmp/sv8b_serve.log)"; return 1
  fi

  local RENDER=(--headless); [ "$GUI" = 1 ] && RENDER=()
  log "  $fn: eval ${EVAL_ROUNDS}-round (GUI=$GUI)..."
  rm -f /tmp/sv8b_eval.log
  ( cd "$ROOT/LeIsaac" && DISPLAY="${DISPLAY:-:0}" conda run -n isaaclab --no-capture-output \
    python -u scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --eval_rounds="$EVAL_ROUNDS" --episode_length_s="$EPISODE_LENGTH_S" --step_hz="$STEP_HZ" \
      --max_round_wall_s="$MAX_ROUND_WALL_S" --stuck_window_s="$STUCK_WINDOW_S" --stuck_eps_rad="$STUCK_EPS_RAD" \
      --policy_type=starvla --policy_host=localhost --policy_port="$PORT" --policy_timeout_ms=60000 \
      --policy_action_horizon="$ACTION_HORIZON" \
      --policy_language_instruction="$PROMPT" \
      --device=cuda "${RENDER[@]}" --enable_cameras ) > /tmp/sv8b_eval.log 2>&1

  kill -9 "$spid" 2>/dev/null; sleep 3

  local fl; fl=$(grep "Final success rate" /tmp/sv8b_eval.log | tail -1)
  if [ -z "$fl" ]; then log "  $fn: no Final line (crash? /tmp/sv8b_eval.log)"; return 1; fi
  local sr succ rnd pl tot
  sr=$(echo "$fl"   | sed -n 's/.*success rate:[[:space:]]*\([0-9.]*\).*/\1/p')
  succ=$(echo "$fl" | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\1/p')
  rnd=$(echo "$fl"  | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\2/p')
  pl=$(echo "$fl"   | sed -n 's/.*oranges:[[:space:]]*\([0-9]*\)\/\([0-9]*\).*/\1/p')
  tot=$(echo "$fl"  | sed -n 's/.*oranges:[[:space:]]*\([0-9]*\)\/\([0-9]*\).*/\2/p')
  echo "$fn,$gs,$sr,$succ,$rnd,$pl,$tot,$(date +%FT%T)" >> "$CSV"
  log "  $fn: gs=$gs success=$sr [$succ/$rnd] oranges=$pl/$tot"
  # local prune DISABLED — never auto-delete a ckpt that might be the winner (see 4B watcher note).
  return 0
}

log "=== StarVLA 8B sweep watcher (westd, q=${VLM_QUANT}, poll=${POLL_S}s, eval=${EVAL_ROUNDS}-round, port=$PORT) ==="
while true; do
  PW="$(pass autodl/westd 2>/dev/null)"
  pull_meta "$PW"
  # self-heal: re-eval any LOCAL ckpt with no CSV row. A serve segfault (e.g. GPU
  # contention) leaves a pulled ckpt stranded — cloud-delete already removed it from
  # the pull queue, so without this it would never be retried.
  for lf in "$CKDIR"/steps_*_pytorch_model.pt; do
    [ -e "$lf" ] || continue
    bn=$(basename "$lf")
    already_done "$bn" && continue
    log "self-heal: local $bn has no result -> eval"
    eval_one "$bn" || log "  $bn: self-heal eval failed, retry next loop"
  done
  names=$(sshpass -p "$PW" ssh $SSHOPT -p "$CLOUD_PORT" root@"$CLOUD_HOST" \
            "ls $CLOUD_CK 2>/dev/null | grep -E 'steps_[0-9]+_pytorch_model.pt'" 2>/dev/null)
  if [ -n "$names" ]; then
    for fn in $names; do
      already_done "$fn" && continue
      log "new ckpt: $fn -> pulling (~18GB)"
      if sshpass -p "$PW" rsync -az --partial --timeout=1800 -e "ssh -p $CLOUD_PORT $SSHOPT" \
           root@"$CLOUD_HOST":"$CLOUD_CK/$fn" "$CKDIR/" >> "$LOG" 2>&1; then
        # verify a plausible full pull (>15GB), then free cloud disk — 100G box can't
        # hold many 17.9G ckpts and an ENOSPC on torch.save crashes training (memory:
        # feedback-training-save-policy). Local keeps every ckpt; cloud needs none.
        local_sz=$(stat -c%s "$CKDIR/$fn" 2>/dev/null || echo 0)
        if [ "$local_sz" -gt 16000000000 ]; then
          sshpass -p "$PW" ssh $SSHOPT -p "$CLOUD_PORT" root@"$CLOUD_HOST" "rm -f $CLOUD_CK/$fn" 2>/dev/null \
            && log "  $fn: pulled ($((local_sz/1000000000))GB) + removed from cloud"
        fi
        eval_one "$fn" || log "  $fn: eval failed, retry next loop"
      else
        log "  $fn: rsync failed, retry next loop"
      fi
    done
  fi
  sleep "$POLL_S"
done
