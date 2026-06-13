#!/usr/bin/env bash
# Auto pull + quick-eval sweep for the StarVLA Qwen3-VL-8B + GR00T_v2 (QwenGR00T_N17
# head) run on the weste box. Unlike starvla_8b_sweep_watcher.sh (pulls the whole
# ~18GB ckpt), this pulls only the ~0.6G HEAD (the box's n17_extract_heads.sh emits
# one per ckpt, and keep-last-2 prunes the fulls before we could grab them), then
# reconstructs the full ckpt LOCALLY by merging the frozen Qwen3-VL-8B base
# (outputs/_head_sweep_tools/vlm_base_8b.pt) with the head — head ∪ base = full,
# byte-exact (merge_ckpt.py). Serves VLM int8 (8B bf16 won't co-locate with Isaac
# on a 24G card; int8 ~11G does). The reconstructed full is deleted after each eval
# (18.8G each; we keep only the tiny heads). Password from `pass autodl/westd`
# (the west boxes share one password). Headless by default (unattended).
#
# Env overrides: MIN_STEP POLL_S EVAL_ROUNDS EPISODE_LENGTH_S STEP_HZ ACTION_HORIZON
#                PORT IMG_SIZE GUI VLM_QUANT
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP=$ROOT/LeIsaac/outputs/starvla-n17-run
RUNDIR=$SWEEP/run
CKDIR=$RUNDIR/checkpoints      # reconstructed fulls live here (transient, deleted post-eval)
HEADDIR=$SWEEP/heads           # pulled heads kept here (the permanent archive)
CSV=$SWEEP/sweep.csv
LOG=$SWEEP/watcher.log
VLM_BASE_PT="$ROOT/LeIsaac/outputs/_head_sweep_tools/vlm_base_8b.pt"   # frozen Qwen3-VL-8B (qwen_vl_interface.*)
MERGE="$ROOT/LeIsaac/scripts/ckpt/merge_ckpt.py"
BASE="${BASE:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
STARVLA_PY=$(conda info --base)/envs/starvla_eval/bin/python
SERVE=$ROOT/LeIsaac/scripts/evaluation/serve_starvla.py
PROMPT="Grab orange and place into plate"

MIN_STEP="${MIN_STEP:-10000}"              # skip undertrained ckpts; peak window starts ~1.1ep
PORT="${PORT:-8015}"                       # distinct from 4B(8002)/8b(8014)/strict(8013)
POLL_S="${POLL_S:-300}"                    # heads arrive ~every 25min; poll every 5min
EVAL_ROUNDS="${EVAL_ROUNDS:-5}"            # quick-screen = 5-round (3-round variance too big; user标准 2026-06-12)
# MATCH run_one.sh authoritative params (else not leaderboard-comparable):
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-120}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-180}"
STEP_HZ="${STEP_HZ:-30}"
STUCK_WINDOW_S="${STUCK_WINDOW_S:-30}"
STUCK_EPS_RAD="${STUCK_EPS_RAD:-0.05}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
IMG_SIZE="${IMG_SIZE:-448}"
VLM_QUANT="${VLM_QUANT:-8}"
GUI="${GUI:-1}"                            # 1 = visible window (headless eval hangs on --enable_cameras render init); 0 = headless

CLOUD_HOST=connect.weste.seetacloud.com
CLOUD_PORT=16763
CLOUD_RUN=/root/autodl-tmp/starvla-outputs/so101_pickorange_qwen3vl8b_gr00t_v2
CLOUD_HEADS=$CLOUD_RUN/heads
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

mkdir -p "$CKDIR" "$HEADDIR"
[ -f "$CSV" ] || echo "ckpt,global_step,success_rate,successes,rounds,oranges_placed,oranges_total,timestamp" > "$CSV"
[ -f "$VLM_BASE_PT" ] || { echo "FATAL: vlm_base_8b.pt not found at $VLM_BASE_PT"; exit 1; }

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

# eval_one <head_filename>  (e.g. steps_10000_pytorch_model_head.pt)
eval_one() {
  local hf="$1" hp="$HEADDIR/$1"
  [ -f "$hp" ] || { log "  $hf: no local head, skip"; return 1; }
  local gs; gs=$(echo "$hf" | sed -n 's/steps_\([0-9]*\)_.*/\1/p')
  local full="$CKDIR/steps_${gs}_pytorch_model.pt"

  # reconstruct full = vlm_base_8b (qwen_vl_interface.*) U head (action_model.*)
  log "  $hf: reconstruct full ckpt (vlm_base_8b + head)"
  if ! "$STARVLA_PY" "$MERGE" "$VLM_BASE_PT" "$hp" "$full" >> "$LOG" 2>&1; then
    log "  $hf: merge FAILED"; rm -f "$full"; return 1
  fi

  local QENV=()
  [ "$VLM_QUANT" = 8 ] && QENV=(STARVLA_VLM_8BIT=1)
  [ "$VLM_QUANT" = 4 ] && QENV=(STARVLA_VLM_4BIT=1)

  # serve with retry: 8B torch-load of the 18GB ckpt intermittently segfaults (~40%).
  local spid="" up=0 attempt
  for attempt in 1 2 3; do
    log "  $hf: serving (gs=$gs, q=$VLM_QUANT) attempt $attempt..."
    rm -f /tmp/svn17_serve.log
    nohup env CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${QENV[@]}" \
      "$STARVLA_PY" -u "$SERVE" --ckpt "$full" --base "$BASE" --port "$PORT" --img_size "$IMG_SIZE" --prompt "$PROMPT" \
      > /tmp/svn17_serve.log 2>&1 &
    spid=$!
    up=0
    for _ in $(seq 1 200); do
      grep -q "SERVE_READY" /tmp/svn17_serve.log 2>/dev/null && { up=1; break; }
      ss -tln 2>/dev/null | grep -q ":$PORT " && { up=1; break; }
      kill -0 "$spid" 2>/dev/null || break
      sleep 2
    done
    [ "$up" = 1 ] && break
    if grep -q "size mismatch" /tmp/svn17_serve.log 2>/dev/null; then
      log "  $hf: SIZE MISMATCH (base/ckpt) — aborting this ckpt"; kill -9 "$spid" 2>/dev/null; break
    fi
    log "  $hf: serve attempt $attempt failed (segfault?), retrying"; kill -9 "$spid" 2>/dev/null; sleep 3
  done
  if [ "$up" != 1 ]; then
    log "  $hf: serve FAILED (see /tmp/svn17_serve.log)"; rm -f "$full"; return 1
  fi

  local RENDER=(--headless); [ "$GUI" = 1 ] && RENDER=()
  # Outer wall timeout: --max_round_wall_s only caps a round once it STARTS; a hang in
  # env-setup / first-handshake (e.g. headless --enable_cameras render init) never starts
  # a round, so without this the eval blocks the whole sweep forever (seen 2026-06-12).
  local EVAL_TIMEOUT=$((300 + EVAL_ROUNDS * (EPISODE_LENGTH_S + 60)))
  log "  $hf: eval ${EVAL_ROUNDS}-round (GUI=$GUI, wall-timeout ${EVAL_TIMEOUT}s)..."
  rm -f /tmp/svn17_eval.log
  ( cd "$ROOT/LeIsaac" && DISPLAY="${DISPLAY:-:0}" timeout --kill-after=30 "$EVAL_TIMEOUT" \
    conda run -n isaaclab --no-capture-output \
    python -u scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --eval_rounds="$EVAL_ROUNDS" --episode_length_s="$EPISODE_LENGTH_S" --step_hz="$STEP_HZ" \
      --max_round_wall_s="$MAX_ROUND_WALL_S" --stuck_window_s="$STUCK_WINDOW_S" --stuck_eps_rad="$STUCK_EPS_RAD" \
      --policy_type=starvla --policy_host=localhost --policy_port="$PORT" --policy_timeout_ms=60000 \
      --policy_action_horizon="$ACTION_HORIZON" \
      --policy_language_instruction="$PROMPT" \
      --device=cuda "${RENDER[@]}" --enable_cameras ) > /tmp/svn17_eval.log 2>&1

  kill -9 "$spid" 2>/dev/null
  pkill -9 -f "[p]olicy_inference.py" 2>/dev/null   # reap orphaned Isaac if timeout fired
  sleep 3
  rm -f "$full"                            # reconstructed full is transient; head is the archive

  local fl; fl=$(grep "Final success rate" /tmp/svn17_eval.log | tail -1)
  if [ -z "$fl" ]; then log "  $hf: no Final line (crash? /tmp/svn17_eval.log)"; return 1; fi
  local sr succ rnd pl tot
  sr=$(echo "$fl"   | sed -n 's/.*success rate:[[:space:]]*\([0-9.]*\).*/\1/p')
  succ=$(echo "$fl" | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\1/p')
  rnd=$(echo "$fl"  | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\2/p')
  pl=$(echo "$fl"   | sed -n 's/.*oranges:[[:space:]]*\([0-9]*\)\/\([0-9]*\).*/\1/p')
  tot=$(echo "$fl"  | sed -n 's/.*oranges:[[:space:]]*\([0-9]*\)\/\([0-9]*\).*/\2/p')
  echo "$hf,$gs,$sr,$succ,$rnd,$pl,$tot,$(date +%FT%T)" >> "$CSV"
  log "  $hf: gs=$gs success=$sr [$succ/$rnd] oranges=$pl/$tot"
  return 0
}

log "=== StarVLA N17 sweep watcher (weste, MIN_STEP=$MIN_STEP, q=$VLM_QUANT, poll=${POLL_S}s, eval=${EVAL_ROUNDS}-round, port=$PORT, GUI=$GUI) ==="
while true; do
  PW="$(pass autodl/westd 2>/dev/null)"
  [ -n "$PW" ] || { log "no pass autodl/westd — sleeping"; sleep "$POLL_S"; continue; }
  pull_meta "$PW"

  # self-heal: any LOCAL head with no CSV row (serve segfault left it stranded) -> re-eval
  for lh in "$HEADDIR"/steps_*_pytorch_model_head.pt; do
    [ -e "$lh" ] || continue
    bn=$(basename "$lh")
    already_done "$bn" && continue
    log "self-heal: local $bn has no result -> eval"
    eval_one "$bn" || log "  $bn: self-heal eval failed, retry next loop"
  done

  # poll cloud heads dir; pull + eval each new head with step >= MIN_STEP
  names=$(sshpass -p "$PW" ssh $SSHOPT -p "$CLOUD_PORT" root@"$CLOUD_HOST" \
            "ls $CLOUD_HEADS 2>/dev/null | grep -E 'steps_[0-9]+_pytorch_model_head.pt'" 2>/dev/null)
  for fn in $names; do
    gs=$(echo "$fn" | sed -n 's/steps_\([0-9]*\)_.*/\1/p')
    [ -n "$gs" ] || continue
    [ "$gs" -lt "$MIN_STEP" ] && continue
    already_done "$fn" && continue
    [ -f "$HEADDIR/$fn" ] || {
      log "new head: $fn (gs=$gs) -> pulling (~0.6G)"
      sshpass -p "$PW" rsync -az --partial --timeout=600 -e "ssh -p $CLOUD_PORT $SSHOPT" \
        root@"$CLOUD_HOST":"$CLOUD_HEADS/$fn" "$HEADDIR/" >> "$LOG" 2>&1 || { log "  $fn: rsync failed, retry next loop"; continue; }
    }
    eval_one "$fn" || log "  $fn: eval failed, retry next loop"
  done
  sleep "$POLL_S"
done
