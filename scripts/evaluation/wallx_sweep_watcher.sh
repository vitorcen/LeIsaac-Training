#!/usr/bin/env bash
# Auto pull + quick-eval sweep watcher for wall-x cloud training checkpoints.
#
# Polls the AutoDL training box for new ckpt dirs under wallx-outputs/, rsyncs
# each (minus optimizer.bin) to a local sweep dir, runs a headless 3-round quick
# eval against a freshly-served wall-x policy, and appends the result to a CSV
# leaderboard. Designed to run unattended over the ~29h cloud training run; the
# winner gets a strict 20-round eval later (separate step).
#
# Cloud keeps only the newest step-ckpt (keep_last_step_ckpts=1) but epoch dirs
# {0,1,2,3} are never pruned and each step-ckpt survives ~2.2h before the next
# save prunes it, so a 3-min poll never misses one.
#
# Env overrides: POLL_S EVAL_ROUNDS EPISODE_LENGTH_S STEP_HZ ACTION_HORIZON PORT
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP=$ROOT/LeIsaac/outputs/wallx-sweep
CSV=$SWEEP/sweep.csv
LOG=$SWEEP/watcher.log
BASE=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--x-square-robot--wall-oss-0.5/snapshots/f2119fd2bc888c249ed42a4004f42dc09ed1fa84
WALLX_PY=$(conda info --base)/envs/wallx/bin/python
SERVE=$ROOT/LeIsaac/scripts/evaluation/serve_wallx.py
PROMPT="Pick three oranges and put them into the plate, then reset the arm to rest state."

PORT="${PORT:-8001}"
POLL_S="${POLL_S:-180}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-60}"
STEP_HZ="${STEP_HZ:-60}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"

CLOUD_HOST=connect.westd.seetacloud.com
CLOUD_PORT=12710
CLOUD_OUT=/root/autodl-tmp/wallx-outputs
STAGE=/root/autodl-tmp/wallx-stage   # prune-safe snapshot dir for step-ckpts
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

mkdir -p "$SWEEP"
[ -f "$CSV" ] || echo "ckpt,global_step,success_rate,successes,rounds,oranges_placed,oranges_total,timestamp" > "$CSV"

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

already_done() { grep -q "^$1," "$CSV"; }

# eval_one <ckpt_name> : serve + headless quick eval, append CSV row. 0=ok 1=fail
eval_one() {
  local name="$1" dir="$SWEEP/$1"
  for f in model.safetensors normalizer_action.pth normalizer_propri.pth config.yml; do
    [ -f "$dir/$f" ] || { log "  $name: missing $f (prune race gutted it?), skip"; return 1; }
  done

  # Defer to any concurrent eval — the StarVLA sweep (other branch) shares this 4090,
  # and two evals (each ~serve 8.4G + Isaac 7G) overflow 24G → the OOM killer SIGKILLs
  # our Isaac mid-episode ("已杀死", no Final line). Wait for enough free VRAM before
  # claiming the GPU instead of burning a 2min Isaac load just to be killed.
  for _ in $(seq 1 30); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${free:-0}" -ge 16000 ] && break
    log "  $name: only ${free}MiB free (<16G), waiting for concurrent eval to finish..."
    sleep 60
  done

  log "  $name: serving..."
  rm -f /tmp/sw_serve.log
  nohup env CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$WALLX_PY" -u "$SERVE" --ckpt "$dir" --base "$BASE" --port "$PORT" --prompt "$PROMPT" \
    > /tmp/sw_serve.log 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 1 120); do
    grep -q "serving on ws" /tmp/sw_serve.log && { up=1; break; }
    kill -0 "$spid" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    log "  $name: serve FAILED (see /tmp/sw_serve.log)"; kill -9 "$spid" 2>/dev/null; return 1
  fi

  log "  $name: eval ${EVAL_ROUNDS}-round GUI (DISPLAY=${DISPLAY:-:0})..."
  rm -f /tmp/sw_eval.log
  # GUI on (no --headless) so the user can eyeball the arm/scene for visual anomalies.
  ( cd "$ROOT/LeIsaac" && export DISPLAY="${DISPLAY:-:0}" && conda run -n isaaclab --no-capture-output \
    python -u scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --eval_rounds="$EVAL_ROUNDS" --episode_length_s="$EPISODE_LENGTH_S" --step_hz="$STEP_HZ" \
      --policy_type=wallx --policy_host=localhost --policy_port="$PORT" --policy_timeout_ms=60000 \
      --policy_action_horizon="$ACTION_HORIZON" \
      --policy_language_instruction="$PROMPT" \
      --device=cuda --enable_cameras ) > /tmp/sw_eval.log 2>&1

  kill -9 "$spid" 2>/dev/null; sleep 3

  # parse: "Final success rate: <sr>  [<succ>/<rounds>], oranges: <pl>/<tot>"
  local fl
  fl=$(grep "Final success rate" /tmp/sw_eval.log | tail -1)
  if [ -z "$fl" ]; then log "  $name: eval produced no Final line (crash? /tmp/sw_eval.log)"; return 1; fi
  local sr succ rnd pl tot
  sr=$(echo "$fl"   | sed -n 's/.*success rate:[[:space:]]*\([0-9.]*\).*/\1/p')
  succ=$(echo "$fl" | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\1/p')
  rnd=$(echo "$fl"  | sed -n 's/.*\[\([0-9]*\)\/\([0-9]*\)\].*/\2/p')
  pl=$(echo "$fl"   | sed -n 's/.*oranges:[[:space:]]*\([0-9]*\)\/\([0-9]*\).*/\1/p')
  tot=$(echo "$fl"  | sed -n 's/.*oranges:[[:space:]]*\([0-9]*\)\/\([0-9]*\).*/\2/p')
  local gs
  gs=$("$WALLX_PY" -c "import torch;print(torch.load('$dir/global_step.pth')['global_step'])" 2>/dev/null || echo "")
  echo "$name,$gs,$sr,$succ,$rnd,$pl,$tot,$(date +%FT%T)" >> "$CSV"
  log "  $name: gs=$gs success=$sr [$succ/$rnd] oranges=$pl/$tot"
  return 0
}

log "=== wall-x sweep watcher started (poll=${POLL_S}s, eval=${EVAL_ROUNDS}-round) ==="
# clear any orphaned staging snapshots from a previously-crashed watcher
sshpass -p "$(pass autodl/westd 2>/dev/null)" ssh $SSHOPT -p "$CLOUD_PORT" \
  root@"$CLOUD_HOST" "rm -rf $STAGE" 2>/dev/null
while true; do
  SSHPASS="$(pass autodl/westd 2>/dev/null)"
  names=$(sshpass -p "$SSHPASS" ssh $SSHOPT -p "$CLOUD_PORT" root@"$CLOUD_HOST" "ls $CLOUD_OUT 2>/dev/null" 2>/dev/null)
  if [ -n "$names" ]; then
    for name in $names; do
      already_done "$name" && continue
      log "new ckpt: $name -> pulling"
      # Step-ckpts ({epoch}_{gs}, name has '_') are pruned by the trainer (keep_last=1)
      # ~7min after creation — faster than the ~10min pull, which otherwise guts the
      # normalizers mid-transfer (model.safetensors survives via the open fd, the small
      # files after it vanish). So snapshot step-ckpts to a cloud staging dir first
      # (tiny files then the 8.3G model); the trainer's prune never touches staging.
      # Epoch anchors (pure-numeric) are never pruned — pull them live.
      src="$CLOUD_OUT/$name"; staged=0
      case "$name" in
        *_*)
          log "  $name: staging on cloud (prune-safe snapshot)"
          if sshpass -p "$SSHPASS" ssh $SSHOPT -p "$CLOUD_PORT" root@"$CLOUD_HOST" \
               "mkdir -p $STAGE/$name && cd $CLOUD_OUT/$name && cp -f config.yml global_step.pth normalizer_action.pth normalizer_propri.pth model.safetensors $STAGE/$name/" 2>>"$LOG"; then
            src="$STAGE/$name"; staged=1
          else
            log "  $name: stage failed (already pruned?), skip"; continue
          fi
          ;;
      esac
      if sshpass -p "$SSHPASS" rsync -az --timeout=600 \
           -e "ssh -p $CLOUD_PORT $SSHOPT" --exclude=optimizer.bin \
           root@"$CLOUD_HOST":"$src/" "$SWEEP/$name/" >> "$LOG" 2>&1; then
        eval_one "$name" || log "  $name: eval failed, will retry next loop"
      else
        log "  $name: rsync failed, retry next loop"
      fi
      # free the cloud staging snapshot regardless of eval outcome
      [ "$staged" = 1 ] && sshpass -p "$SSHPASS" ssh $SSHOPT -p "$CLOUD_PORT" \
        root@"$CLOUD_HOST" "rm -rf $STAGE/$name" 2>/dev/null
    done
  fi
  # training-done sentinel: epoch 3 ckpt evaluated
  already_done "3" && { log "=== epoch 3 evaluated — sweep complete ==="; break; }
  sleep "$POLL_S"
done
