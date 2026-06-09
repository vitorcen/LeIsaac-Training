#!/usr/bin/env bash
# Unified single-GPU eval QUEUE for all frozen-VLM StarVLA families (PI_v3 / Cosmos / Qwen3.5).
# Each family is pulled as HEADS only; a sweep point = merge {vlm_base + head} -> full ckpt ->
# serve (family's env + base + quant) -> strict_eval (GUI, 3-round) -> record -> rm merged.
# Serial by design (one local GPU): one job at a time, newest-step-first across families.
# Waits if an EXTERNAL serve/eval is using the GPU (e.g. a standalone 20-round strict), so it
# never double-books. All results land in ONE CSV the user can `cat` anytime.
#
# Check results:  column -t -s, $OUT/eval_queue.csv      |  live log: tail -f /tmp/eval_queue.log
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=$ROOT/LeIsaac/outputs
TOOLS=$OUT/_head_sweep_tools
MERGE=$TOOLS/merge_head.py
STRICT=$ROOT/LeIsaac/scripts/evaluation/starvla_strict_eval.sh
EVAL_ENV=$(conda info --base)/envs/starvla_eval/bin/python
QWEN35_ENV=$(conda info --base)/envs/starvla_eval_qwen35/bin/python
ROUNDS="${ROUNDS:-3}"; GUI="${GUI:-1}"; POLL_S="${POLL_S:-45}"
CSV=$OUT/eval_queue.csv
LOG=/tmp/eval_queue.log
PIDF=/tmp/eval_queue.pid
# single-instance pidfile guard (rsync-style flock inheritance not an issue here, but keep it simple)
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then echo "queue already running $(cat "$PIDF")"; exit 0; fi
echo $$ >"$PIDF"
[ -f "$CSV" ] || echo "family,step,placed_per_ep,E_oranges_pct,timestamp" > "$CSV"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }

g8b=$(ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ 2>/dev/null|head -1); g8b=${g8b%/}
cosmos=$(ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/*/ 2>/dev/null|head -1); cosmos=${cosmos%/}
q35=$(ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3.5-2B/snapshots/*/ 2>/dev/null|head -1); q35=${q35%/}
q35_4b=$(ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ 2>/dev/null|head -1); q35_4b=${q35_4b%/}
q35_9b=$(ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null|head -1); q35_9b=${q35_9b%/}

# name|heads_dir|vlm_base|base_vlm|env_py|quant|min_step|min_mb|ckdir
FAM=(
  "piv3|$OUT/starvla-pi_v3-run/heads|$TOOLS/vlm_base_8b.pt|$g8b|$EVAL_ENV|8|0|1200|$OUT/starvla-pi_v3-run/checkpoints"
  "cosmos|$OUT/starvla-cosmos-run/heads|$TOOLS/vlm_base_cosmos.pt|$cosmos|$EVAL_ENV|8|0|300|$OUT/starvla-cosmos-run/checkpoints"
  "qwen35|$OUT/starvla-qwen35-2b-run/heads|$TOOLS/vlm_base_qwen35_2b.pt|$q35|$QWEN35_ENV|0|5000|700|$OUT/starvla-qwen35-2b-run/checkpoints"
  "qwen35_4b|$OUT/starvla-qwen35-4b-run/heads|$TOOLS/vlm_base_qwen35_4b.pt|$q35_4b|$QWEN35_ENV|0|5000|1000|$OUT/starvla-qwen35-4b-run/checkpoints"
  "qwen35_9b|$OUT/starvla-qwen35-9b-run/heads|$TOOLS/vlm_base_qwen35_9b.pt|$q35_9b|$QWEN35_ENV|8|4000|1000|$OUT/starvla-qwen35-9b-run/checkpoints"
)

gpu_busy(){ pgrep -f "serve_starvla.py" >/dev/null 2>&1 || pgrep -f "policy_inference.py" >/dev/null 2>&1; }

# ---- DP (diffusion policy) cooperative filler ------------------------------------
# Single-GPU courtesy: starvla sweep evals ALWAYS preempt. DP only runs when nothing
# is pending. DP resumes +1 epoch per turn (clean ckpt at each epoch boundary), so a
# pending sweep ckpt waits at most one ~9-min epoch chunk. Eval of DP epoch ckpts
# (>=3ep) is best-effort and never crashes the queue.
DP_ON="${DP_ON:-1}"
DP_OUT=$OUT/dp-6ep-earlytest
DP_EPOCH=4537                 # 1 epoch = 36293 frames / batch 8
DP_TARGET=27220               # 6 epochs
DP_TRAIN_PY=$(conda info --base)/envs/lerobot-dp311/bin/lerobot-train
DP_EVAL_FROM=13611            # eval epoch ckpts >= 3 ep
DP_SERVE_PY=$(conda info --base)/envs/lerobot-v040/bin/python  # patched async server
DP_REPO=$HOME/work/lerobot-v040
dp_last_step(){ python3 -c "import json;print(json.load(open('$DP_OUT/checkpoints/last/training_state/training_step.json'))['step'])" 2>/dev/null || echo 0; }
# train one epoch chunk; 0=ran, 1=complete/disabled
dp_chunk(){
  [ "$DP_ON" = 1 ] && [ ! -f "$DP_OUT/.dp_off" ] && [ -d "$DP_OUT/checkpoints/last/pretrained_model" ] || return 1
  local last; last=$(dp_last_step); [ "${last:-0}" -ge "$DP_TARGET" ] && return 1
  local next=$(( (last/DP_EPOCH + 1) * DP_EPOCH )); [ "$next" -gt "$DP_TARGET" ] && next=$DP_TARGET
  log "[dp] train chunk $last -> $next (pyav/NW0, save_freq 500)"
  # kernel-6.17 + torchcodec heap-corrupts a few hundred steps into a resume -> segfault.
  # pyav+NW0 survives longer but a full 4537-step epoch chunk still rarely completes. So
  # save_freq=500: every survived 500-step segment is checkpointed, the next chunk resumes
  # from there, and DP creeps forward despite intermittent segfaults. Backoff disables only
  # on a chunk that made ZERO progress (truly stuck), not on a mid-chunk segfault that saved.
  ( cd "$ROOT" && "$DP_TRAIN_PY" --config_path="$DP_OUT/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true --steps="$next" --save_freq=500 --num_workers=0 --dataset.video_backend=pyav >>"$LOG" 2>&1 )
  local rc=$? now; now=$(dp_last_step)
  if [ "${now:-0}" -gt "$last" ]; then
    echo 0 >"$DP_OUT/.dp_fails"; log "[dp] progressed $last -> $now (rc=$rc)"
    ls -dt "$DP_OUT"/checkpoints/[0-9]*/ 2>/dev/null | tail -n +4 | xargs -r rm -rf   # keep last 3 dirs
  else
    local n=$(( $(cat "$DP_OUT/.dp_fails" 2>/dev/null || echo 0) + 1 )); echo "$n" >"$DP_OUT/.dp_fails"
    log "[dp] NO progress (rc=$rc, still $now) — stuck fail $n/4"
    [ "$n" -ge 4 ] && { touch "$DP_OUT/.dp_off"; log "[dp] DISABLED after $n stuck fails. rm $DP_OUT/.dp_off to retry"; }
    sleep 30
  fi
  return 0
}
# eval one not-yet-evaled DP epoch ckpt (>=3ep); 0=evaled one, 1=none pending
dp_eval(){
  [ "$DP_ON" = 1 ] || return 1
  # eval once per crossed epoch (>=3ep) using the current `last` ckpt; epoch-gated so the
  # frequent 500-step saves don't each trigger an eval.
  local now ep step d slug mj o t epf
  now=$(dp_last_step); ep=$(( now / DP_EPOCH ))
  [ "$ep" -ge 3 ] || return 1
  grep -qx "$ep" "$DP_OUT/.dp_evaled_eps" 2>/dev/null && return 1
  d="$DP_OUT/checkpoints/last"; [ -d "$d/pretrained_model" ] || return 1
  step="$now"
  log "[dp] eval ~${ep}ep (step $step)"
  echo "$ep" >>"$DP_OUT/.dp_evaled_eps"
  if ! ss -tln 2>/dev/null | grep -q ':8080 '; then
    ( cd "$DP_REPO" && nohup "$DP_SERVE_PY" -m lerobot.async_inference.policy_server --host 0.0.0.0 --port 8080 >/tmp/dp_policy_server.log 2>&1 & )
    for _ in $(seq 1 30); do sleep 2; ss -tln 2>/dev/null | grep -q ':8080 ' && break; done
  fi
  if ! ss -tln 2>/dev/null | grep -q ':8080 '; then
    log "[dp] server :8080 down, skip $step"; echo "dp,$step,\"SERVER_DOWN\",NA,$(date +%FT%T)" >>"$CSV"; pkill -f async_inference.policy_server 2>/dev/null; return 0
  fi
  slug="dp-6ep-${step}-h8"
  ( cd "$ROOT" && EVAL_ROUNDS="$ROUNDS" EPISODE_LENGTH_S=60 MAX_ROUND_WALL_S=90 STEP_HZ=30 \
      PROMPT="Grab orange and place into plate" \
      bash scripts/benchmark/run_one.sh "$slug" lerobot-diffusion 8 "$d/pretrained_model" lerobot "$slug" >>"$LOG" 2>&1 ) || true
  pkill -f async_inference.policy_server 2>/dev/null; sleep 2
  mj="$ROOT/results/benchmark/${slug}.metrics.json"
  epf=$(awk "BEGIN{printf \"%.1f\", $step*8/36293}")
  if [ -f "$mj" ]; then
    read -r o t < <(python3 -c "import json;m=json.load(open('$mj'));print(m.get('oranges_placed_strict',m.get('oranges_placed_total',0)),m.get('oranges_max_total',$((ROUNDS*3))))" 2>/dev/null)
    echo "dp,$step,\"oranges ${o:-?}/${t:-?} (${epf}ep)\",NA,$(date +%FT%T)" >>"$CSV"; log "[dp] step $step -> oranges ${o:-?}/${t:-?}"
  else
    echo "dp,$step,\"NO_METRICS (${epf}ep)\",NA,$(date +%FT%T)" >>"$CSV"; log "[dp] step $step no metrics"
  fi
  return 0
}

while true; do
  # collect candidates: "step|name|head|vlm_base|base|env|quant|ckdir"
  cands=()
  for spec in "${FAM[@]}"; do
    IFS='|' read -r name hd vlmb base env quant minstep minmb ckdir <<<"$spec"
    [ -d "$hd" ] || continue
    for h in "$hd"/steps_*_head.pt; do
      [ -f "$h" ] || continue
      step=$(echo "$h" | sed -n 's/.*steps_\([0-9]*\)_.*/\1/p')
      [ "${step:-0}" -lt "$minstep" ] && continue
      sz=$(stat -c %s "$h" 2>/dev/null || echo 0); [ "$sz" -gt $((minmb*1000000)) ] || continue
      # skip heads the puller is still writing (mtime within 90s): a half-pulled file can pass
      # the size gate but fail torch.load -> permanent MERGE_FAIL. wait until size is stable.
      [ $(( $(date +%s) - $(stat -c %Y "$h" 2>/dev/null || echo 0) )) -lt 90 ] && continue
      grep -q "^$name,$step," "$CSV" && continue
      cands+=("$step|$name|$h|$vlmb|$base|$env|$quant|$ckdir")
    done
  done
  if [ ${#cands[@]} -eq 0 ]; then
    # no starvla sweep ckpt pending -> fill the gap with DP (eval epoch ckpts first,
    # then train one epoch chunk). starvla always preempts at the next loop iteration.
    if gpu_busy; then sleep "$POLL_S"; else dp_eval || dp_chunk || sleep "$POLL_S"; fi
    continue
  fi
  # Priority: qwen35 family first (the active sweep we watch — its steps are tiny vs
  # piv3/cosmos high steps, so a plain global newest-first sort would starve it forever
  # while the others keep training). Within each group, newest-step-first.
  q35=$(printf '%s\n' "${cands[@]}" | awk -F'|' '$2 ~ /^qwen35/' | sort -t'|' -k1 -n -r)
  rest=$(printf '%s\n' "${cands[@]}" | awk -F'|' '$2 !~ /^qwen35/' | sort -t'|' -k1 -n -r)
  IFS=$'\n' cands=($(printf '%s\n%s\n' "$q35" "$rest" | grep -v '^$')); unset IFS
  # wait if GPU busy with an external eval (e.g. standalone 20-round strict)
  while gpu_busy; do log "GPU busy (external eval), wait"; sleep 30; done

  IFS='|' read -r step name h vlmb base env quant ckdir <<<"${cands[0]}"
  mkdir -p "$ckdir"
  full=$ckdir/steps_${step}_pytorch_model.pt
  log "=== [$name] step $step: merge+eval (q=$quant) ==="
  if [ ! -f "$vlmb" ]; then log "[$name] vlm_base missing ($vlmb) — skip family this round"; echo "$name,$step,\"NO_VLM_BASE\",NA,$(date +%FT%T)" >>"$CSV"; continue; fi
  # merge with retry: kernel-6.17 heap corruption hits merge intermittently (segfault /
  # AttributeError _PosixFlavour / 'int' no stale_possible). 3 tries -> one usually succeeds;
  # only give up (and permanently skip the ckpt) after 3 consecutive corruptions.
  mok=0; for mtry in 1 2 3; do
    "$env" "$MERGE" "$vlmb" "$h" "$full" >>"$LOG" 2>&1 && { mok=1; break; }
    log "[$name] merge try $mtry/3 failed (kernel-6.17 corruption?), retry"; rm -f "$full"; sleep 2
  done
  [ "$mok" = 1 ] || { log "[$name] merge FAIL $step (3 tries)"; echo "$name,$step,\"MERGE_FAIL\",NA,$(date +%FT%T)" >>"$CSV"; rm -f "$full"; continue; }
  # 18-min hard cap: Isaac intermittently hangs on plugin-unload teardown AFTER writing the
  # metrics md, blocking the queue forever (GUI freezes). timeout caps it (metrics already on
  # disk so the result below still parses); then nuke orphaned serve/Isaac so gpu_busy() clears.
  STARVLA_PY="$env" BASE="$base" VLM_QUANT="$quant" ROUNDS="$ROUNDS" GUI="$GUI" \
    EPISODE_LENGTH_S=60 MAX_ROUND_WALL_S=90 DISPLAY="${DISPLAY:-:0}" \
    timeout -k 30 1080 bash "$STRICT" "$full" >>"$LOG" 2>&1
  erc=$?
  [ "$erc" = 124 ] || [ "$erc" = 137 ] && log "[$name] eval TIMEOUT $step (Isaac teardown hang) — killing strays"
  pkill -9 -f "serve_starvla.py" 2>/dev/null; pkill -9 -f "policy_inference.py" 2>/dev/null; sleep 2
  md=$ckdir/strict_eval_q${quant}_r${ROUNDS}demo.distribution.md
  raw=$(grep -oE 'placed_per_ep = \[[^]]*\]' "$md" 2>/dev/null | tail -1)
  pct=$(grep -oE 'E\(oranges/ep\) = [0-9.]+ / [0-9]+ = [0-9.]+%' "$md" 2>/dev/null | grep -oE '[0-9.]+%' | tail -1)
  echo "$name,$step,\"${raw:-NA}\",${pct:-NA},$(date +%FT%T)" >> "$CSV"
  log "[$name] step $step -> ${raw:-NA} ${pct:-NA}"
  rm -f "$full"   # keep head, drop merged full ckpt (re-mergeable anytime)
done
