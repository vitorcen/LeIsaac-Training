#!/usr/bin/env bash
# STRICT 20-round eval for a single StarVLA ckpt + placed-count distribution P(placed=k).
# Mirrors wallx_strict_eval.sh, but serves via serve_starvla.py and supports VLM
# quantization (VLM_QUANT=8|4) so the same ckpt can be benchmarked bf16 vs int8/nf4 —
# the apples-to-apples way to prove "8bit ≈ bf16" instead of eyeballing 2 episodes.
#
# Usage:
#   VLM_QUANT=8 starvla_strict_eval.sh <ckpt.pt>       # 20-round, VLM int8
#   VLM_QUANT=0 ROUNDS=20 starvla_strict_eval.sh <ckpt.pt>   # bf16 baseline
#   GUI=1 VLM_QUANT=8 starvla_strict_eval.sh <ckpt.pt>  # watch the window (slower)
#
# Writes <ckpt_dir>/strict_eval_q<QUANT>.json (per-round metrics) + a distribution
# .md/.svg next to it via aggregate_distribution.py. Headless by default (speed).
set -uo pipefail

CKPT="${1:?usage: starvla_strict_eval.sh <ckpt.pt>}"
CKPT="$(readlink -f "$CKPT")"
[ -f "$CKPT" ] || { echo "no such ckpt: $CKPT"; exit 1; }
ROOT=/home/david/work/isaaclab-experience
# Fool-proof backbone resolution: infer 2B/4B/8B from the ckpt path and pick the matching
# local Qwen3-VL snapshot. A hardcoded BASE silently mis-pairs (e.g. 4B base + 8B ckpt) and
# the load fails with a size mismatch the serve-retry loop misreads as "corruption" (4 wasted
# attempts). Derive from the data that's already there; fail loud if it can't be inferred.
if [ -z "${BASE:-}" ]; then
  case "$CKPT" in
    *2b*|*2B*) VLM_TAG=2B ;;
    *4b*|*4B*) VLM_TAG=4B ;;
    *8b*|*8B*) VLM_TAG=8B ;;
    *) echo "[strict] cannot infer VLM backbone (2b/4b/8b) from ckpt path; set BASE=<snapshot dir>"; exit 1 ;;
  esac
  BASE="$(ls -d /home/david/.cache/huggingface/hub/models--Qwen--Qwen3-VL-${VLM_TAG}-Instruct/snapshots/*/ 2>/dev/null | head -1)"
  BASE="${BASE%/}"
  [ -n "$BASE" ] && [ -d "$BASE" ] || { echo "[strict] no local Qwen3-VL-${VLM_TAG}-Instruct snapshot — hf download it first, or set BASE=<dir>"; exit 1; }
  echo "[strict] inferred backbone=$VLM_TAG -> BASE=$BASE"
fi
# env-overridable: Qwen3.5 backbone needs the transformers-5.2 env (starvla_eval_qwen35),
# Qwen3-VL/Cosmos use the default tf-4.57 env (starvla_eval).
STARVLA_PY="${STARVLA_PY:-/home/david/miniconda3/envs/starvla_eval/bin/python}"
SERVE=$ROOT/LeIsaac/scripts/evaluation/serve_starvla.py
AGG=$ROOT/scripts/benchmark/aggregate_distribution.py
PROMPT="${PROMPT:-Grab orange and place into plate}"

PORT="${PORT:-8013}"
ROUNDS="${ROUNDS:-20}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-120}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-180}"
STEP_HZ="${STEP_HZ:-30}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
STUCK_WINDOW_S="${STUCK_WINDOW_S:-30}"
STUCK_EPS_RAD="${STUCK_EPS_RAD:-0.05}"
VLM_QUANT="${VLM_QUANT:-8}"
GUI="${GUI:-0}"

CKDIR="$(dirname "$CKPT")"
# A non-20-round run is a demo/smoke, NOT the strict result — write it to a separate file so
# it can't clobber the canonical strict_eval_q<QUANT>.json (same ckpt+quant = same name).
DEMOSUF=""; [ "${ROUNDS:-20}" != 20 ] && DEMOSUF="_r${ROUNDS}demo"
METRICS="$CKDIR/strict_eval_q${VLM_QUANT}${DEMOSUF}.json"
# label from run dir name (e.g. so101_pickorange_qwen3vl8b_gr00t) so 2B/4B/8B don't mislabel
LABEL="${LABEL:-StarVLA-$(basename "$(dirname "$CKDIR")")-q${VLM_QUANT}-$(echo "$CKPT" | sed -n 's/.*steps_\([0-9]*\)_.*/s\1/p')}"

# quant env for serve
QENV=()
[ "$VLM_QUANT" = 8 ] && QENV=(STARVLA_VLM_8BIT=1)
[ "$VLM_QUANT" = 4 ] && QENV=(STARVLA_VLM_4BIT=1)

echo "[strict] serving $CKPT q=${VLM_QUANT} on :$PORT"
# serve with retry: large-ckpt torch load intermittently corrupts (segfault OR a weird
# yaml/sre AttributeError — same heap-corruption root cause, ~40%); just relaunch.
SPID="" up=0
for attempt in 1 2 3 4; do
  rm -f /tmp/sv_strict_serve.log
  nohup env CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${QENV[@]}" \
    "$STARVLA_PY" -u "$SERVE" --ckpt "$CKPT" --base "$BASE" --port "$PORT" --img_size 448 --prompt "$PROMPT" \
    > /tmp/sv_strict_serve.log 2>&1 &
  SPID=$!
  up=0
  for _ in $(seq 1 200); do
    grep -q "SERVE_READY" /tmp/sv_strict_serve.log 2>/dev/null && { up=1; break; }
    ss -tln 2>/dev/null | grep -q ":$PORT " && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  [ "$up" = 1 ] && break
  # A size-mismatch is a backbone/ckpt config error, NOT corruption — retrying is futile.
  # Abort loudly so a wrong BASE surfaces immediately instead of burning 4 attempts.
  if grep -q "size mismatch" /tmp/sv_strict_serve.log 2>/dev/null; then
    echo "[strict] BACKBONE MISMATCH: BASE=$BASE does not match this ckpt (size mismatch in load). Wrong VLM size — fix BASE."
    kill -9 "$SPID" 2>/dev/null; exit 1
  fi
  echo "[strict] serve attempt $attempt failed (corruption?), retrying"; kill -9 "$SPID" 2>/dev/null; sleep 3
done
[ "$up" = 1 ] || { echo "[strict] serve FAILED after 4 attempts"; tail -15 /tmp/sv_strict_serve.log; exit 1; }
grep -i 'quantized' /tmp/sv_strict_serve.log | tail -1
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1

RENDER=(--headless)
[ "$GUI" = 1 ] && RENDER=()
BENCH="$ROOT/scripts/benchmark"
# run_eval ROUNDS METRICS_OUT LOGFILE  — one policy_inference pass against the live serve.
run_eval(){
  ( cd "$ROOT/LeIsaac" && DISPLAY="${DISPLAY:-:0}" conda run -n isaaclab --no-capture-output \
    python -u scripts/evaluation/policy_inference.py \
      --task=LeIsaac-SO101-PickOrange-v0 \
      --eval_rounds="$1" --episode_length_s="$EPISODE_LENGTH_S" --step_hz="$STEP_HZ" \
      --max_round_wall_s="$MAX_ROUND_WALL_S" --stuck_window_s="$STUCK_WINDOW_S" --stuck_eps_rad="$STUCK_EPS_RAD" \
      --policy_type=starvla --policy_host=localhost --policy_port="$PORT" --policy_timeout_ms=60000 \
      --policy_action_horizon="$ACTION_HORIZON" --policy_language_instruction="$PROMPT" \
      --metrics_out="$2" --metrics_label="$LABEL" \
      --device=cuda "${RENDER[@]}" --enable_cameras ) 2>&1 | tee "$3" | grep -iE "success rate|oranges|Episode [0-9]" | tail -8
}

echo "[strict] ${ROUNDS}-round eval (q=${VLM_QUANT}, ep=${EPISODE_LENGTH_S}s, GUI=${GUI})..."
run_eval "$ROUNDS" "$METRICS" /tmp/sv_strict_eval.log

# --- serve-hang retest loop ----------------------------------------------------
# A serve drop/stall (websocket ConnectionClosedError, OOM, teardown hang) freezes the
# arm for a whole episode -> fake 0 that isn't the policy's true ability. Detect those
# (client got 'no actions' for most of the episode) and re-run exactly that many rounds,
# keeping the serve alive, then merge only VALID episodes. RETEST=0 to disable.
RETEST="${RETEST:-1}"; RETEST_MAX="${RETEST_MAX:-2}"; HANG_THRESH="${HANG_THRESH:-30}"
if [ "$RETEST" = 1 ] && [ -f "$METRICS" ]; then
  python3 "$BENCH/flag_serve_hang.py" --log /tmp/sv_strict_eval.log --metrics "$METRICS" --threshold "$HANG_THRESH"
  nhang0=$?
  if [ "$nhang0" -gt 0 ]; then
    MERGED="${METRICS%.json}.merged.json"; pass=0
    pairs=("$METRICS:/tmp/sv_strict_eval.log")
    while [ "$pass" -lt "$RETEST_MAX" ]; do
      python3 "$BENCH/merge_valid_episodes.py" --pairs "${pairs[@]}" --target "$ROUNDS" --threshold "$HANG_THRESH" --out "$MERGED"
      n=$(python3 -c "import json;print(json.load(open('$MERGED'))['rounds'])" 2>/dev/null || echo 0)
      [ "$n" -ge "$ROUNDS" ] && break
      pass=$((pass+1)); deficit=$((ROUNDS - n))
      rmet="${METRICS%.json}.retest${pass}.json"; rlog="/tmp/sv_strict_eval.retest${pass}.log"
      echo "[strict] RETEST pass $pass: need $deficit more valid round(s) -> re-running $deficit (serve kept alive)..."
      run_eval "$deficit" "$rmet" "$rlog"
      pairs+=("$rmet:$rlog")
    done
    python3 "$BENCH/merge_valid_episodes.py" --pairs "${pairs[@]}" --target "$ROUNDS" --threshold "$HANG_THRESH" --out "$MERGED"
    cp "$MERGED" "$METRICS"   # promote the hang-free merged set to the canonical metrics
    echo "[strict] retest done — final metrics use only valid (non-serve-hang) episodes."
  fi
fi
# --------------------------------------------------------------------------------

kill -9 "$SPID" 2>/dev/null; sleep 3
echo "[strict] === result ==="
grep "Final success rate" /tmp/sv_strict_eval.log | tail -1
if [ -f "$METRICS" ]; then
  python3 "$AGG" "$METRICS" --out "${METRICS%.json}.distribution.md" --svg "${METRICS%.json}.distribution.svg"
  echo "[strict] distribution -> ${METRICS%.json}.distribution.md"
  cat "${METRICS%.json}.distribution.md"
else
  echo "[strict] NO metrics produced — eval crashed (see /tmp/sv_strict_eval.log)"
fi
