#!/usr/bin/env bash
# Record a publishable demo video of a lerobot/flow policy running headed in the
# Isaac viewport. Launches the eval (server + policy_inference, GUI on DISPLAY=:0),
# waits for the Isaac window, x11grabs that window region to mp4 for the whole run,
# then stops cleanly. Bench log episode timestamps are printed at the end so a
# highlight clip can be cut from the successful rounds.
#
# Usage: SLUG=flowdp-9800-demo CKPT=outputs/flowdp-eval/009800 POLICY_TYPE=lerobot-flowdp \
#        ROUNDS=6 bash scripts/evaluation/record_demo.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
SLUG="${SLUG:?}"; CKPT="${CKPT:?}"; POLICY_TYPE="${POLICY_TYPE:?}"
ROUNDS="${ROUNDS:-6}"; HORIZON="${HORIZON:-8}"
OUT="${OUT:-$ROOT/outputs/demo_videos}"; mkdir -p "$OUT"
RAW="$OUT/${SLUG}.mp4"
FFMPEG=/usr/bin/ffmpeg
export DISPLAY="${DISPLAY:-:0}"

echo "[rec] launch eval $SLUG ($ROUNDS rounds, headed)"
( LEROBOT_PYTHON="${LEROBOT_PYTHON:-$HOME/miniconda3/envs/lerobot-v044/bin/python}" \
  POLICY_TYPE="$POLICY_TYPE" HORIZON="$HORIZON" \
  EVAL_ROUNDS="$ROUNDS" EPISODE_LENGTH_S=120 MAX_ROUND_WALL_S=180 \
  bash "$ROOT/scripts/evaluation/dp_quick_eval.sh" "$SLUG" "$CKPT" "$SLUG" \
  > "/tmp/rec_${SLUG}.log" 2>&1 ) &
EVAL_PID=$!

# wait for the Isaac viewport window (kit). search a few title variants.
WIN=""
for i in $(seq 1 120); do
  for name in "Isaac Sim" "Isaac Lab" "Viewport" "Kit" "omni"; do
    WIN=$(xdotool search --name "$name" 2>/dev/null | head -1)
    [ -n "$WIN" ] && break
  done
  [ -n "$WIN" ] && break
  kill -0 "$EVAL_PID" 2>/dev/null || { echo "[rec] eval died before window appeared; see /tmp/rec_${SLUG}.log"; exit 1; }
  sleep 5
done
[ -z "$WIN" ] && { echo "[rec] no Isaac window after 10min — fallback full-screen 1920x1080@0,0"; GEO="1920x1080"; OFF="+0,0"; }
if [ -n "$WIN" ]; then
  eval "$(xdotool getwindowgeometry --shell "$WIN")"   # sets X Y WIDTH HEIGHT
  # even dimensions for libx264
  W=$((WIDTH - WIDTH%2)); H=$((HEIGHT - HEIGHT%2))
  GEO="${W}x${H}"; OFF="+${X},${Y}"
  echo "[rec] window $WIN geo=$GEO off=$OFF"
  # raise to top so it isn't occluded, but DO NOT activate — stealing keyboard
  # focus lets stray keystrokes (e.g. the user typing elsewhere) leak in as the
  # viewport's "R" reset key and skip episodes.
  xdotool windowraise "$WIN" 2>/dev/null || true
fi

# give the policy a moment to actually start stepping before we roll
sleep 8
echo "[rec] recording -> $RAW"
"$FFMPEG" -hide_banner -loglevel warning -y \
  -f x11grab -framerate 30 -video_size "$GEO" -i "${DISPLAY}.0${OFF}" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p -crf 20 "$RAW" &
FF_PID=$!

# record until the eval finishes
wait "$EVAL_PID"
echo "[rec] eval done, stopping recorder"
kill -INT "$FF_PID" 2>/dev/null; sleep 3; kill -9 "$FF_PID" 2>/dev/null || true

echo "[rec] ===== episode results (for clipping) ====="
grep -E 'Episode .* (successful|skipped)|now success rate' "/tmp/rec_${SLUG}.log" 2>/dev/null | tail -40
echo "[rec] video: $RAW ($(du -h "$RAW" 2>/dev/null | cut -f1))"
