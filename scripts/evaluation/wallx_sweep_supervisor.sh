#!/usr/bin/env bash
# Self-healing supervisor for wallx_sweep_watcher.sh. The watcher vanished once
# mid-run (session/nohup teardown), stalling the sweep for hours. This respawns
# it whenever it exits, UNLESS the sweep is complete (epoch-3 ckpt already in the
# CSV) — so it survives the full ~17h unattended training without babysitting.
set -uo pipefail
ROOT=/home/david/work/isaaclab-experience
WATCHER=$ROOT/LeIsaac/scripts/evaluation/wallx_sweep_watcher.sh
CSV=$ROOT/LeIsaac/outputs/wallx-sweep/sweep.csv
SUP_LOG=$ROOT/LeIsaac/outputs/wallx-sweep/supervisor.log
export DISPLAY="${DISPLAY:-:0}"

mkdir -p "$(dirname "$CSV")"
echo "[$(date +%T)] supervisor up" | tee -a "$SUP_LOG"
while true; do
  if grep -q "^3," "$CSV" 2>/dev/null; then
    echo "[$(date +%T)] epoch-3 ckpt evaluated — sweep complete, supervisor exit" | tee -a "$SUP_LOG"
    break
  fi
  echo "[$(date +%T)] (re)spawning watcher" | tee -a "$SUP_LOG"
  bash "$WATCHER"
  rc=$?
  echo "[$(date +%T)] watcher exited rc=$rc; respawn in 30s" | tee -a "$SUP_LOG"
  sleep 30
done
