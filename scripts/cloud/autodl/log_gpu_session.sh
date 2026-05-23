#!/usr/bin/env bash
# Append a GPU mode session record to LeIsaac/outputs/autodl_gpu_hours.csv.
# Goal: every GPU mode period gets a row → habit-build cost reflection at the
# end of each session ("did I spend GPU time on things that actually needed GPU?").
#
# Usage on AutoDL after a GPU session ends (or just before powering down):
#   bash log_gpu_session.sh "purpose summary" [gpu_hourly_rmb] [start_iso] [end_iso]
#
# Defaults:
#   gpu_hourly_rmb = 7.0  (RTX PRO 6000 Blackwell on AutoDL approximate)
#   start_iso = $(date -d 'this-session-start') — must be passed explicitly
#   end_iso = now
#
# Example:
#   bash log_gpu_session.sh "smoke 100 step + 10k step training" 7.0 \
#                           2026-05-22T21:13:00 2026-05-22T23:30:00

set -euo pipefail

PURPOSE="${1:?usage: bash $0 \"purpose\" [gpu_rmb] [start_iso] [end_iso]}"
RMB_PER_HOUR="${2:-7.0}"
START_ISO="${3:?must pass start_iso (YYYY-MM-DDTHH:MM)}"
END_ISO="${4:-$(date -Iminutes)}"

# CSV target lives in repo (so it commits + survives instance teardown).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CSV="${CSV:-$REPO_ROOT/outputs/autodl_gpu_hours.csv}"

# duration in minutes
START_SEC=$(date -d "$START_ISO" +%s)
END_SEC=$(date -d "$END_ISO" +%s)
DUR_MIN=$(( (END_SEC - START_SEC) / 60 ))
COST=$(awk -v m="$DUR_MIN" -v r="$RMB_PER_HOUR" 'BEGIN{printf "%.2f", m*r/60}')

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

# header if file doesn't exist
if [[ ! -f "$CSV" ]]; then
    mkdir -p "$(dirname "$CSV")"
    echo "timestamp_start,timestamp_end,duration_min,gpu_type,gpu_hourly_rmb,cost_rmb,purpose,session_notes" > "$CSV"
fi

NOTES="${SESSION_NOTES:-}"
echo "$START_ISO,$END_ISO,$DUR_MIN,$GPU,$RMB_PER_HOUR,$COST,$PURPOSE,$NOTES" >> "$CSV"
echo "[log_gpu_session] appended:"
tail -1 "$CSV"
echo
echo "=== running totals ==="
awk -F, 'NR>1 {n++; total_min+=$3; total_cost+=$6} END {
    printf "sessions: %d\n", n
    printf "total GPU minutes: %d (%.1f hours)\n", total_min, total_min/60
    printf "total cost: ¥%.2f\n", total_cost
}' "$CSV"
