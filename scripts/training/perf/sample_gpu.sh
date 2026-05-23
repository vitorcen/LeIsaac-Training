#!/usr/bin/env bash
# Sample GPU util + memory + SM clock + CPU util every 1 s.
# Usage:
#   sample_gpu.sh <out_csv> [duration_s=1500] [interval_s=1]
#
# CSV columns: time,gpu_util,mem_used_mib,sm_clock,cpu_pct
#
# Pair with scripts/training/perf/analyze_gpu_csv.py to get mid-window stats.
# Generic — used across GR00T / DreamZero / π0.5 / X-VLA / SmolVLA training.
set -eu
OUT="${1:?out_csv required}"
DUR="${2:-1500}"
INT="${3:-1}"

mkdir -p "$(dirname "$OUT")"
echo "time,gpu_util,mem_used_mib,sm_clock,cpu_pct" > "$OUT"
end=$(($(date +%s) + DUR))
while [[ $(date +%s) -lt $end ]]; do
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.current.sm \
                     --format=csv,noheader,nounits | tr -d ' ')
    cpu=$(top -bn1 | head -2 | tail -1 | awk '{print 100-$8}')
    echo "$(date +%H:%M:%S),$line,$cpu" >> "$OUT"
    sleep "$INT"
done
