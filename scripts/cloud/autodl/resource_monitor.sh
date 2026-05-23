#!/usr/bin/env bash
# Continuous resource monitor for AutoDL training runs.
# Logs every INTERVAL seconds: GPU util/mem, disk usage on autodl-tmp,
# latest train_loss + step from the training log file. Run alongside train_n17.sh.
#
# Usage:
#   bash resource_monitor.sh /path/to/train.log [interval_sec]
#
# Outputs:
#   - human-readable rolling tail to stdout
#   - CSV to /root/autodl-tmp/monitor.csv (post-mortem analysis)

set -uo pipefail

TRAIN_LOG="${1:-}"
INTERVAL="${2:-15}"
CSV=/root/autodl-tmp/monitor.csv

if [[ -z "$TRAIN_LOG" ]]; then
    echo "usage: bash $0 <train_log_path> [interval_sec]" >&2
    exit 1
fi

# CSV header if new file
if [[ ! -f "$CSV" ]]; then
    echo "epoch_sec,gpu_util_pct,vram_used_mib,vram_total_mib,vram_pct,disk_used_gb,disk_pct,latest_step,latest_loss" > "$CSV"
fi

START=$(date +%s)
echo "[monitor] start $(date), train_log=$TRAIN_LOG, interval=${INTERVAL}s"
echo "[monitor] CSV → $CSV"
printf "%-8s  %-7s  %-22s  %-14s  %-8s  %-10s\n" "T+sec" "GPU%" "VRAM (MiB / %)" "Disk %used" "Step" "Loss"

while true; do
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START ))

    # nvidia-smi
    SMI=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0, 0, 1")
    GPU_UTIL=$(echo "$SMI" | awk -F',' '{print $1+0}')
    VRAM_USED=$(echo "$SMI" | awk -F',' '{print $2+0}')
    VRAM_TOTAL=$(echo "$SMI" | awk -F',' '{print $3+0}')
    VRAM_PCT=$(( VRAM_USED * 100 / (VRAM_TOTAL>0?VRAM_TOTAL:1) ))

    # disk
    DISK_LINE=$(df --output=used,size,pcent /root/autodl-tmp | tail -1)
    DISK_USED_KB=$(echo "$DISK_LINE" | awk '{print $1}')
    DISK_USED_GB=$(( DISK_USED_KB / 1024 / 1024 ))
    DISK_PCT=$(echo "$DISK_LINE" | awk '{print $3}' | tr -d '%')

    # train log: extract latest step + loss
    STEP=""
    LOSS=""
    if [[ -f "$TRAIN_LOG" ]]; then
        # match HF Trainer log lines like {'loss': 0.42, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.5, 'step': 50}
        # or human-readable lines containing "step XXX  loss YYY"
        LATEST=$(grep -oE "'loss': [0-9.eE+-]+|'step': [0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -20)
        STEP=$(echo "$LATEST" | grep "'step'" | tail -1 | sed "s/'step': //")
        LOSS=$(echo "$LATEST" | grep "'loss'" | tail -1 | sed "s/'loss': //")
    fi
    STEP="${STEP:-—}"
    LOSS="${LOSS:-—}"

    printf "%-8s  %-7s  %-22s  %-14s  %-8s  %-10s\n" \
        "$ELAPSED" "${GPU_UTIL}%" "${VRAM_USED}/${VRAM_TOTAL} (${VRAM_PCT}%)" "${DISK_PCT}%" "$STEP" "$LOSS"
    echo "$ELAPSED,$GPU_UTIL,$VRAM_USED,$VRAM_TOTAL,$VRAM_PCT,$DISK_USED_GB,$DISK_PCT,$STEP,$LOSS" >> "$CSV"

    # safety: warn if VRAM > 90% or disk > 90%
    if [[ $VRAM_PCT -gt 90 ]]; then
        echo "  ⚠️ VRAM > 90% — OOM risk; consider lowering GLOBAL_BATCH" >&2
    fi
    if [[ $DISK_PCT -gt 90 ]]; then
        echo "  ⚠️ disk > 90% — prune callback should be deleting; check LossDrivenPrune logs" >&2
    fi

    sleep "$INTERVAL"
done
