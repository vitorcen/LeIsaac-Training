#!/bin/bash
# Watchdog for SO-101 StarVLA training. Polls; if the trainer dies before
# producing final_model, relaunches with RESUME=1 (loads latest checkpoint).
exec >> /root/starvla_watchdog.log 2>&1
RUN=${RUN:-/root/run_train.sh}
OUT=/root/autodl-tmp/starvla-outputs/so101_pickorange_qwengr00t
echo "=== watchdog start $(date) ==="
restarts=0
while true; do
  sleep 120
  # done?
  if [ -d "$OUT/final_model" ]; then
    echo "$(date) final_model present -> training complete; watchdog exit"
    break
  fi
  # alive?
  if pgrep -f "train_starvla.py" >/dev/null; then
    continue
  fi
  # dead & not complete -> resume
  restarts=$((restarts+1))
  if [ $restarts -gt 30 ]; then echo "$(date) too many restarts ($restarts); giving up"; break; fi
  echo "$(date) trainer dead, RESUME=1 relaunch #$restarts"
  RESUME=1 nohup bash $RUN >/dev/null 2>&1 &
  sleep 90
done
