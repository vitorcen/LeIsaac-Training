#!/usr/bin/env bash
# Single-instance while-true puller for all 3 AutoDL boxes.
# Pulls sweep-preservation artifacts (2B full rescued ckpts / PI_v3+Cosmos extracted heads)
# to local, then rm's the cloud copy once a size-verified complete pull lands (frees box disk).
# Requires SSHPASS env (same password all 3 boxes). NEVER hardcode the password here.
set -u
LOG=/tmp/master_pull.log
PIDF=/tmp/master_pull.pid
# PID-file guard (NOT flock): rsync/ssh children inherit an flock fd and keep the lock held
# after this script dies, blocking the next instance. A pidfile checks the *script* PID's
# liveness, which orphaned rsyncs don't share.
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "$(date +%H:%M:%S) live instance $(cat "$PIDF") exists; exit" >>"$LOG"; exit 0
fi
echo $$ >"$PIDF"
: "${SSHPASS:?need SSHPASS env}"; export SSHPASS

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=$ROOT/LeIsaac/outputs
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=12 -o ServerAliveInterval=15"
# tag|port|host|remote-dir|file-glob|local-dir|min-MB
# Per-box rows (fill in YOUR cloud boxes; these were the 2026-06 sweep boxes, now offline).
# Override the whole list via PULL_BOXES env (newline-separated rows of the same format),
# or just edit below. No personal endpoints are committed.
SRC=(
  "qwen35|<PORT>|<USER>@<HOST>|/root/autodl-tmp/starvla-outputs/so101_qwen3_5_2b_pi_v3/heads|steps_*_head.pt|$OUT/starvla-qwen35-2b-run/heads|700"
  "qwen35_4b|<PORT>|<USER>@<HOST>|/root/autodl-tmp/starvla-outputs/so101_qwen3_5_4b_pi_v3/heads|steps_*_head.pt|$OUT/starvla-qwen35-4b-run/heads|900"
  "qwen35_9b|<PORT>|<USER>@<HOST>|/root/autodl-tmp/starvla-outputs/so101_qwen3_5_9b_pi_v3/heads|steps_*_head.pt|$OUT/starvla-qwen35-9b-run/heads|900"
)
[ -n "${PULL_BOXES:-}" ] && mapfile -t SRC <<<"$PULL_BOXES"

rsh(){ timeout 40 sshpass -e ssh -p "$1" $SSH_OPTS "$2" "$3"; }

while true; do
  for row in "${SRC[@]}"; do
    IFS='|' read -r tag port host rdir glob ldir minmb <<<"$row"
    mkdir -p "$ldir"
    # list remote files + sizes (bytes); retry once on empty
    listing="$(rsh "$port" "$host" "ls -la $rdir/$glob 2>/dev/null")"
    [ -z "$listing" ] && { sleep 6; listing="$(rsh "$port" "$host" "ls -la $rdir/$glob 2>/dev/null")"; }
    [ -z "$listing" ] && { sleep 6; continue; }
    while read -r perm links own grp bytes mon day tm path; do
      [ -z "${path:-}" ] && continue
      fn="$(basename "$path")"
      lpath="$ldir/$fn"
      lbytes=$(stat -c %s "$lpath" 2>/dev/null || echo 0)
      if [ "$lbytes" = "$bytes" ] && [ "$bytes" -gt $((minmb*1000000)) ]; then
        # already fully local -> free cloud
        echo "$(date +%H:%M:%S) [$tag] $fn complete local ($((bytes/1000000))MB); rm cloud" >>"$LOG"
        rsh "$port" "$host" "rm -f '$path'"
        continue
      fi
      echo "$(date +%H:%M:%S) [$tag] pulling $fn ($((bytes/1000000))MB) local=$((lbytes/1000000))MB" >>"$LOG"
      timeout 3600 sshpass -e rsync -e "ssh -p $port $SSH_OPTS" --partial --inplace --timeout=120 \
        "root@$host:$path" "$lpath" >>"$LOG" 2>&1
      nbytes=$(stat -c %s "$lpath" 2>/dev/null || echo 0)
      if [ "$nbytes" = "$bytes" ]; then
        echo "$(date +%H:%M:%S) [$tag] $fn DONE ($((nbytes/1000000))MB); rm cloud" >>"$LOG"
        rsh "$port" "$host" "rm -f '$path'"
      else
        echo "$(date +%H:%M:%S) [$tag] $fn partial ($((nbytes/1000000))/$((bytes/1000000))MB), retry next cycle" >>"$LOG"
      fi
    done <<<"$listing"
    sleep 4
  done
  sleep 30
done
