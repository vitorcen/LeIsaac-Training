#!/usr/bin/env bash
# Manage local VLA policy inference servers used by LeIsaac SO-101 PickOrange.
#
# Usage:
#   bash scripts/policy_server.sh start gr00t-n15 [MODEL_PATH]
#   bash scripts/policy_server.sh start gr00t-n16 [MODEL_PATH]
#   bash scripts/policy_server.sh start lerobot
#   bash scripts/policy_server.sh stop  gr00t-n15
#   bash scripts/policy_server.sh stop  gr00t-n16
#   bash scripts/policy_server.sh stop  lerobot
#
# Backends:
#   gr00t-n15  GR00T N1.5 inference_service.py over ZMQ :5555.
#              The server *loads the checkpoint*. MODEL_PATH = HF repo_id
#              (resolved via from_pretrained against the default HF cache) or
#              absolute local directory.
#              Default: LightwheelAI/leisaac-pick-orange-v0
#              Pre-fetch with: bash scripts/download_hf_model.sh LightwheelAI/leisaac-pick-orange-v0
#              Env overrides: GR00T_N15_DIR / GR00T_N15_PYTHON / GR00T_N15_HOST / GR00T_N15_PORT
#
#   gr00t-n16  GR00T N1.6 run_gr00t_server.py over ZMQ :5555 (shares port with
#              N1.5 — only one can listen). embodiment_tag = NEW_EMBODIMENT
#              (uppercase enum, N1.6-specific).
#              Default MODEL_PATH: hi-space/GR00T-N1.6-3B-Pick-Orange
#              Delegates to server/start_server.sh --gr00t-only via env vars.
#
#   lerobot    LeRobot async-inference policy_server :8080.
#              The *client* (policy_inference.py --policy_checkpoint_path=...)
#              selects which model to load, so no MODEL_PATH here.
#              Delegates to server/start_server.sh --lerobot-only.
#
# Idempotent: start is a no-op if the port already listens; stop is a no-op
# if no server is running.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(cd "$ROOT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

ACTION="${1:-}"
BACKEND="${2:-}"
case "$ACTION:$BACKEND" in
    start:gr00t-n15|start:gr00t-n16|start:gr00t-n17|start:lerobot) ;;
    stop:gr00t-n15|stop:gr00t-n16|stop:gr00t-n17|stop:lerobot) ;;
    *)
        echo "usage: $0 {start|stop} {gr00t-n15|gr00t-n16|gr00t-n17|lerobot} [MODEL_PATH]" >&2
        exit 2
        ;;
esac

port_listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"
    else
        netstat -an 2>/dev/null | grep -E "[.:]${port}[[:space:]]+" | grep -qi LISTEN
    fi
}

# -------- GR00T N1.5 ZMQ :5555 --------
n15_dir="${GR00T_N15_DIR:-$ROOT_DIR/../dependencies/Isaac-GR00T-N1.5}"
n15_python="${GR00T_N15_PYTHON:-$HOME/miniconda3/envs/gr00t-n15/bin/python}"
n15_host="${GR00T_N15_HOST:-0.0.0.0}"
n15_port="${GR00T_N15_PORT:-5555}"
n15_pidfile="$LOG_DIR/gr00t_n15_server.pid"
n15_logfile="$LOG_DIR/gr00t_n15_server.log"

start_gr00t_n15() {
    local model_path="${1:-${LEISAAC_N15_CKPT:-LightwheelAI/leisaac-pick-orange-v0}}"

    if port_listening "$n15_port"; then
        echo "[INFO] GR00T N1.5 server already listening on :$n15_port"
        return 0
    fi
    [ -d "$n15_dir/gr00t" ] || { echo "[ERROR] GR00T N1.5 repo not found: $n15_dir" >&2; exit 1; }
    [ -x "$n15_python" ]    || { echo "[ERROR] gr00t-n15 python not found: $n15_python" >&2; exit 1; }
    # Don't pre-validate model_path — from_pretrained accepts both repo_id and
    # local path. If it's a repo_id not in cache, inference_service.py will
    # download it (slow first launch); pre-fetch with download_hf_model.sh.

    local detach=()
    command -v setsid >/dev/null 2>&1 && detach=(setsid)

    echo "[INFO] launching GR00T N1.5 server: model=$model_path port=$n15_port"
    nohup "${detach[@]}" bash -lc "cd '$n15_dir' && PYTHONPATH='$n15_dir' '$n15_python' scripts/inference_service.py --server --model-path '$model_path' --embodiment-tag new_embodiment --data-config so100_dualcam --host '$n15_host' --port '$n15_port'" \
        > "$n15_logfile" 2>&1 < /dev/null &
    echo $! > "$n15_pidfile"
    echo "[INFO] pid=$(cat "$n15_pidfile"), bind=$n15_host:$n15_port"

    # Eagle backbone + DiT head: ~20-30s cold load
    for i in $(seq 1 60); do
        if port_listening "$n15_port"; then
            echo "[INFO] GR00T N1.5 server listening on :$n15_port"
            return 0
        fi
        sleep 1
    done
    echo "[ERROR] server did not come up within 60s; last log lines:" >&2
    tail -n 30 "$n15_logfile" >&2 || true
    return 1
}

stop_gr00t_n15() {
    if [ -f "$n15_pidfile" ]; then
        local pid
        pid=$(cat "$n15_pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
            echo "[INFO] stopped GR00T N1.5 server (pid=$pid)"
        fi
        rm -f "$n15_pidfile"
    fi
    pkill -9 -f "inference_service.py.*--port[= ]*$n15_port" 2>/dev/null || true
}

# -------- GR00T N1.6 / N1.7 ZMQ :5555 (self-contained, on the shared Isaac-GR00T engine) --------
# Launched here directly (like gr00t-n15) so LeIsaac depends ONLY on the shared engine submodule
# under ../dependencies — NOT on any umbrella server/ script. Each release tag = its own engine dir
# + isolated uv venv (N1.6 wants transformers 4.51.3, N1.7 wants 4.57.3); both expose
# gr00t/eval/run_gr00t_server.py. PickOrange always uses embodiment NEW_EMBODIMENT.
gr00t_host="${GR00T_HOST:-0.0.0.0}"
gr00t_port="${GR00T_PORT:-5555}"
gr00t_pidfile="$LOG_DIR/gr00t_server.pid"
gr00t_logfile="$LOG_DIR/gr00t_server.log"

_start_gr00t_engine() {  # $1=engine_dir  $2=model_path
    local engine_dir="$1" model_path="$2"
    if port_listening "$gr00t_port"; then
        echo "[INFO] GR00T server already listening on :$gr00t_port"; return 0
    fi
    [ -d "$engine_dir/gr00t" ] || { echo "[ERROR] GR00T engine not found: $engine_dir" >&2; exit 1; }
    local wrapper_arg=""
    [ "${GR00T_SIM_WRAPPER:-0}" = "1" ] && wrapper_arg="--use-sim-policy-wrapper"
    local detach=(); command -v setsid >/dev/null 2>&1 && detach=(setsid)
    echo "[INFO] launching GR00T server: engine=$engine_dir model=$model_path port=$gr00t_port"
    nohup "${detach[@]}" bash -lc "cd '$engine_dir' && uv run --no-sync python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path '$model_path' --host '$gr00t_host' --port '$gr00t_port' $wrapper_arg" \
        > "$gr00t_logfile" 2>&1 < /dev/null &
    echo $! > "$gr00t_pidfile"
    echo "[INFO] pid=$(cat "$gr00t_pidfile"), bind=$gr00t_host:$gr00t_port"
    local wait_s="${GR00T_SERVER_WAIT_S:-300}" i
    for i in $(seq 1 "$wait_s"); do
        port_listening "$gr00t_port" && { echo "[INFO] GR00T server listening on :$gr00t_port"; return 0; }
        sleep 1
    done
    echo "[ERROR] GR00T server not ready after ${wait_s}s; last log:" >&2; tail -n 30 "$gr00t_logfile" >&2 || true
    return 1
}
_stop_gr00t_engine() {
    if [ -f "$gr00t_pidfile" ]; then
        local pid; pid=$(cat "$gr00t_pidfile")
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        sleep 1; kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        rm -f "$gr00t_pidfile"
    fi
    pkill -9 -f "run_gr00t_server.py.*--port[= ]*$gr00t_port" 2>/dev/null || true
    echo "[INFO] stopped GR00T server on :$gr00t_port"
}
start_gr00t_n16() { _start_gr00t_engine "$ROOT_DIR/../dependencies/Isaac-GR00T-N1.6" "${1:-hi-space/GR00T-N1.6-3B-Pick-Orange}"; }
stop_gr00t_n16()  { _stop_gr00t_engine; }
start_gr00t_n17() { _start_gr00t_engine "$ROOT_DIR/../dependencies/Isaac-GR00T"      "${1:-hi-space/GR00T-N1.7-3B-Pick-Orange}"; }
stop_gr00t_n17()  { _stop_gr00t_engine; }

# -------- LeRobot async :8080 (self-contained, on the conda lerobot env) --------
lerobot_host="${LEROBOT_HOST:-0.0.0.0}"
lerobot_port="${LEROBOT_PORT:-8080}"
lerobot_pidfile="$LOG_DIR/lerobot_server.pid"
lerobot_logfile="$LOG_DIR/lerobot_server.log"

start_lerobot() {
    if port_listening "$lerobot_port"; then echo "[INFO] LeRobot server already on :$lerobot_port"; return 0; fi
    local py="${LEROBOT_PYTHON:-}"
    [ -z "$py" ] && command -v conda >/dev/null 2>&1 && py="$(conda info --base)/envs/lerobot/bin/python"
    { [ -n "$py" ] && [ -x "$py" ]; } || { echo "[ERROR] lerobot python not found: ${py:-<unset>} (set LEROBOT_PYTHON)" >&2; exit 1; }
    local detach=(); command -v setsid >/dev/null 2>&1 && detach=(setsid)
    nohup "${detach[@]}" bash -lc "'$py' -m lerobot.async_inference.policy_server --host '$lerobot_host' --port '$lerobot_port'" \
        > "$lerobot_logfile" 2>&1 < /dev/null &
    echo $! > "$lerobot_pidfile"
    echo "[INFO] LeRobot server pid=$(cat "$lerobot_pidfile"), bind=$lerobot_host:$lerobot_port"
    local i; for i in $(seq 1 40); do port_listening "$lerobot_port" && { echo "[INFO] LeRobot listening on :$lerobot_port"; return 0; }; sleep 0.5; done
    echo "[ERROR] LeRobot server failed; last log:" >&2; tail -n 40 "$lerobot_logfile" >&2 || true; return 1
}
stop_lerobot() {
    if [ -f "$lerobot_pidfile" ]; then
        local pid; pid=$(cat "$lerobot_pidfile")
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        sleep 1; kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        rm -f "$lerobot_pidfile"
    fi
    pkill -9 -f "lerobot.async_inference.policy_server.*--port[= ]*$lerobot_port" 2>/dev/null || true
    echo "[INFO] stopped LeRobot server on :$lerobot_port"
}

case "$ACTION:$BACKEND" in
    start:gr00t-n15) start_gr00t_n15 "${3:-}" ;;
    stop:gr00t-n15)  stop_gr00t_n15 ;;
    start:gr00t-n16) start_gr00t_n16 "${3:-}" ;;
    stop:gr00t-n16)  stop_gr00t_n16 ;;
    start:gr00t-n17) start_gr00t_n17 "${3:-}" ;;
    stop:gr00t-n17)  stop_gr00t_n17 ;;
    start:lerobot)   start_lerobot ;;
    stop:lerobot)    stop_lerobot ;;
esac
