#!/usr/bin/env bash
# Auto-resume watchdog for FastWAM QLoRA training.
#
# The bnb 4bit + flash-attn + peft + grad-ckpt combination crashes the
# Python process at random step counts (no Python-level traceback, just
# SIGKILL / SIGSEGV).  This script keeps relaunching the trainer; each
# relaunch resumes from the latest checkpoint in the phase's output dir.
#
# Usage:
#   watchdog.sh <phase_name> <target_steps> [<initial_resume_path>]
#
# Examples:
#   watchdog.sh phase1 2000
#   watchdog.sh phase2 4000 \
#       $HOME/work/fastwam-repo/runs/train/fastwam_qlora_pickorange_5phase/phase1/checkpoints/state/step_002000

set -uo pipefail

PHASE_NAME="${1:-phase1}"
TARGET_STEPS="${2:-2000}"
INITIAL_RESUME="${3:-none}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
# Save policy: keep last KEEP_LAST_N ckpts within the *current* phase so the
# disk doesn't explode at 4.5GB × 200 saves = 900GB.  Latest = resume target,
# +2 older = fallbacks if the latest one was corrupted by SIGSEGV mid-save.
KEEP_LAST_N="${KEEP_LAST_N:-3}"

REPO_ROOT="${FASTWAM_REPO:-$HOME/work/fastwam-repo}"
LEISAAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CONFIG_DIR="${LEISAAC_ROOT}/LeIsaac/scripts/finetune/fastwam_qlora/configs"
LOG_DIR="${LEISAAC_ROOT}/logs"
LOG_FILE="${LOG_DIR}/fastwam_qlora_${PHASE_NAME}.log"
OUTPUT_DIR_REL="./runs/train/fastwam_qlora_pickorange_5phase/${PHASE_NAME}"
OUTPUT_DIR_ABS="${REPO_ROOT}/runs/train/fastwam_qlora_pickorange_5phase/${PHASE_NAME}"

mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"
source $(conda info --base)/etc/profile.d/conda.sh
# fastwam_stable (torch 2.5.1+cu124 / bnb 0.45.5 / flash-attn 2.7.4.post1)
# is the stable production env.  Override via CONDA_ENV=... to swap.
conda activate "${CONDA_ENV:-fastwam_stable}"

export PYTHONPATH="${LEISAAC_ROOT}/LeIsaac/scripts/finetune:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HYDRA_FULL_ERROR=1

prune_old_ckpts() {
    # Keep the last KEEP_LAST_N step_* directories; remove the rest.
    # Also remove any size-truncated (mid-save SIGSEGV) safetensors.
    local d="${OUTPUT_DIR_ABS}/checkpoints/state"
    [[ -d "$d" ]] || return 0
    # Drop corrupted ckpts: empty dir (SIGSEGV killed BEFORE first byte) OR
    # truncated model.safetensors (< 4GB; full file is ~4.6GB).
    for c in "$d"/step_*; do
        [[ -d "$c" ]] || continue
        local mf="$c/model.safetensors"
        if [[ ! -f "$mf" ]]; then
            echo "[watchdog]   prune empty $(basename "$c")" | tee -a "${LOG_FILE}"
            rm -rf "$c"
            continue
        fi
        local sz; sz=$(stat -c %s "$mf" 2>/dev/null || echo 0)
        if [[ "$sz" -lt 4000000000 ]]; then
            echo "[watchdog]   prune corrupt $(basename "$c") (size=$sz)" | tee -a "${LOG_FILE}"
            rm -rf "$c"
        fi
    done
    # Keep last N by step number.
    mapfile -t all < <(ls -1d "$d"/step_* 2>/dev/null | sort -V)
    local total=${#all[@]}
    if [[ "$total" -gt "$KEEP_LAST_N" ]]; then
        local drop=$((total - KEEP_LAST_N))
        for ((i=0; i<drop; i++)); do
            echo "[watchdog]   prune old $(basename "${all[$i]}")" | tee -a "${LOG_FILE}"
            rm -rf "${all[$i]}"
        done
    fi
}

echo "[watchdog] $(date '+%F %T') phase=${PHASE_NAME} target_steps=${TARGET_STEPS} initial_resume=${INITIAL_RESUME} keep_last_n=${KEEP_LAST_N}" | tee -a "${LOG_FILE}"

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    # Prune before each attempt: drop truncated ckpts (SIGSEGV mid-save) and
    # cap retention to KEEP_LAST_N.  This is the only place we trim — so a
    # crashed attempt's partial write gets cleaned before the next resume.
    prune_old_ckpts

    # Find latest ckpt under this phase's output dir.
    LATEST_CKPT=""
    if [[ -d "${OUTPUT_DIR_ABS}/checkpoints/state" ]]; then
        LATEST_CKPT="$(ls -1d "${OUTPUT_DIR_ABS}"/checkpoints/state/step_* 2>/dev/null | sort -V | tail -1)"
    fi

    if [[ -n "${LATEST_CKPT}" ]]; then
        RESUME_ARG="resume=${LATEST_CKPT}"
        echo "[watchdog] $(date '+%F %T') attempt=${attempt}/${MAX_ATTEMPTS} resume=${LATEST_CKPT}" | tee -a "${LOG_FILE}"
    elif [[ "${INITIAL_RESUME}" != "none" && -d "${INITIAL_RESUME}" ]]; then
        RESUME_ARG="resume=${INITIAL_RESUME}"
        echo "[watchdog] $(date '+%F %T') attempt=${attempt}/${MAX_ATTEMPTS} initial-resume=${INITIAL_RESUME}" | tee -a "${LOG_FILE}"
    else
        RESUME_ARG=""
        echo "[watchdog] $(date '+%F %T') attempt=${attempt}/${MAX_ATTEMPTS} fresh-start" | tee -a "${LOG_FILE}"
    fi

    python -u -m fastwam_qlora.train \
        --config-path "${CONFIG_DIR}" \
        --config-name train \
        max_steps="${TARGET_STEPS}" \
        save_every=50 \
        eval_every=999999 \
        log_every=20 \
        output_dir="${OUTPUT_DIR_REL}" \
        ${RESUME_ARG} \
        >> "${LOG_FILE}" 2>&1
    PY_EXIT=$?

    # Check completion: either "max_steps reached" log message OR
    # clean exit (PY_EXIT=0) with the resume ckpt's step matching target
    # (the case where global_step already >= max_steps so the loop runs 0 times).
    if tail -30 "${LOG_FILE}" | grep -qE "max_steps reached step=${TARGET_STEPS}"; then
        echo "[watchdog] $(date '+%F %T') ${PHASE_NAME} COMPLETED at attempt=${attempt}" | tee -a "${LOG_FILE}"
        exit 0
    fi
    if [[ ${PY_EXIT} -eq 0 ]]; then
        LATEST_STEP_DIR="$(ls -1d "${OUTPUT_DIR_ABS}"/checkpoints/state/step_* 2>/dev/null | sort -V | tail -1)"
        if [[ -n "${LATEST_STEP_DIR}" ]]; then
            CUR_STEP="$(basename "${LATEST_STEP_DIR}" | sed 's/^step_0*//')"
            CUR_STEP="${CUR_STEP:-0}"
            if [[ "${CUR_STEP}" -ge "${TARGET_STEPS}" ]]; then
                echo "[watchdog] $(date '+%F %T') ${PHASE_NAME} COMPLETED (fast-skip, step=${CUR_STEP} >= ${TARGET_STEPS}) at attempt=${attempt}" | tee -a "${LOG_FILE}"
                exit 0
            fi
        fi
    fi

    echo "[watchdog] $(date '+%F %T') attempt=${attempt} crashed (exit=${PY_EXIT}), will retry" | tee -a "${LOG_FILE}"
    sleep 5
done

echo "[watchdog] $(date '+%F %T') ${PHASE_NAME} FAILED after ${MAX_ATTEMPTS} attempts" | tee -a "${LOG_FILE}"
exit 1
