#!/usr/bin/env bash
# Generic LeRobot policy fine-tune on a local LeRobot dataset.
#
# Use this for SmolVLA / ACT / pi0 / DreamZero / etc. — any model that ships as
# a LeRobot policy and can be loaded via `--policy.path=<HF_REPO_OR_LOCAL_DIR>`,
# OR for from-scratch training of in-tree policies via `--policy.type=<NAME>`
# (e.g. diffusion, multi_task_dit, act, vqbet).
#
# Usage A — fine-tune from a pretrained base (e.g. SmolVLA):
#
#   BASE_MODEL=lerobot/smolvla_base \
#   DATASET_REPO_ID=LightwheelAI/leisaac-pick-orange \
#   OUTPUT_NAME=smolvla2-leisaac-pick-orange \
#   STEPS=20000 BATCH_SIZE=64 \
#   bash scripts/training/lerobot_finetune.sh
#
# Usage B — from-scratch training (Diffusion Policy / DiT / ACT / ...):
#
#   POLICY_TYPE=diffusion \
#   DATASET_REPO_ID=LightwheelAI/leisaac-pick-orange \
#   OUTPUT_NAME=dp-leisaac-pick-orange \
#   STEPS=100000 BATCH_SIZE=64 \
#   bash scripts/training/lerobot_finetune.sh
#
# Common knobs:
#   BASE_MODEL          HF repo / local dir for `--policy.path`     (mutually exclusive w/ POLICY_TYPE)
#   POLICY_TYPE         In-tree policy class for `--policy.type`    (mutually exclusive w/ BASE_MODEL)
#   DATASET_REPO_ID     HF dataset repo for `--dataset.repo_id`    (default LightwheelAI/leisaac-pick-orange)
#   DATASET_ROOT        Local v3.0 path (auto: datasets/raw/<basename>)
#   OUTPUT_NAME         Output dir name under outputs/             (auto: <base>-<dataset>)
#   STEPS               Total training steps                       (default 20000)
#   BATCH_SIZE          Per-device batch                           (default 64)
#   NUM_WORKERS         Dataloader workers                         (default 4)
#   SAVE_FREQ           Checkpoint save interval                   (default 5000)
#   RENAME_MAP          JSON dict (sim→policy keys). Empty if natural-key match.
#   EXTRA_ARGS          Free-form extra flags appended to lerobot-train
#   CONDA_ENV           Conda env name                             (default lerobot)
#   AUTO_EVAL           1=spawn eval_watcher.sh per-ckpt sanity eval (default 1)
#                       Set 0 to disable. See LeIsaac/CLAUDE.md "incremental
#                       sanity eval" rule + eval_watcher.sh for env knobs.
#   EVAL_HORIZON        policy_action_horizon for watcher (auto from POLICY_TYPE)
#
# Behavior:
#   - Refuses to start if DATASET_ROOT not present (hints to run download.sh).
#   - Refuses to start if dataset codebase_version != v3.0 (hints convert_to_v30.sh).
#   - Logs full stdout/stderr to outputs/<OUTPUT_NAME>/train.log.
#   - Final ckpt: outputs/<OUTPUT_NAME>/checkpoints/last/pretrained_model
#   - If AUTO_EVAL=1 (default): spawns eval_watcher.sh in background; watcher
#     polls $OUTPUT_DIR/checkpoints/ and runs 3-round 60s quick eval per ckpt.
#     Wrapper polls $OUTPUT_DIR/.eval_abort and SIGTERMs lerobot-train if set
#     (3 consecutive 0-orange or stuck slices → don't burn N more hours).

set -euo pipefail

# -------- defaults --------
BASE_MODEL="${BASE_MODEL:-}"
POLICY_TYPE="${POLICY_TYPE:-}"
# Back-compat: if neither is set, fall back to the historical SmolVLA default.
if [[ -z "${BASE_MODEL}" && -z "${POLICY_TYPE}" ]]; then
    BASE_MODEL="lerobot/smolvla_base"
fi
if [[ -n "${BASE_MODEL}" && -n "${POLICY_TYPE}" ]]; then
    echo "[finetune] ERROR: set BASE_MODEL *or* POLICY_TYPE, not both." >&2
    exit 1
fi
DATASET_REPO_ID="${DATASET_REPO_ID:-LightwheelAI/leisaac-pick-orange}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
RENAME_MAP="${RENAME_MAP:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
AUTO_EVAL="${AUTO_EVAL:-1}"
EVAL_HORIZON="${EVAL_HORIZON:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_BASENAME="$(basename "${DATASET_REPO_ID}")"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/raw/${DATASET_BASENAME}}"

# default OUTPUT_NAME = <base_short_or_policy_type>-<dataset_basename>
if [[ -n "${BASE_MODEL}" ]]; then
    BASE_SHORT="$(basename "${BASE_MODEL}" | sed 's/[^a-zA-Z0-9._-]/-/g')"
else
    BASE_SHORT="${POLICY_TYPE}"
fi
OUTPUT_NAME="${OUTPUT_NAME:-${BASE_SHORT}-${DATASET_BASENAME}}"
OUTPUT_DIR="${REPO_ROOT}/outputs/${OUTPUT_NAME}"

# -------- preflight --------
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "[finetune] dataset not found at ${DATASET_ROOT}" >&2
    echo "[finetune] hint: bash datasets/download.sh ${DATASET_REPO_ID}" >&2
    exit 1
fi

CURRENT_VERSION="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version',''))" "${DATASET_ROOT}/meta/info.json")"
if [[ "${CURRENT_VERSION}" != "v3.0" ]]; then
    echo "[finetune] dataset is ${CURRENT_VERSION}, lerobot ≥0.5 requires v3.0" >&2
    echo "[finetune] hint: bash datasets/convert_to_v30.sh ${DATASET_REPO_ID}" >&2
    exit 1
fi

# lerobot-train refuses to start if output_dir exists, so we cannot pre-mkdir it.
# Logs go to a sibling path that we own.
LOG_DIR="${REPO_ROOT}/outputs/.logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${OUTPUT_NAME}.log"

cat <<EOF | tee "${LOG_FILE}"
[finetune] base model:     ${BASE_MODEL:-(none — from-scratch)}
[finetune] policy type:    ${POLICY_TYPE:-(via base model)}
[finetune] dataset repo:   ${DATASET_REPO_ID}
[finetune] dataset root:   ${DATASET_ROOT}  (${CURRENT_VERSION})
[finetune] output dir:     ${OUTPUT_DIR}
[finetune] steps:          ${STEPS}
[finetune] batch size:     ${BATCH_SIZE}
[finetune] num workers:    ${NUM_WORKERS}
[finetune] save freq:      ${SAVE_FREQ}
[finetune] rename_map:     ${RENAME_MAP:-(none)}
[finetune] extra args:     ${EXTRA_ARGS:-(none)}
[finetune] conda env:      ${CONDA_ENV}
EOF

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "[finetune] ERROR: ${OUTPUT_DIR} already exists; lerobot-train refuses to overwrite." >&2
    echo "[finetune] either delete it, or rerun with OUTPUT_NAME=<new>" >&2
    exit 1
fi

# -------- compose & run --------
ARGS=(
    --policy.push_to_hub=false
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --output_dir="${OUTPUT_DIR}"
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --num_workers="${NUM_WORKERS}"
    --policy.device=cuda
    --wandb.enable=false
)
if [[ -n "${BASE_MODEL}" ]]; then
    ARGS=(--policy.path="${BASE_MODEL}" "${ARGS[@]}")
else
    ARGS=(--policy.type="${POLICY_TYPE}" "${ARGS[@]}")
fi
if [[ -n "${RENAME_MAP}" ]]; then
    ARGS+=(--rename_map="${RENAME_MAP}")
fi
# shellcheck disable=SC2206
EXTRA_ARGS_ARR=(${EXTRA_ARGS})
ARGS+=("${EXTRA_ARGS_ARR[@]}")

echo "[finetune] launching lerobot-train; full log: ${LOG_FILE}"
set -o pipefail
conda run -n "${CONDA_ENV}" --no-capture-output lerobot-train "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}" &
TRAIN_PID=$!

# -------- AUTO_EVAL=1: spawn per-ckpt sanity-eval watcher (LeIsaac/CLAUDE.md rule) --------
WATCHER_PID=""
if [[ "${AUTO_EVAL}" == "1" ]]; then
    # Infer inference-side policy_type slug
    if [[ -n "${POLICY_TYPE:-}" ]]; then
        case "${POLICY_TYPE}" in
            diffusion)      EVAL_POLICY_TYPE="lerobot-diffusion" ;;
            act)            EVAL_POLICY_TYPE="lerobot-act" ;;
            smolvla)        EVAL_POLICY_TYPE="lerobot-smolvla" ;;
            *)              EVAL_POLICY_TYPE="lerobot-${POLICY_TYPE}" ;;
        esac
    elif [[ -n "${BASE_MODEL:-}" ]]; then
        case "${BASE_MODEL,,}" in
            *smolvla*)      EVAL_POLICY_TYPE="lerobot-smolvla" ;;
            *)              EVAL_POLICY_TYPE="lerobot-act" ;;  # best-effort
        esac
    else
        EVAL_POLICY_TYPE="lerobot-act"
    fi
    OUTPUT_DIR="${OUTPUT_DIR}" POLICY_TYPE="${EVAL_POLICY_TYPE}" \
        EVAL_HORIZON="${EVAL_HORIZON}" \
        nohup bash "${REPO_ROOT}/scripts/training/eval_watcher.sh" \
        > "${OUTPUT_DIR}/auto_eval.log" 2>&1 &
    WATCHER_PID=$!
    disown "${WATCHER_PID}" 2>/dev/null || true
    echo "[finetune] AUTO_EVAL watcher PID=${WATCHER_PID} → ${OUTPUT_DIR}/auto_eval.log"
    echo "[finetune] watcher polls ${OUTPUT_DIR}/checkpoints/ and SIGTERMs training if 3 consecutive 0-orange slices"
fi

# -------- monitor abort marker + reap training --------
ABORT_MARKER="${OUTPUT_DIR}/.eval_abort"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    if [[ -f "${ABORT_MARKER}" ]]; then
        echo "[finetune] ABORT: eval_watcher signaled (3 consecutive 0-orange / stuck slices)" >&2
        kill -TERM "${TRAIN_PID}" 2>/dev/null || true
        sleep 10
        kill -9 "${TRAIN_PID}" 2>/dev/null || true
        break
    fi
    sleep 30
done
wait "${TRAIN_PID}"
TRAIN_RC=$?

if [[ -n "${WATCHER_PID}" ]]; then
    kill -TERM "${WATCHER_PID}" 2>/dev/null || true
fi

if [[ ${TRAIN_RC} -ne 0 ]]; then
    echo "[finetune] FAILED with rc=${TRAIN_RC}; see ${LOG_FILE}" >&2
    exit "${TRAIN_RC}"
fi

echo "[finetune] done; final ckpt at ${OUTPUT_DIR}/checkpoints/last/pretrained_model"
