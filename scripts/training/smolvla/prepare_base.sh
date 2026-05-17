#!/usr/bin/env bash
# Make a local SmolVLA base copy with `input_features` and `empty_cameras`
# cleared, so a downstream lerobot-train run will auto-populate
# `input_features` from the dataset (natural keys, real resolution) instead
# of inheriting the base's placeholder `camera1/2/3 @ 256x256` schema.
#
# Why this matters: smolvla_base's saved config.json declares 3 placeholder
# image slots that get *merged* with any CLI override (draccus dict-merge
# semantics), so you cannot replace them via `--policy.input_features=...`.
# Cloning the base locally + stripping `input_features` is the only clean way
# to get a "schema-free" base for fine-tuning on a 2-camera SO-101 dataset.
#
# Usage:
#   bash scripts/training/smolvla/prepare_base.sh
#   # → outputs/.bases/smolvla_base_no_features/   (input_features = {})
#
# Then pass it to lerobot_finetune.sh:
#   BASE_MODEL=$REPO/outputs/.bases/smolvla_base_no_features  \
#   DATASET_REPO_ID=... \
#   bash scripts/training/lerobot_finetune.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_REPO="${BASE_REPO:-lerobot/smolvla_base}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/.bases/$(basename "${BASE_REPO}")_no_features}"
CONDA_ENV="${CONDA_ENV:-lerobot}"

echo "[prepare-base] source HF repo:  ${BASE_REPO}"
echo "[prepare-base] output local dir: ${OUT_DIR}"

if [[ -f "${OUT_DIR}/config.json" && "${FORCE:-0}" != "1" ]]; then
    echo "[prepare-base] already exists; FORCE=1 to rebuild"
    exit 0
fi

# Find the cached snapshot dir for BASE_REPO.
CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}/hub/models--${BASE_REPO//\//--}"
if [[ ! -d "${CACHE_DIR}" ]]; then
    echo "[prepare-base] HF cache miss for ${BASE_REPO}; downloading…" >&2
    conda run -n "${CONDA_ENV}" --no-capture-output \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('${BASE_REPO}')"
fi
SNAPSHOT="$(ls -d "${CACHE_DIR}"/snapshots/*/ 2>/dev/null | head -1)"
if [[ -z "${SNAPSHOT}" ]]; then
    echo "[prepare-base] could not locate snapshot under ${CACHE_DIR}" >&2
    exit 1
fi
echo "[prepare-base] snapshot:        ${SNAPSHOT}"

# Dereference symlinks so the new dir is self-contained (HF cache uses
# symlinks into ../blobs/<etag>).
rm -rf "${OUT_DIR}"
mkdir -p "$(dirname "${OUT_DIR}")"
cp -RL "${SNAPSHOT%/}" "${OUT_DIR}"

# Strip input_features + empty_cameras from config.json.
python - <<PY
import json
p = "${OUT_DIR}/config.json"
d = json.load(open(p))
before = sorted(d.get("input_features", {}).keys())
d["input_features"] = {}
d["empty_cameras"] = 0
json.dump(d, open(p, "w"), indent=2)
print(f"[prepare-base] cleared input_features (was: {before})")
print(f"[prepare-base] cleared empty_cameras  (set to 0)")
PY

echo "[prepare-base] done: ${OUT_DIR}"
