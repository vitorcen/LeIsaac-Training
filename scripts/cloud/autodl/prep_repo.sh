#!/usr/bin/env bash
# Shallow-clone isaaclab-experience + selective submodules (LeIsaac + Isaac-GR00T).
# Skip the heavy ones (IsaacLab, IsaacSim, BEHAVIOR-1K, lerobot) — not needed for GR00T training.
#
# Env knobs:
#   REPO_URL          default https://github.com/vitorcen/isaaclab-experience
#   REPO_DIR          default /root/autodl-tmp/isaaclab-experience
#   GR00T_COMMIT      default 3df8b3825d67f755e69141446f4315f281b9b7e6
#   GR00T_URL         default https://github.com/NVIDIA/Isaac-GR00T.git

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/vitorcen/isaaclab-experience}"
REPO_DIR="${REPO_DIR:-/root/autodl-tmp/isaaclab-experience}"
GR00T_URL="${GR00T_URL:-https://github.com/NVIDIA/Isaac-GR00T.git}"
GR00T_COMMIT="${GR00T_COMMIT:-3df8b3825d67f755e69141446f4315f281b9b7e6}"

source /etc/network_turbo >/dev/null 2>&1

echo "[prep_repo] step 1: shallow clone main repo"
if [[ ! -d "$REPO_DIR" ]]; then
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "[prep_repo] step 2: init LeIsaac submodule (shallow)"
git submodule update --init --depth 1 LeIsaac 2>&1 | tail -3

echo "[prep_repo] step 3: clone Isaac-GR00T separately at pinned commit"
if [[ ! -d "$REPO_DIR/dependencies/Isaac-GR00T" ]]; then
    mkdir -p "$REPO_DIR/dependencies"
    git clone "$GR00T_URL" "$REPO_DIR/dependencies/Isaac-GR00T"
fi
cd "$REPO_DIR/dependencies/Isaac-GR00T"
git checkout "$GR00T_COMMIT" 2>&1 | tail -3

echo "[prep_repo] step 4: git lfs pull (Isaac-GR00T wheels)"
git lfs pull 2>&1 | tail -3
echo "[prep_repo] verify LFS wheels expanded:"
find scripts/deployment -name "*.whl" -exec ls -la {} \; 2>&1 | head -5

echo "[prep_repo] DONE"
du -sh "$REPO_DIR"
