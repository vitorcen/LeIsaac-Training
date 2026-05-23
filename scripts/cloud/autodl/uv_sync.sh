#!/usr/bin/env bash
# Run `uv sync` for Isaac-GR00T. MUST be done in GPU mode — tensorrt-cu12-libs build
# requires nvidia-smi access which is permission-denied in no-card mode.
#
# Pre-req: bootstrap.sh + prep_repo.sh already done.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/isaaclab-experience}"
GR00T_DIR="$REPO_DIR/dependencies/Isaac-GR00T"

if [[ ! -d "$GR00T_DIR" ]]; then
    echo "[uv_sync] ERROR: $GR00T_DIR not found — run prep_repo.sh first" >&2
    exit 1
fi

# verify GPU mode: nvidia-smi should be accessible
if ! nvidia-smi >/dev/null 2>&1; then
    echo "[uv_sync] ERROR: nvidia-smi inaccessible — you're in no-card mode" >&2
    echo "  tensorrt-cu12-libs build will fail. Reboot in GPU mode first." >&2
    exit 2
fi

export PATH=/root/.local/bin:/root/miniconda3/bin:$PATH
source /etc/network_turbo >/dev/null 2>&1
cd "$GR00T_DIR"

echo "[uv_sync] starting uv sync (no --extra=gpu, that extra doesn't exist)"
echo "  expected: ~10 min for ~10 GB download incl. torch 2.7.1 cu128 + flash-attn 2.7.4.post1"
time uv sync 2>&1 | tail -20

echo "[uv_sync] DONE"
du -sh "$GR00T_DIR/.venv"
"$GR00T_DIR/.venv/bin/python" -c "
import torch
print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}')
try:
    import flash_attn; print(f'flash_attn={flash_attn.__version__}')
except ImportError: print('flash_attn NOT installed')
"
