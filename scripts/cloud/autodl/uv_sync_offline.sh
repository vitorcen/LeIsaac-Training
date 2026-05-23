#!/usr/bin/env bash
# Run `uv sync` using a pre-fetched local cache from prefetch_uv_cache.sh.
# Works in AutoDL NO-CARD mode (no nvidia-smi needed because no wheel builds happen —
# everything is installed from cached .whl files).
#
# Usage:
#   bash uv_sync_offline.sh /root/autodl-tmp                    # cache root after tar extract
#
# Inputs:
#   - $1 = cache root dir containing archive-v0/ builds-v0/ wheels-v3/ sdists-v9/
#          and repo_meta/uv.lock + repo_meta/pyproject.toml
#   - or set CACHE_ROOT env var

set -euo pipefail

CACHE_ROOT="${1:-${CACHE_ROOT:-/root/autodl-tmp}}"
REPO_DIR="${REPO_DIR:-/root/autodl-tmp/isaaclab-experience}"
GR00T_DIR="$REPO_DIR/dependencies/Isaac-GR00T"

if [[ ! -d "$GR00T_DIR" ]]; then
    echo "ERROR: $GR00T_DIR not found (run prep_repo.sh first)" >&2; exit 1
fi
if [[ ! -d "$CACHE_ROOT/archive-v0" ]] || [[ ! -d "$CACHE_ROOT/wheels-v3" ]]; then
    echo "ERROR: $CACHE_ROOT missing archive-v0/ + wheels-v3/" >&2
    echo "  did you untar the prefetched bundle? expected layout:" >&2
    echo "    $CACHE_ROOT/archive-v0/" >&2
    echo "    $CACHE_ROOT/builds-v0/" >&2
    echo "    $CACHE_ROOT/wheels-v3/" >&2
    echo "    $CACHE_ROOT/sdists-v9/" >&2
    echo "    $CACHE_ROOT/repo_meta/uv.lock" >&2
    exit 1
fi

# Optional: sync uv.lock from bundle to repo (in case lockfile evolved on local)
if [[ -f "$CACHE_ROOT/repo_meta/uv.lock" ]]; then
    LOCAL_HASH=$(sha256sum "$GR00T_DIR/uv.lock" | awk '{print $1}')
    BUNDLE_HASH=$(sha256sum "$CACHE_ROOT/repo_meta/uv.lock" | awk '{print $1}')
    if [[ "$LOCAL_HASH" != "$BUNDLE_HASH" ]]; then
        echo "[uv_sync_offline] uv.lock differs between repo and bundle:"
        echo "  repo:   $LOCAL_HASH"
        echo "  bundle: $BUNDLE_HASH"
        echo "  → using bundle's lockfile (overwriting repo's)"
        cp "$CACHE_ROOT/repo_meta/uv.lock" "$GR00T_DIR/uv.lock"
    fi
fi

export PATH=/root/.local/bin:/root/miniconda3/bin:$PATH
export UV_CACHE_DIR="$CACHE_ROOT"

cd "$GR00T_DIR"

echo "[uv_sync_offline] using UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[uv_sync_offline] running uv sync --offline (no network needed)"
echo "  expected: 2-5 min, install phase only (linking cached wheels to .venv)"

# --offline means uv won't even try to hit the network — fails fast if cache is missing pieces
time uv sync --offline 2>&1 | tail -30

echo
echo "[uv_sync_offline] verifying torch + flash_attn importable"
.venv/bin/python -c "
import torch
print(f'torch={torch.__version__}  cuda_available={torch.cuda.is_available()}')
try:
    import flash_attn; print(f'flash_attn={flash_attn.__version__}')
except ImportError as e: print(f'flash_attn import FAILED: {e}')
try:
    import tensorrt; print(f'tensorrt={tensorrt.__version__}')
except ImportError as e: print(f'tensorrt import FAILED: {e}')
"

echo "[uv_sync_offline] DONE"
du -sh .venv
