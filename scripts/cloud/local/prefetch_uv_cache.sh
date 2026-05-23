#!/usr/bin/env bash
# Pre-fetch all Isaac-GR00T wheels into a portable tarball — runs on LOCAL DEV BOX.
# Goal: avoid AutoDL's throttled academic proxy (1 MB/s for big wheels), do the
# heavy lifting on local FTTH (10-100 MB/s), then scp the bundle to AutoDL.
#
# Approach:
#   1. Ensure local Isaac-GR00T is uv-synced (populates ~/.cache/uv with all wheels)
#   2. Bundle uv cache + lockfile + pyproject.toml into a tar.gz
#   3. Report bundle size + scp suggested next step
#
# Constraints:
#   - Local machine must be Linux x86_64 same arch as AutoDL target
#   - Local needs ~50 GB free (uv cache during sync) + ~15 GB for tarball
#   - Local does NOT need a GPU; uv sync with tensorrt-cu12 wheel_stub WILL invoke
#     nvidia-smi, so if you don't have an NVIDIA driver, tensorrt-cu12 build fails.
#     Workaround: install nvidia-headless driver locally OR fake nvidia-smi via PATH shim.
#
# Usage:
#   bash prefetch_uv_cache.sh
#   # → produces /tmp/n17_uv_cache_bundle_<date>.tar.gz
#   # next step printed at end

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
GR00T_DIR="${GR00T_DIR:-$REPO_ROOT/dependencies/Isaac-GR00T}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
OUT="${OUT:-/tmp/n17_uv_cache_bundle_$(date +%Y%m%d_%H%M%S).tar.gz}"

if [[ ! -d "$GR00T_DIR" ]]; then
    echo "ERROR: $GR00T_DIR not found (set GR00T_DIR env)" >&2; exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not installed locally. curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<EOF
WARN: nvidia-smi not in PATH. tensorrt-cu12 wheel build invokes it and may fail.
       Workaround: install nvidia-utils-535 (or any nvidia driver pkg) locally,
       or create a fake shim that returns sensible compute capability:
       cat >/usr/local/bin/nvidia-smi <<'STUB'
       #!/bin/sh
       case "\$*" in
         *--query-gpu=compute_cap*) echo "9.0" ;;
         *) echo "NVIDIA-SMI shim" ;;
       esac
       STUB
       chmod +x /usr/local/bin/nvidia-smi
EOF
fi

echo "=== step 0: ensure aliyun PyPI is uv's default index (~/.config/uv/uv.toml) ==="
# 必要：默认 PyPI files.pythonhosted.org 从 CN TLS handshake EOF，影响 metadata 解析。
# 让 uv 把 aliyun 当 default index 即可。这是用户级配置，不影响其他项目。
mkdir -p "$HOME/.config/uv"
if ! grep -q "aliyun" "$HOME/.config/uv/uv.toml" 2>/dev/null; then
    cat > "$HOME/.config/uv/uv.toml" <<'EOF'
[[index]]
name = "aliyun-pypi"
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
EOF
    echo "  wrote $HOME/.config/uv/uv.toml (aliyun PyPI as default)"
else
    echo "  $HOME/.config/uv/uv.toml already has aliyun"
fi

echo "=== step 1: uv sync locally (~10-30 min, populates $UV_CACHE_DIR) ==="
cd "$GR00T_DIR"
if [[ ! -f "$GR00T_DIR/.venv/lib/python3.10/site-packages/torch/__init__.py" ]]; then
    # CRITICAL: PyTorch CDN (download.pytorch.org / download-r2.pytorch.org) is unreachable
    # from China without VPN (TLS handshake EOF). Patch pyproject.toml to use direct aliyun
    # wheel URLs instead of the pytorch-cu128 index (aliyun is a flat dir, not a PEP 503 index,
    # so we can't just substitute the URL — must use {url=...} sources directly).
    # We keep a .orig backup and restore it after uv sync to avoid polluting git status.
    if grep -q 'index = "pytorch-cu128"' pyproject.toml 2>/dev/null; then
        echo "  patching pyproject.toml: pytorch-cu128 index → direct aliyun wheel URLs"
        cp pyproject.toml pyproject.toml.uv_sync_orig
        python3 - <<'PYEOF'
import re
p = 'pyproject.toml'
src = open(p).read()
# Replace torch / torchvision / triton blocks with direct URL or PyPI fallback
torch_block = '''torch = [
    { url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.10'" },
]
torchvision = [
    { url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.10'" },
]
# triton: PyPI default (drop pytorch-cu128 index since aliyun is not PEP 503)'''
src = re.sub(
    r'torch = \[\s*\{ index = "pytorch-cu128".*?\}\s*\]\s*\n'
    r'torchvision = \[\s*\{ index = "pytorch-cu128".*?\}\s*\]\s*\n'
    r'triton = \[\s*\{ index = "pytorch-cu128".*?\}\s*\]',
    torch_block, src, flags=re.DOTALL)
open(p, 'w').write(src)
PYEOF
        _PATCHED=1
    fi
    UV_CACHE_DIR="$UV_CACHE_DIR" uv sync
    _RC=$?
    if [[ "${_PATCHED:-0}" == "1" ]]; then
        echo "  restoring original pyproject.toml"
        mv pyproject.toml.uv_sync_orig pyproject.toml
    fi
    [[ $_RC -ne 0 ]] && exit $_RC
else
    echo "  local .venv already has torch installed, skipping sync"
fi

echo
echo "=== step 2: figure out which cache subdirs we need ==="
# uv cache layout (v0.7+):
#   archive-v0/  ← extracted wheels (used as install source)
#   builds-v0/   ← sdist→wheel builds (tensorrt etc.)
#   wheels-v3/   ← downloaded .whl files
#   sdists-v9/   ← downloaded sdists (smaller)
SUBDIRS=(archive-v0 builds-v0 wheels-v3 sdists-v9)
TOTAL_KB=0
for s in "${SUBDIRS[@]}"; do
    if [[ -d "$UV_CACHE_DIR/$s" ]]; then
        SZ=$(du -sk "$UV_CACHE_DIR/$s" | cut -f1)
        echo "  $s: $((SZ/1024)) MB"
        TOTAL_KB=$((TOTAL_KB + SZ))
    fi
done
echo "  uv cache total: $((TOTAL_KB/1024/1024)) GB"
echo "  ⚠️ this includes ALL local uv projects; bundle will be similarly sized"
echo "  → for selective bundling, see step 3"

echo
echo "=== step 3: bundle uv cache + lockfile + pyproject ==="
# Include the lockfile and pyproject so the target machine knows what to install.
# Tar with relative paths so it can be extracted anywhere.
EXISTING_SUBDIRS=()
for s in "${SUBDIRS[@]}"; do
    [[ -d "$UV_CACHE_DIR/$s" ]] && EXISTING_SUBDIRS+=("$s")
done

# build relative file list for tar
TAR_TMP=$(mktemp -d)
mkdir -p "$TAR_TMP/repo_meta"
cp "$GR00T_DIR/uv.lock" "$TAR_TMP/repo_meta/uv.lock"
cp "$GR00T_DIR/pyproject.toml" "$TAR_TMP/repo_meta/pyproject.toml"

echo "  creating $OUT (this takes a few minutes for 5-10 GB)..."
time tar czf "$OUT" \
    -C "$UV_CACHE_DIR" "${EXISTING_SUBDIRS[@]}" \
    -C "$TAR_TMP" repo_meta

rm -rf "$TAR_TMP"

BUNDLE_SIZE=$(du -h "$OUT" | cut -f1)
echo
echo "=== DONE ==="
echo "bundle:        $OUT"
echo "bundle size:   $BUNDLE_SIZE"
echo
cat <<EOF
=== next steps (scp + offline install on AutoDL) ===
  # on local box:
  SSHPASS=xxxxx sshpass -e scp -P <port> $OUT root@<host>:/root/autodl-tmp/

  # on AutoDL (no-card mode is fine — install never invokes nvidia-smi):
  cd /root/autodl-tmp
  tar xzf $(basename "$OUT") -C /root/autodl-tmp/
  bash isaaclab-experience/LeIsaac/scripts/cloud/autodl/uv_sync_offline.sh \\
       /root/autodl-tmp                                                     # cache root
EOF
