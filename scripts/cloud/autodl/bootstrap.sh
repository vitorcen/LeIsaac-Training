#!/usr/bin/env bash
# AutoDL one-time bootstrap: install uv + git-lfs + hf cli, persist env to .bashrc.
# Idempotent — safe to re-run.
#
# Required: HF_TOKEN env var passed in (we don't hard-code it).
#   HF_TOKEN=hf_xxxx bash bootstrap.sh

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[bootstrap] ERROR: HF_TOKEN env var required" >&2
    echo "  usage: HF_TOKEN=hf_xxxx bash bootstrap.sh" >&2
    exit 1
fi

echo "[bootstrap] step 1: apt deps (git-lfs, aria2 fallback, sshpass for scp loops)"
source /etc/network_turbo >/dev/null 2>&1
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git-lfs aria2 >/dev/null
git lfs install >/dev/null

echo "[bootstrap] step 2: install uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[bootstrap] step 3: persist env to .bashrc"
if ! grep -q 'AutoDL HF acceleration' "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<'EOF'

# --- AutoDL HF acceleration (added by LeIsaac/scripts/cloud/autodl/bootstrap.sh) ---
export HF_ENDPOINT=https://hf-mirror.com
export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=/root/autodl-tmp/hf_cache
export PATH=/root/.local/bin:/root/miniconda3/bin:$PATH
EOF
fi
mkdir -p /root/autodl-tmp/hf_cache

echo "[bootstrap] step 4: install hf cli + huggingface_hub"
/root/miniconda3/bin/pip install -q -U huggingface_hub 'huggingface_hub[hf_xet]' >/dev/null

echo "[bootstrap] step 5: HF login (both endpoints — gated repos need canonical)"
export PATH=/root/.local/bin:/root/miniconda3/bin:$PATH
HF_ENDPOINT=https://hf-mirror.com hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -2
HF_ENDPOINT=https://huggingface.co hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -2

echo "[bootstrap] step 6: verify"
HF_ENDPOINT=https://huggingface.co hf auth whoami 2>&1 | head -3
df -h /root/autodl-tmp | tail -1
echo "[bootstrap] DONE — source ~/.bashrc to pick up env in current shell"
