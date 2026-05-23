#!/usr/bin/env bash
# Sanity check after prep + download + uv_sync — before launching real training.

set -uo pipefail

# Ubuntu's default .bashrc returns early in non-interactive shells, so we can't
# source it to pick up our PATH additions. Set them explicitly instead.
export PATH="${HOME}/.local/bin:/root/miniconda3/bin:$PATH"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/isaaclab-experience}"
COSMOS_DIR="${COSMOS_DIR:-/root/autodl-tmp/cosmos_raw}"
DATASET_REPO="${DATASET_REPO:-LightwheelAI/leisaac-pick-orange}"

PASS=0
FAIL=0
check() {
    local desc="$1"; local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  ✓ $desc"
        PASS=$((PASS+1))
    else
        echo "  ✗ $desc   ← FAIL"
        FAIL=$((FAIL+1))
    fi
}

echo "=== disk ==="
df -h /root/autodl-tmp | tail -1

echo ""
echo "=== tools ==="
check "uv installed"          "command -v uv"
check "git-lfs installed"     "command -v git-lfs"
check "hf cli installed"      "command -v hf"
check "miniconda python"      "test -x /root/miniconda3/bin/python"

echo ""
echo "=== repo ==="
check "main repo cloned"      "test -d $REPO_DIR/.git"
check "LeIsaac submodule"     "test -d $REPO_DIR/LeIsaac/scripts/finetune/gr00t"
check "Isaac-GR00T submodule" "test -d $REPO_DIR/dependencies/Isaac-GR00T/gr00t"
check "N1.7 train.sh"         "test -f $REPO_DIR/LeIsaac/scripts/finetune/gr00t/train_n17.sh"
check "N1.7 wrapper"          "test -f $REPO_DIR/LeIsaac/scripts/finetune/gr00t/launch_finetune_ckpt_n17.py"
check "N1.7 modality cfg"     "test -f $REPO_DIR/LeIsaac/scripts/finetune/gr00t/leisaac_config_n17.py"
check "Isaac-GR00T LFS wheels (dense, not pointer)" \
    "test $(stat -c %s $REPO_DIR/dependencies/Isaac-GR00T/scripts/deployment/dgpu/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl 2>/dev/null || echo 0) -gt 1000000"

echo ""
echo "=== base model ==="
check "Cosmos dir exists"            "test -d $COSMOS_DIR"
check "Cosmos config.json"           "test -f $COSMOS_DIR/config.json"
check "Cosmos model.safetensors"     "test -f $COSMOS_DIR/model.safetensors"
# dense file check (no sparse holes)
if [[ -f "$COSMOS_DIR/model.safetensors" ]]; then
    SIZE=$(stat -c %s "$COSMOS_DIR/model.safetensors")
    BLOCKS=$(stat -c %b "$COSMOS_DIR/model.safetensors")
    DENSE=$((BLOCKS * 512))
    RATIO=$((DENSE * 100 / SIZE))
    if [[ $RATIO -gt 95 ]]; then
        echo "  ✓ Cosmos model.safetensors dense ($RATIO%)"
        PASS=$((PASS+1))
    else
        echo "  ✗ Cosmos model.safetensors SPARSE ($RATIO%) — sparse-hole bug; redownload" >&2
        FAIL=$((FAIL+1))
    fi
fi
check "Cosmos tokenizer.json"        "test -f $COSMOS_DIR/tokenizer.json"
check "Cosmos preprocessor_config"   "test -f $COSMOS_DIR/preprocessor_config.json"

echo ""
echo "=== dataset ==="
DS_SLUG="${DATASET_REPO//\//--}"
check "dataset cached"               "test -d /root/autodl-tmp/hf_cache/hub/datasets--$DS_SLUG"

echo ""
echo "=== uv venv ==="
VENV="$REPO_DIR/dependencies/Isaac-GR00T/.venv"
check "venv exists"                  "test -d $VENV"
if [[ -d "$VENV" ]]; then
    check "torch importable"         "$VENV/bin/python -c 'import torch'"
    check "flash_attn importable"    "$VENV/bin/python -c 'import flash_attn'"
fi

echo ""
echo "=== GPU mode? ==="
if nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -1
    echo "  → GPU mode (training-ready)"
else
    echo "  → no-card mode (training will fail; reboot in GPU mode)"
fi

echo ""
echo "=== summary ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
[[ $FAIL -eq 0 ]]
