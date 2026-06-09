#!/usr/bin/env bash
#
# LeIsaac 资产下载 / asset provisioning.
# 从 HF 拉 LightwheelAI/leisaac_env 并把 assets 同步进 LeIsaac/assets。
# submodule 初始化由伞仓 init.sh 负责（git submodule update --init），这里只管资产。
#
# 用法（在伞仓根或任意位置）:  bash LeIsaac/init.sh
#   env LEISAAC_ENV_CACHE  覆盖本地缓存目录（默认 LeIsaac/.cache/leisaac_env）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = LeIsaac
ASSETS_DIR="$ROOT_DIR/assets"
CACHE_DIR="${LEISAAC_ENV_CACHE:-$ROOT_DIR/.cache/leisaac_env}"
HF_REPO_URL="https://huggingface.co/LightwheelAI/leisaac_env"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log_error "缺少命令: $1"; exit 1
    fi
}

sync_hf_repo() {
    if [ ! -d "$CACHE_DIR/.git" ]; then
        log_info "克隆 leisaac_env 到本地缓存 $CACHE_DIR ..."
        mkdir -p "$(dirname "$CACHE_DIR")"
        git clone --depth 1 "$HF_REPO_URL" "$CACHE_DIR"
    else
        log_info "更新本地缓存仓库..."
        git -C "$CACHE_DIR" pull --ff-only
    fi
}

copy_assets() {
    mkdir -p "$ASSETS_DIR"
    log_info "同步资产到 LeIsaac/assets ..."
    rsync -a "$CACHE_DIR/assets/" "$ASSETS_DIR/"
}

verify_assets() {
    local required_files=(
        "$ASSETS_DIR/robots/so101_follower.usd"
        "$ASSETS_DIR/scenes/kitchen_with_orange/scene.usd"
    )
    for f in "${required_files[@]}"; do
        if [ ! -f "$f" ]; then
            log_error "缺少关键资产: $f"; exit 1
        fi
    done
    log_info "资产校验通过。"
}

main() {
    log_info "========================================="
    log_info "LeIsaac 资产下载脚本"
    log_info "========================================="
    require_cmd git
    require_cmd rsync
    sync_hf_repo
    copy_assets
    verify_assets
    echo ""
    log_info "完成。你现在可以运行 LeIsaac/LeIsaac.ipynb 里的推理单元。"
}

main "$@"
