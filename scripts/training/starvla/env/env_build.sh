#!/bin/bash
# StarVLA env build on westc. Logs everything. Env on autodl-tmp (system disk too small).
set -o pipefail
exec > /root/starvla_env_build.log 2>&1
echo "=== StarVLA env build start ==="
date

ENV=/root/autodl-tmp/envs/starvla
CONDA=/root/miniconda3/bin/conda
export PATH=/usr/local/cuda-12.4/bin:$PATH
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890

echo "### nvcc check"; nvcc -V | tail -2

# --- Phase A: conda env + torch ---
if [ ! -x "$ENV/bin/python" ]; then
  echo "### creating conda env (py3.10) on autodl-tmp"
  # conda create needs no proxy (uses tsinghua/aliyun); unset for conda step
  https_proxy= http_proxy= $CONDA create -y --prefix $ENV python=3.10 2>&1 | tail -5
fi
PIP="$ENV/bin/pip"
echo "### python:"; $ENV/bin/python --version

# aliyun has no torch 2.6.0+cu124 -> use download.pytorch.org via mihomo proxy (proven in wallx).
echo "### install torch 2.6.0 + torchvision 0.21.0 (pytorch.org cu124 via proxy)"
$PIP install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -10
echo "### torch check"; $ENV/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" || echo "TORCH FAIL"

# --- Phase B: requirements (aliyun pypi default) ---
echo "### install requirements.txt"
https_proxy= http_proxy= $PIP install --no-cache-dir -r /root/autodl-tmp/starVLA/requirements.txt 2>&1 | tail -15
echo "### transformers check"; $ENV/bin/python -c "import transformers;print('transformers',transformers.__version__)" || echo "TF FAIL"

# --- Phase C: flash-attn prebuilt wheel (cp310 torch2.6 cxx11abiFALSE) via proxy ---
echo "### install flash-attn prebuilt wheel"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
$PIP install --no-cache-dir --no-deps "$FA_URL" 2>&1 | tail -6
echo "### flash-attn check"; $ENV/bin/python -c "import flash_attn;print('flash_attn',flash_attn.__version__)" || echo "FA FAIL"

# --- Phase D: editable install of starVLA ---
echo "### pip install -e starVLA (no deps)"
cd /root/autodl-tmp/starVLA
https_proxy= http_proxy= $PIP install --no-cache-dir -e . --no-deps 2>&1 | tail -6

echo "=== final import smoke ==="
$ENV/bin/python -c "
import torch, transformers, flash_attn, deepspeed, accelerate
print('torch', torch.__version__, '| tf', transformers.__version__, '| fa', flash_attn.__version__, '| ds', deepspeed.__version__, '| acc', accelerate.__version__)
print('cuda avail', torch.cuda.is_available())
" || echo "FINAL IMPORT FAIL"
echo "=== StarVLA env build DONE ==="
date
