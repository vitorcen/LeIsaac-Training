#!/bin/bash
# StarVLA SO-101 PickOrange training launcher — runs ON the AutoDL cloud box.
# VLM-agnostic: every variant-specific knob (framework.name, base_vlm,
# freeze_modules, action head) lives in the CONFIG yaml, NOT here. To train a new
# VLM x head variant you add ONE configs/*.yaml and point CONFIG at it — this
# launcher does not change.
#
#   CONFIG=examples/SO101_PickOrange/train_files/configs/so101_gemma4_pi_v3.yaml \
#   RUN_ID=so101_gemma4_pi_v3 RESUME=0 bash run_train.sh
#
# No `conda activate` (hangs in non-interactive SSH) -> full env-binary paths.
exec > /root/starvla_train.log 2>&1
set -o pipefail

ENV=${ENV:-/root/autodl-tmp/envs/starvla}
REPO=${REPO:-/root/autodl-tmp/starVLA}
# config yaml is read relative to $REPO cwd (deploy maps the kit into the repo,
# see README "云端部署映射"); default = the Qwen3-VL + GR00T-head Run-1 recipe.
CONFIG=${CONFIG:-examples/SO101_PickOrange/train_files/configs/so101_qwen_gr00t.yaml}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/starvla-outputs}
KEEP=${KEEP:-2}                 # patches/starvla/0003 prunes to newest-N -> never ENOSPC
cd "$REPO"

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.9
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export PATH=$ENV/bin:/usr/local/cuda-12.4/bin:$PATH
# training is offline (local base_vlm + local dataset) -> no proxy needed

# optional overrides — all framework-AGNOSTIC keys (do NOT add per-VLM keys here)
OVERRIDES="--datasets.vla_data.data_root_dir $DATA_ROOT --run_root_dir $RUN_ROOT --trainer.keep_last_checkpoints $KEEP"
[ -n "$RUN_ID" ]    && OVERRIDES="$OVERRIDES --run_id $RUN_ID"
[ -n "$MAX_STEPS" ] && OVERRIDES="$OVERRIDES --trainer.max_train_steps $MAX_STEPS"
[ -n "$BATCH" ]     && OVERRIDES="$OVERRIDES --datasets.vla_data.per_device_batch_size $BATCH"
if [ "${RESUME:-0}" = "1" ]; then
  OVERRIDES="$OVERRIDES --trainer.is_resume True"
  echo "### RESUME mode: will pick up latest checkpoint"
fi

echo "=== StarVLA training start: CONFIG=$CONFIG ==="; date
$ENV/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

$ENV/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --main_process_port ${MAIN_PORT:-29521} \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  $OVERRIDES

echo "=== training exited ==="; date
