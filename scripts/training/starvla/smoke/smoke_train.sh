#!/bin/bash
# 3-step forward/backward smoke for QwenGR00T SO-101. Validates model load + freeze +
# deepspeed init + dataloader-through-trainer + forward + loss + backward, no OOM.
exec > /root/starvla_smoke.log 2>&1
set -o pipefail
ENV=/root/autodl-tmp/envs/starvla
cd /root/autodl-tmp/starVLA
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export HF_HOME=/root/autodl-tmp/hf_cache
export PATH=$ENV/bin:/usr/local/cuda-12.4/bin:$PATH
echo "=== SMOKE start ==="; date
$ENV/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 --main_process_port 29531 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/SO101_PickOrange/train_files/configs/so101_qwen_gr00t.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm /root/autodl-tmp/models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /root/autodl-tmp/datasets \
  --datasets.vla_data.data_mix so101_pickorange \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps 3 \
  --trainer.save_interval 999999 \
  --trainer.logging_frequency 1 \
  --run_root_dir /root/autodl-tmp/starvla-smoke \
  --run_id smoke
echo "=== SMOKE exited code $? ==="; date
