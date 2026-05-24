#!/bin/bash
# DreamZero bf16 LoRA finetune on LeIsaac SO-101 PickOrange.
# Pivoted from INT8 LoRA (2026-05-23 reflection: DreamZero source has no bnb hooks → patch cost > GPU upgrade cost).
# Target: 1× RTX Pro 6000 Blackwell 96 GB (or H100 80 GB), CUDA 12.8 + PyTorch 2.7+.
#
# Prerequisites (do these once on the cloud machine):
#   1. bash convert_leisaac_to_gear.sh /root/autodl-tmp/leisaac-pick-orange
#   2. cp leisaac_relative.yaml /root/autodl-tmp/dreamzero-repo/groot/vla/configs/data/dreamzero/leisaac_relative.yaml
#   3. cd /root/autodl-tmp/dreamzero-repo && pip install -e .
#
# Usage:
#   bash train_dreamzero_bf16_lora.sh

set -e
export HYDRA_FULL_ERROR=1
# Reduce CUDA memory fragmentation (helps when GPU is near-full like Pro 6000 96G with 14B model + activations)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Multi-GPU NCCL fixes for Blackwell sm_120 on cross-root-complex PCIe (e.g. Pro 6000 0x38: + 0xB8:):
# P2P fails with "illegal memory access" → fall back to shared-memory via CPU.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
# Force-preload newer NCCL 2.30+ for Blackwell sm_120 support (torch 2.7 bundle ships 2.26 → illegal-mem).
export LD_PRELOAD=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}

# Ensure conda python/torchrun on PATH (AutoDL bash doesn't auto-activate conda)
if [ -d /root/miniconda3/bin ] && [[ ":$PATH:" != *":/root/miniconda3/bin:"* ]]; then
    export PATH="/root/miniconda3/bin:$PATH"
fi
TORCHRUN=${TORCHRUN:-$(command -v torchrun || echo /root/miniconda3/bin/torchrun)}

# ============ USER CONFIGURATION ============
DREAMZERO_REPO=${DREAMZERO_REPO:-/root/autodl-tmp/dreamzero-repo}
LEISAAC_DATA_ROOT=${LEISAAC_DATA_ROOT:-/root/autodl-tmp/leisaac-pick-orange-h264}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/dreamzero_leisaac_so101_lora}
# LoRA rank: DreamZero default = 4 (video diffusion DiT — much smaller than LLM LoRA norms).
#   r=4: 19M params, ~200 MB per ckpt — match Vizuara SO-101 LoRA (209 MB) and DreamZero pretrains.
#   r=16: 76M, ~600 MB — only try if r=4 underfits on first 2k step.
#   r>=32: untested; risk of overfit on 50-demo + dropping pretrain-compatible init structure.
LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-4}
# SMOKE=1 → run quick 100-step smoke test (verifies save_only_model fix without 3h wait)
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_STEPS=${MAX_STEPS:-100}
    SAVE_STEPS=${SAVE_STEPS:-50}
else
    MAX_STEPS=${MAX_STEPS:-10000}
    SAVE_STEPS=${SAVE_STEPS:-2000}
fi
# DISABLE_SAVE=1 → entirely skip HF Trainer auto-save (verifies training itself can pass step 50/100 without RAM spike from save path)
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
if [ "${DISABLE_SAVE:-0}" = "1" ]; then
    SAVE_STRATEGY=no
fi
WAN_CKPT_DIR=${WAN_CKPT_DIR:-/root/autodl-tmp/wan2.1-i2v-14b-480p}
TOKENIZER_DIR=${TOKENIZER_DIR:-/root/autodl-tmp/umt5-xxl}
NUM_GPUS=${NUM_GPUS:-1}
# =============================================

# Validate paths
for p in "$LEISAAC_DATA_ROOT" "$WAN_CKPT_DIR" "$TOKENIZER_DIR" "$DREAMZERO_REPO"; do
    if [ ! -d "$p" ]; then
        echo "ERROR: required path not found: $p"
        exit 1
    fi
done

if [ ! -f "$LEISAAC_DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: $LEISAAC_DATA_ROOT/meta/embodiment.json missing — run convert_leisaac_to_gear.sh first"
    exit 1
fi

cd "$DREAMZERO_REPO"

# Single-GPU on Pro 6000 96G → no DeepSpeed needed (96G easily fits bf16 LoRA r=32 ~58G peak)
# Multi-GPU (e.g. 2× cards) → uncomment DeepSpeed line below
"$TORCHRUN" --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/leisaac_relative \
    wandb_project=dreamzero_leisaac \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    frame_seqlen=880 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    save_lora_only=true \
    "+training_args.save_only_model=true" \
    max_chunk_size=2 \
    save_strategy=$SAVE_STRATEGY \
    leisaac_data_root="$LEISAAC_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR"

# Notes on key hyperparameters:
#   - per_device_train_batch_size=1: smallest, 96G Pro 6000 could try 2 if loss looks stable
#   - max_steps=10000: ~3-4h on Pro 6000 96G; covers ~13 effective epochs on 60-demo data
#   - save_steps=2000 + save_total_limit=4: rolling 4 ckpts (~12.5 GB)
#   - num_views=2: front + wrist (LeIsaac has no third view; DreamZero base trained with 3 — slight schema mismatch may need adapter)
#   - num_frames=33, action_horizon=24, image 320×176: matches DROID/YAM official configs
#   - frame_seqlen=880: 33 frames × (320/8 × 176/8 / 4) ≈ 880 video tokens per chunk
#   - save_lora_only=false: include optimizer state to allow resume (4 × ~3 GB ckpt = 12 GB)
#   - learning_rate=1e-4: same as DROID LoRA; yam uses 1e-5 because of pretrain init
#   - LoRA rank default = 4 (video diffusion DiT norm; matches Vizuara SO-101 LoRA at 209 MB).
#     Do NOT crank to 32/64/128 like LLM LoRA — video diffusion LoRA breaks early past r=16
#     and 50-demo overfit risk is already high. Bump to r=16 only if loss plateaus by step 2000.
