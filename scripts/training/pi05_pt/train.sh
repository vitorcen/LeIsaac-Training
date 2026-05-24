#!/usr/bin/env bash
# π0.5 PyTorch full fine-tune (expert-only, freeze-VLM) on LeIsaac SO-101 PickOrange.
#
# This is the "5M LoRA → 693M expert FT + main PyTorch stack" pivot from the
# MLX LoRA experiment (which scored 0/15 on the leaderboard). Same recipe as
# GR00T's `tune_visual=False, tune_llm=False, tune_diffusion_model=True`:
#
#   - VLM (PaliGemma 2B + SigLIP-So400m): FROZEN (3.45B params, ~7 GB VRAM as cold weights)
#   - Action expert (Gemma-300M-style, 18 layers): TRAINABLE (693M)
#   - Backend: PyTorch (not MLX), lerobot-train CLI, async-server compatible
#
# Why this should work better than MLX LoRA:
#   - 5M trainable → 693M trainable (~140× capacity)
#   - Same dual-system architecture family as SmolVLA (8/15) and X-VLA (9/18)
#   - Reuses the auto-eval watcher + 5-round leaderboard pipeline
#
# Phase 1 smoke (this script): 2500-step quick sanity, expect ≥2/9 in 3-round
# quick eval. If pass → continue to 10k step. If fail → diff config vs SmolVLA.
#
# Usage:
#   bash scripts/training/pi05_pt/train.sh                # 2500-step smoke
#   STEPS=10000 SAVE_FREQ=1000 bash scripts/training/pi05_pt/train.sh
#
# Knobs (env vars):
#   STEPS         total training steps                  (default 2500)
#   BATCH_SIZE    per-device batch                      (default 1; lerobot-train has no grad-accum,
#                                                        bump to 2 if VRAM permits)
#   SAVE_FREQ     ckpt save interval                    (default 500)
#   LR            AdamW peak lr                         (default 2.5e-5, openpi default)
#   DATASET_REPO_ID                                     (default LightwheelAI/leisaac-pick-orange)
#   OUTPUT_NAME   output dir name under outputs/        (default pi05-expert-leisaac-pick-orange)
#   AUTO_EVAL     0 to disable per-ckpt sanity eval     (default 1)
#
# VRAM expectation on RTX 4090 24G (bf16, batch=1, grad-ckpt ON):
#   - cold weight load:  ~9 GB
#   - peak during train: ~22-26 GB (might brush OOM — if so, GRAD_ACCUM up + lower max_seqlen)

set -euo pipefail

# -------- defaults --------
STEPS="${STEPS:-2500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# Default SAVE_FREQ = STEPS/5 (5 ckpts × 9.4 GB ≈ 47 GB disk; bump SAVE_FREQ to halve).
# lerobot saves the FULL 4B model (not just 693M trainable expert), so ckpts are big.
SAVE_FREQ="${SAVE_FREQ:-$((STEPS/5))}"
LR="${LR:-2.5e-5}"
DATASET_REPO_ID="${DATASET_REPO_ID:-LightwheelAI/leisaac-pick-orange}"
OUTPUT_NAME="${OUTPUT_NAME:-pi05-expert-leisaac-pick-orange}"
AUTO_EVAL="${AUTO_EVAL:-1}"

# Compose pi05-specific extra args:
#   - train_expert_only=true:        freeze all PaliGemma (VLM + LM), train only gemma_expert
#   - freeze_vision_encoder=true:    redundant w/ above but explicit (same as upstream defaults)
#   - gradient_checkpointing=true:   needed to fit 693M trainable + chunk=50 activations on 24G
#   - optimizer_lr / chunk_size:     match openpi reference (peak 2.5e-5, chunk 50)
EXTRA_ARGS="\
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.gradient_checkpointing=true \
    --policy.optimizer_lr=${LR} \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.dtype=bfloat16"

# Camera key mapping: pi05_base expects DROID schema (3 cams: base + left_wrist + right_wrist).
# LeIsaac SO-101 has 2 cams: front + wrist. Map our 2 to 2 of pi05's expected; pi05 will
# runtime-pad the missing left_wrist with -1 tensor + mask=0 (see modeling_pi05.py:1195).
# `validate_visual_features_consistency` accepts "provided ⊂ expected" so this passes.
RENAME_MAP="${RENAME_MAP:-{\"observation.images.front\":\"observation.images.base_0_rgb\",\"observation.images.wrist\":\"observation.images.right_wrist_0_rgb\"}}"

# Eval watcher horizon: chunk=50 trains, but horizon=35 is the known sweet spot
# from v3 MLX experiment (chunk-execution trap, see pi05_finetune_pick_orange.html §3).
# Override the watcher default via env so quick-eval matches the eventual full eval.
export EVAL_HORIZON="${EVAL_HORIZON:-35}"

# Hand off to the canonical launcher
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec env \
    BASE_MODEL=lerobot/pi05_base \
    DATASET_REPO_ID="${DATASET_REPO_ID}" \
    OUTPUT_NAME="${OUTPUT_NAME}" \
    STEPS="${STEPS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    SAVE_FREQ="${SAVE_FREQ}" \
    AUTO_EVAL="${AUTO_EVAL}" \
    RENAME_MAP="${RENAME_MAP}" \
    EXTRA_ARGS="${EXTRA_ARGS}" \
    bash "${REPO_ROOT}/scripts/training/lerobot_finetune.sh"
