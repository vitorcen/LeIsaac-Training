"""Smoke test: load OpenVLA 4-bit + LoRA, decode 1 frame, 1 forward+backward.

Run BEFORE the long training to catch:
  - dataset / video-decode wiring errors
  - tokenizer / collator shape mismatches
  - OOM with the chosen LoRA rank + batch size
  - LoRA target_modules that don't actually match any module

Usage:
    bash -c 'PYTHONPATH=$PWD/LeIsaac/scripts/finetune \
        conda run -n openvla --no-capture-output \
        python -m openvla.smoke \
        --dataset LeIsaac/datasets/raw/leisaac-pick-orange'
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from openvla.dataset import (
    ActionTokenizer,
    LeIsaacOpenVLADataset,
    OpenVLACollator,
    compute_action_stats,
    load_actions_states,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_targets", default="q_proj,v_proj")
    args = ap.parse_args()

    print("[smoke] === stage 1: dataset ===", flush=True)
    t0 = time.time()
    actions, _, _, _ = load_actions_states(args.dataset)
    stats = compute_action_stats(actions)
    print(f"  actions shape={actions.shape}  q01={stats['q01']}  q99={stats['q99']}")
    print(f"  load+stats: {time.time()-t0:.2f}s", flush=True)

    ds = LeIsaacOpenVLADataset(args.dataset, action_stats=stats)
    t0 = time.time()
    sample = ds[0]
    print(f"  __getitem__[0]: img={sample['image'].shape} {sample['image'].dtype}, "
          f"action={sample['action']}, prompt='{sample['prompt']}'", flush=True)
    print(f"  decode 1 frame: {time.time()-t0:.3f}s", flush=True)

    # Try a deep index that will hit the second mp4 (episode 58)
    t0 = time.time()
    deep_idx = int(np.where(ds.eps_idx == 58)[0][100])
    sample2 = ds[deep_idx]
    print(f"  __getitem__[{deep_idx}] (ep58 frame100): img={sample2['image'].shape}, "
          f"decode {time.time()-t0:.3f}s", flush=True)

    print("[smoke] === stage 2: model load (4-bit NF4) ===", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"  loaded in {time.time()-t0:.1f}s  GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    print("[smoke] === stage 3: LoRA wrap ===", flush=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[t.strip() for t in args.lora_targets.split(",")],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    print(f"  post-LoRA GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    print("[smoke] === stage 4: 1 forward + backward ===", flush=True)
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    action_tok = ActionTokenizer(tokenizer_vocab_size=tok.vocab_size, n_bins=256)
    collator = OpenVLACollator(processor, action_tok, stats)
    batch = collator([ds[0], ds[100]])
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    batch = {k: v.to("cuda:0") for k, v in batch.items()}
    # Leave pixel_values as float32 — the vision backbone (timm DINOv2+SigLIP)
    # keeps its conv/LN weights at fp32; only the Llama backbone is bf16.

    model.train()
    t0 = time.time()
    out = model(**batch)
    print(f"  forward: loss={out.loss.item():.4f}  GPU={torch.cuda.max_memory_allocated()/1e9:.2f}GB  "
          f"({time.time()-t0:.2f}s)", flush=True)

    t0 = time.time()
    out.loss.backward()
    print(f"  backward: GPU peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB  "
          f"({time.time()-t0:.2f}s)", flush=True)
    print("[smoke] ✅ all stages passed", flush=True)


if __name__ == "__main__":
    main()
