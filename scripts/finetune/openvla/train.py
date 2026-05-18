"""OpenVLA-7B QLoRA finetune on LeIsaac SO-101 PickOrange.

Single-GPU (24GB 4090) training:
  - 4-bit NF4 base (bitsandbytes)
  - LoRA r=32, target_modules=[q_proj, v_proj] (OpenVLA paper recipe)
  - HF Trainer + gradient checkpointing
  - bf16 compute dtype, fp32 LoRA + AdamW(8-bit)
  - 10k steps, save every 1000 (watchdog prunes; see feedback-training-save-policy)

Usage:
    bash LeIsaac/scripts/finetune/openvla/train.sh
"""
from __future__ import annotations

# Eagerly *fully iterate* every Enum in torchgen.model BEFORE peft/bnb walk
# the module tree.  Python enum members are lazy-populated; under racy
# multi-thread/_named_members iteration the partial state surfaces as
#   ValueError: not enough values to unpack (expected 92, got 2)
#   ValueError: too many values to unpack (expected 0)
# A bare `import` is NOT enough — must call list(Enum) to force the
# `_member_names_` cache to fully populate.  DispatchKey has 135 members and
# is the usual culprit.
import enum as _enum
import torchgen.model as _torchgen_model  # noqa: F401
for _name in dir(_torchgen_model):
    _obj = getattr(_torchgen_model, _name, None)
    if isinstance(_obj, type) and issubclass(_obj, _enum.Enum) and _obj is not _enum.Enum:
        list(_obj)         # forces _member_names_ + _member_map_ to populate
        list(_obj.__members__)
del _name, _obj

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# Allow `python -m openvla.train` after PYTHONPATH=$LEISAAC/scripts/finetune
from openvla.dataset import (
    ACTION_DIM,
    ActionTokenizer,
    LeIsaacOpenVLADataset,
    OpenVLACollator,
    compute_action_stats,
    load_actions_states,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--dataset", required=True, help="LeRobot v3.0 dataset root")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume", default=None, help="resume from checkpoint dir")
    # LoRA
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_targets", default="q_proj,v_proj")
    # training
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_steps", type=int, default=10_000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="Disable gradient checkpointing (24GB has headroom; "
                         "ckpt + accelerate hooks have flaky bnb interactions).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --------------------------------------------------------------------- #
    # Action statistics — computed once on the full training set
    # --------------------------------------------------------------------- #
    stats_path = out_dir / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            raw = json.load(f)
        action_stats = {k: np.asarray(v) for k, v in raw["action"].items() if k != "mask"}
        action_stats["mask"] = np.asarray(raw["action"]["mask"], dtype=bool)
        canonical_prompt = raw.get("prompt", "Grab orange and place into plate")
        print(f"[openvla-train] loaded action stats from {stats_path}", flush=True)
    else:
        print("[openvla-train] computing action stats on full dataset...", flush=True)
        actions, _, _, _ = load_actions_states(args.dataset)
        action_stats = compute_action_stats(actions)
        # Canonical prompt = the dataset's stored task string.  Train, offline
        # eval, and server MUST all use this exact string (codex review
        # 2026-05-18: prompt drift is a #1 train/serve mismatch source).
        from openvla.dataset import load_episodes
        eps = load_episodes(args.dataset)
        canonical_prompt = eps[0].task
        with open(stats_path, "w") as f:
            json.dump(
                {"action": {k: v.tolist() for k, v in action_stats.items()},
                 "prompt": canonical_prompt},
                f, indent=2,
            )
        print(f"[openvla-train] wrote {stats_path}", flush=True)
        print(f"  q01={action_stats['q01']}\n  q99={action_stats['q99']}", flush=True)
        print(f"  canonical_prompt={canonical_prompt!r}", flush=True)

    # --------------------------------------------------------------------- #
    # Model + processor (4-bit base, bf16 compute)
    # --------------------------------------------------------------------- #
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[openvla-train] loading 4-bit NF4 {args.model}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"[openvla-train] base loaded; GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # --------------------------------------------------------------------- #
    # LoRA wrap (q_proj + v_proj on the Llama backbone only)
    # --------------------------------------------------------------------- #
    use_grad_ckpt = not args.no_grad_ckpt
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_grad_ckpt)
    if not use_grad_ckpt:
        # prepare_model_for_kbit_training skips enable_input_require_grads when
        # checkpointing is off, but for 4-bit training the embeddings still need
        # gradients enabled so backward can flow through Linear4bit.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --------------------------------------------------------------------- #
    # Dataset + collator
    # --------------------------------------------------------------------- #
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    action_tok = ActionTokenizer(tokenizer_vocab_size=tok.vocab_size, n_bins=256)

    train_ds = LeIsaacOpenVLADataset(args.dataset, action_stats=action_stats)
    collator = OpenVLACollator(processor, action_tok, action_stats)
    print(f"[openvla-train] dataset size: {len(train_ds)} samples", flush=True)

    # --------------------------------------------------------------------- #
    # Trainer
    # --------------------------------------------------------------------- #
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.log_steps,
        bf16=True,
        # NOTE: paged_adamw_8bit triggers a "Linear4bit has no attribute 'set'"
        # crash mid-training (bnb 0.43 + transformers 4.40 + peft 0.11).  Our
        # trainable params are only ~17M → optim state is ~70 MB even in fp32,
        # paged offload buys nothing.  Stick with the standard AdamW.
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    # use_cache must be off w/ grad ckpt — also during training in general
    model.config.use_cache = False

    # Auto-resume from latest checkpoint in output_dir if --resume not given.
    # Validate completeness before accepting — a watchdog kill mid-save can
    # leave a torn checkpoint with missing optimizer.pt or trainer_state.json,
    # which then loads as zeroed state and silently destroys the run.
    def _ckpt_is_complete(d: Path) -> bool:
        required = [
            "trainer_state.json",
            "training_args.bin",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ]
        # adapter weights live in either adapter_model.safetensors or .bin
        has_weights = (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists()
        return has_weights and all((d / f).exists() for f in required)

    resume_target = args.resume
    if resume_target is None:
        ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1)
        while ckpts:
            cand = ckpts[-1]
            if _ckpt_is_complete(cand):
                resume_target = str(cand)
                print(f"[openvla-train] auto-resuming from {resume_target}", flush=True)
                break
            print(f"[openvla-train] skipping torn ckpt {cand}", flush=True)
            ckpts.pop()

    print("[openvla-train] starting train loop", flush=True)
    trainer.train(resume_from_checkpoint=resume_target)
    print("[openvla-train] training complete; saving final adapter", flush=True)

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
