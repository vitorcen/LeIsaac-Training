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
import transformers.modeling_utils as _mu
import accelerate.utils.modeling as _am
import accelerate.utils as _au
import transformers.integrations.bitsandbytes as _tb

# Root-cause fix for two related _named_members tuple-unpack crash classes:
#
#   (1) Mid-training, from Trainer.log:
#       Trainer.log -> model.floating_point_ops -> num_parameters
#       -> named_parameters -> _named_members -> ValueError
#
#   (2) At model load, from bnb 4-bit quantizer:
#       quantizer_bnb_4bit.preprocess_model -> get_keys_to_not_convert
#       -> accelerate.find_tied_parameters -> model.named_parameters
#       -> _named_members -> ValueError
#
# Both crash inside torch's _named_members iterating bnb Linear4bit's
# _parameters proxy under PEFT wrapping (malformed tuple yield).
#
# (1) floating_point_ops -> 0 — only loses the cosmetic TFLOPS log column.
# (2) find_tied_parameters -> [] — declares "no tied params"; for OpenVLA
#     this means bnb will quantize lm_head, which is fine because OpenVLA
#     emits action tokens (not text) and inference already runs with the
#     quantized head.
# Watchdog stays as belt-and-braces for the remaining segfault class.
_mu.PreTrainedModel.floating_point_ops = lambda self, inputs, exclude_embeddings=True: 0
_noop_tied = lambda *a, **kw: []
_am.find_tied_parameters = _noop_tied
_au.find_tied_parameters = _noop_tied  # re-export in accelerate.utils
_tb.find_tied_parameters = _noop_tied  # local copy already imported by bnb integration

# (3) bnb 0.46.x removed MatmulLtState.memory_efficient_backward but PEFT 0.11.x
#     still references it during PeftModel.from_pretrained for 8-bit quantized
#     bases.  Patch attr at class level so all instances have it (default False).
from bitsandbytes.autograd._functions import MatmulLtState as _MatmulLtState
_MatmulLtState.memory_efficient_backward = False

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
    ap.add_argument("--load_lora", default=None,
                    help="Path to a checkpoint dir to seed LoRA weights ONLY "
                         "(adapter_model.safetensors).  Skips Trainer auto-resume, "
                         "fresh AdamW + step=0.  Use to recover LoRA progress "
                         "across precision/optimizer changes without dragging "
                         "the stale optimizer.pt / trainer_state.json baggage.")
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
    ap.add_argument("--quant", choices=["4bit", "8bit", "bf16"], default="8bit",
                    help="Base precision. 4bit=NF4 QLoRA (CRASH-PRONE on torch 2.3 "
                         "+ PEFT due to Params4bit.__tensor_flatten__ bug, see "
                         "openvla_crash_diagnosis.html). 8bit=Linear8bit_lt LoRA "
                         "(default; ~7GB base, no tensor-subclass bug). bf16=no "
                         "bnb at all (~14GB base, slowest, zero quant risk).")
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
    # Model + processor.  --quant selects base precision:
    #   4bit  → NF4 QLoRA (Params4bit, CRASH-PRONE — see crash_diagnosis HTML)
    #   8bit  → Linear8bit_lt LoRA (default; no tensor-subclass bug)
    #   bf16  → no bnb at all
    # --------------------------------------------------------------------- #
    if args.quant == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:  # bf16
        bnb_cfg = None
    print(f"[openvla-train] loading {args.quant} {args.model}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = dict(
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if bnb_cfg is not None:
        model_kwargs["quantization_config"] = bnb_cfg
    model = AutoModelForVision2Seq.from_pretrained(args.model, **model_kwargs)
    print(f"[openvla-train] base loaded; GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # --------------------------------------------------------------------- #
    # LoRA wrap (q_proj + v_proj on the Llama backbone only)
    # --------------------------------------------------------------------- #
    use_grad_ckpt = not args.no_grad_ckpt
    if args.quant in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_grad_ckpt)
        if not use_grad_ckpt:
            # prepare_model_for_kbit_training skips enable_input_require_grads
            # when checkpointing is off, but embeddings still need grad-enabled
            # for backward to flow through bnb quantized layers.
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
    else:  # bf16 — no kbit prep needed
        if use_grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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

    # If --load_lora is set BUT output_dir already has a more recent complete
    # ckpt (e.g. watchdog retry after first chunk partially saved), prefer
    # auto-resume over seeded LoRA.  Otherwise we'd silently redo work and lose
    # post-seed training progress on every crash.
    if args.load_lora:
        _existing = sorted(
            Path(args.output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1,
        )
        for cand in reversed(_existing):
            required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
            has_weights = (cand / "adapter_model.safetensors").exists() or (cand / "adapter_model.bin").exists()
            if has_weights and all((cand / f).exists() for f in required):
                print(f"[openvla-train] --load_lora overridden: output_dir has newer "
                      f"complete ckpt {cand.name} → using Trainer auto-resume instead "
                      f"(load_lora was for first-chunk seed only)", flush=True)
                args.load_lora = None
                break

    # Optional: seed LoRA weights from an existing checkpoint without dragging
    # the stale optimizer/scheduler/trainer_state.  Useful when changing base
    # precision (4bit → 8bit) — adapter weights transfer fine but optimizer
    # state would be invalidated.
    if args.load_lora:
        from safetensors.torch import load_file as _safe_load
        from peft.utils import set_peft_model_state_dict
        lora_path = Path(args.load_lora)
        weights_file = lora_path / "adapter_model.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(f"--load_lora: missing {weights_file}")
        lora_sd = _safe_load(str(weights_file))
        result = set_peft_model_state_dict(model, lora_sd)
        n_loaded = len([k for k in lora_sd if "lora_" in k])
        print(f"[openvla-train] seeded LoRA from {weights_file}: {n_loaded} tensors", flush=True)
        if hasattr(result, "missing_keys") and result.missing_keys:
            print(f"[openvla-train]   missing: {len(result.missing_keys)}", flush=True)
        if hasattr(result, "unexpected_keys") and result.unexpected_keys:
            print(f"[openvla-train]   unexpected: {len(result.unexpected_keys)}", flush=True)

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
    if args.load_lora:
        # --load_lora forces a fresh-optimizer restart: LoRA weights already
        # seeded above, Trainer must NOT load optimizer.pt / trainer_state.
        resume_target = None
        print("[openvla-train] --load_lora set → skipping Trainer auto-resume "
              "(fresh AdamW, step=0)", flush=True)
    elif resume_target is None:
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

    # Override save_steps in resumed trainer_state.  Trainer prefers state-file
    # value over CLI, so resume from ckpt-5200 with save_steps=200 always saves
    # next at 5400 — but bnb/CUDA segfaults reliably ~120 steps post-resume
    # (i.e. step 5320), so 5400 is never reached.  Shorter save interval lets
    # each watchdog attempt make permanent progress before the next crash.
    if resume_target:
        state_file = Path(resume_target) / "trainer_state.json"
        if state_file.exists():
            st = json.loads(state_file.read_text())
            if st.get("save_steps") != args.save_steps:
                old = st.get("save_steps")
                st["save_steps"] = args.save_steps
                state_file.write_text(json.dumps(st, indent=2))
                print(f"[openvla-train] override save_steps in trainer_state: {old} -> {args.save_steps}", flush=True)

    print("[openvla-train] starting train loop", flush=True)
    trainer.train(resume_from_checkpoint=resume_target)

    # Force a checkpoint at the exact max_steps boundary.  Trainer respects
    # trainer_state.json's save_steps (e.g. 200) over CLI's --save_steps so
    # the natural saves land at 5200/5400 — but the chunked loop expects
    # checkpoint-$MAX_STEPS to exist.  This guarantees it.
    end_step = trainer.state.global_step
    end_ckpt = out_dir / f"checkpoint-{end_step}"
    if not end_ckpt.exists():
        print(f"[openvla-train] saving boundary ckpt at step {end_step}", flush=True)
        trainer._save_checkpoint(model, trial=None)  # writes checkpoint-{global_step}

    print("[openvla-train] training complete; saving final adapter", flush=True)
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
