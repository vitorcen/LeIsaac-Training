"""FastWAM QLoRA finetune scaffold for LeIsaac SO-101 PickOrange.

Modules:
- qlora_utils:  In-place NF4 quantization of `nn.Linear` and PEFT LoRA injection.
- train:        Hydra entry that builds FastWAM (bf16), quantizes non-target Linears
                to NF4, attaches LoRA adapters on attention q/k/v/o, then hands the
                model to the upstream Wan22Trainer with a freeze override.
- precompute_text_embeds:  Tiny wrapper that delegates to the upstream
                `scripts/precompute_text_embeds.py` with `override_instruction`
                so we only encode the one PickOrange prompt.
"""
