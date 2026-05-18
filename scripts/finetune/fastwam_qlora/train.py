"""FastWAM QLoRA finetune entry — wraps fastwam.runtime.run_training.

Pipeline:
  1. Build FastWAM with upstream `create_fastwam` (bf16, base ckpt loaded).
  2. Quantize all non-LoRA-target Linears in `model.dit` to NF4 (bitsandbytes).
  3. Wrap q/k/v/o with PEFT LoRA adapters.
  4. Monkey-patch Wan22Trainer's freeze hook to respect PEFT's requires_grad.
  5. Run the upstream Wan22Trainer loop unchanged.

Usage (via train.sh):
    bash LeIsaac/scripts/finetune/fastwam/train.sh

Or directly:
    cd ~/work/fastwam-repo
    conda activate fastwam
    PYTHONPATH=$LEISAAC/scripts/finetune python -m fastwam.train_qlora ...
"""

from __future__ import annotations

# Eagerly construct torch's `torchgen.model.DispatchKey` enum class.  On
# torch 2.7.1 + python 3.10, lazy first-use of this enum from deep inside
# peft/bnb walks intermittently fails with
#   `ValueError: not enough values to unpack (expected 92, got 2)`
# crashing the process at random training steps.  Forcing the class to
# construct here, before anything else touches the model, caches it
# globally and dodges the race.
import torchgen.model as _torchgen_model  # noqa: F401

import logging
import os
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# fastwam imports
from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

# local imports
from fastwam_qlora.qlora_utils import (
    DEFAULT_LORA_TARGETS,
    apply_lora,
    patch_trainer_for_qlora,
    quantize_linears_nf4,
)

register_default_resolvers()
logger = get_logger(__name__)


def _apply_qlora(model: torch.nn.Module, qlora_cfg: DictConfig) -> torch.nn.Module:
    """LoRA wrap FIRST, then NF4 quantize the rest.

    The reverse order (quantize → wrap) segfaults intermittently in peft's
    `_mark_only_adapters_as_trainable` when it walks a tree containing
    `bnb.Linear4bit` — bnb's `Params4bit` quant_state confuses torch's
    `_named_members` C iterator.  Wrapping first means peft only ever sees
    plain `nn.Linear`; the subsequent NF4 swap skips the now-wrapped LoRA Linears.
    """
    targets = tuple(qlora_cfg.get("target_modules", list(DEFAULT_LORA_TARGETS)))
    r = int(qlora_cfg.get("lora_r", 16))
    alpha = int(qlora_cfg.get("lora_alpha", 16))
    dropout = float(qlora_cfg.get("lora_dropout", 0.0))

    logger.info("LoRA-wrap target_modules=%s (r=%d α=%d) BEFORE quantize", targets, r, alpha)
    model.dit = apply_lora(model.dit, r=r, alpha=alpha, dropout=dropout, target_modules=targets)

    logger.info("NF4-quantize remaining Linears (LoRA-wrapped q/k/v/o left alone)...")
    quantize_linears_nf4(model.dit, lora_target_suffixes=targets)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9
        logger.info("Post-QLoRA GPU mem: used=%.2fGB free=%.2fGB", used_gb, free_gb)

    return model


def run_training_qlora(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0
        if torch.distributed.is_initialized()
        else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    logger.info("Building FastWAM (device=%s dtype=%s)...", model_device, model_dtype)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)

    # ---- QLoRA hook ----
    qlora_cfg = cfg.get("qlora")
    if qlora_cfg is None:
        raise ValueError("`cfg.qlora` block required for QLoRA training.")
    model = _apply_qlora(model, qlora_cfg)
    patch_trainer_for_qlora()

    train_ds, val_ds = build_datasets(cfg.data)
    trainer = Wan22Trainer(cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds)
    trainer.train()


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training_qlora(cfg)


if __name__ == "__main__":
    main()
