"""Wrapper around Isaac-GR00T launch_finetune.py that enables gradient
checkpointing (not exposed via FinetuneConfig CLI).

Required for 4090 24GB single-GPU N1.6 finetune — without it, even per-step
batch=2 chunk=50 image=448×448 OOMs at ~22GB.

⭐ Also monkey-patches `transformers.PreTrainedModel.gradient_checkpointing_enable`
to default `use_reentrant=False`.  Upstream default is `True`, which is suspected
of triggering `RuntimeError: d.is_cuda() INTERNAL ASSERT FAILED at CUDAGuardImpl.h:34`
on bf16 + grad-ckpt + cross-attn during the backward recomputation pass
(reentrant checkpoint historical issues: pytorch#124788, #84864, #141896).

Drop-in replacement: same CLI as launch_finetune.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import tyro

# Ensure Isaac-GR00T's gr00t package is importable.
_GR00T_ROOT = Path(os.environ.get("GR00T_ROOT", str(Path.home() / "work" / "Isaac-GR00T")))
if str(_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(_GR00T_ROOT))

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run
from gr00t.experiment.launch_finetune import load_modality_config


# ⭐ Force `use_reentrant=False` on gradient-checkpointing.
# Without this, transformers' default is `use_reentrant=True`, the buggy mode
# that recomputes forward in a way that can leak CPU tensors into a CUDA-only
# autograd graph (root cause of `d.is_cuda() INTERNAL ASSERT FAILED`).
def _patch_gradient_checkpointing_default():
    from transformers import PreTrainedModel

    _orig = PreTrainedModel.gradient_checkpointing_enable

    def _patched(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        gradient_checkpointing_kwargs.setdefault("use_reentrant", False)
        # NOTE: NOT setting preserve_rng_state=False — N1.6 DiT has dropout=0.2,
        # so disabling RNG preservation would cause real numerical drift between
        # forward and backward recompute.  Stick to default True.
        print(
            f"[gr00t-train] patched gradient_checkpointing_enable kwargs = "
            f"{gradient_checkpointing_kwargs}",
            flush=True,
        )
        return _orig(self, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    PreTrainedModel.gradient_checkpointing_enable = _patched


_patch_gradient_checkpointing_default()


# ⭐ Selective checkpoint pruning via HF Trainer callback:
# After each save, keep multiples of KEEP_MULTIPLE permanent + only last KEEP_TEMPORARY
# non-multiples.  Cleaner than relying on watchdog post-cycle prune which only fires
# on crash/exit — this prunes within a long training cycle too.
def _patch_save_pruning():
    import re, shutil
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer

    keep_mult = int(os.environ.get("KEEP_MULTIPLE", "500"))
    keep_temp = int(os.environ.get("KEEP_TEMPORARY", "3"))

    class CheckpointPruneCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            out = args.output_dir
            if not os.path.isdir(out):
                return
            ckpts = []
            for d in os.listdir(out):
                m = re.match(r"^checkpoint-(\d+)$", d)
                if m:
                    ckpts.append((int(m.group(1)), os.path.join(out, d)))
            ckpts.sort()
            permanent = {s for s, _ in ckpts if s > 0 and s % keep_mult == 0}
            temporary = [(s, p) for s, p in ckpts if s not in permanent]
            to_keep = set(s for s, _ in temporary[-keep_temp:])
            removed = []
            for s, p in temporary:
                if s in to_keep:
                    continue
                shutil.rmtree(p, ignore_errors=True)
                removed.append(s)
            if removed:
                print(f"[prune] removed {removed}  kept perm={sorted(permanent)} last_temp={sorted(to_keep)}", flush=True)

    _orig_trainer_init = Gr00tTrainer.__init__

    def _patched_trainer_init(self, *args, **kwargs):
        _orig_trainer_init(self, *args, **kwargs)
        self.add_callback(CheckpointPruneCallback())
        print(f"[gr00t-train] CheckpointPruneCallback installed (keep_mult={keep_mult}, keep_temp={keep_temp})", flush=True)

    Gr00tTrainer.__init__ = _patched_trainer_init


_patch_save_pruning()


# ⭐ LISA opt-in: when LISA_ENABLE=1, monkey-patch Gr00tTrainer.__init__ to
# auto-attach LISACallback so layer-sampling kicks in from step 0.
# Block path on Gr00tN1d6:  action_head.model.transformer_blocks
def _patch_lisa_injection():
    if os.environ.get("LISA_ENABLE", "0") != "1":
        return
    from gr00t.experiment.trainer import Gr00tTrainer

    k = int(os.environ.get("LISA_K", "2"))
    resample_every = int(os.environ.get("LISA_RESAMPLE_EVERY", "1"))

    _sys_path_lisa = str(Path(__file__).resolve().parent)
    if _sys_path_lisa not in sys.path:
        sys.path.insert(0, _sys_path_lisa)
    from lisa_callback import LISACallback

    _orig_trainer_init = Gr00tTrainer.__init__

    def _patched_trainer_init(self, *args, **kwargs):
        _orig_trainer_init(self, *args, **kwargs)
        cb = LISACallback(
            model=self.model,
            k=k,
            block_attr_paths=["action_head.model.transformer_blocks"],
            resample_every=resample_every,
        )
        self.add_callback(cb)
        print(f"[gr00t-train] LISA callback installed (K={k}, resample_every={resample_every})", flush=True)

    Gr00tTrainer.__init__ = _patched_trainer_init


_patch_lisa_injection()


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    # ⭐ 4090 memory squeeze: skip the fp32 master copy of trainable backbone
    # params.  Saves ~3 GB at minor numerical accuracy cost.  Upstream warns
    # this option will be deprecated; revisit when targeting ≥40GB GPUs.
    config.model.backbone_trainable_params_fp32 = False
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    # ⭐ 4090 memory squeeze: Adafactor uses factorized 2nd-moment (no per-param
    # variance), ~1 byte per param vs Adam's 8 bytes.  Saves ~4 GB on 600M
    # trainable.  Pure torch, no bnb instability.
    #
    # Why not bitsandbytes 8-bit Adam:
    #   - paged_adamw_8bit: `d.is_cuda() INTERNAL ASSERT FAILED` at step 501
    #     resume (bnb 0.49.2 + torch 2.7.1 stream race)
    #   - adamw_8bit (non-paged): same crash at step 501 resume
    config.training.optim = "adafactor"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    # ⭐ The whole reason this wrapper exists: enable activation checkpointing
    # to fit on a single 24GB 4090. Trades ~30% throughput for ~half activation
    # memory.  Not exposed via FinetuneConfig CLI in upstream.
    #
    # Override: when LISA is enabled, we can DISABLE grad-ckpt because LISA
    # already shrinks per-step optimizer state to K/L of the trainable params.
    # This avoids the use_reentrant codepath that triggers `d.is_cuda()` asserts.
    _lisa_enabled = os.environ.get("LISA_ENABLE", "0") == "1"
    config.training.gradient_checkpointing = not _lisa_enabled
    if _lisa_enabled:
        print(f"[gr00t-train] LISA enabled — disabling gradient_checkpointing", flush=True)

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)
