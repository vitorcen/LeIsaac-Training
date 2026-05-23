"""N1.7 wrapper around Isaac-GR00T launch_finetune.py.

Difference vs launch_finetune_ckpt.py (N1.6):
- DO NOT override model_name — upstream default Gr00tN1d7Config already sets
  `nvidia/Cosmos-Reason2-2B` as the VLM backbone for N1.7.
- DO NOT override `backbone_trainable_params_fp32` — upstream N1.7 default is
  True; N1.6 had to flip it to False for 4090 24GB; for N1.7 we keep True
  initially and only flip if smoke OOMs.
- Keep adafactor + grad_ckpt + checkpoint prune + use_reentrant=False patches.

Inference / training path:
- `--base-model-path nvidia/Cosmos-Reason2-2B`   → Path 1 (replicate hi-space, cold-start action head)
- `--base-model-path hi-space/GR00T-N1.7-3B-Pick-Orange` → Path 2 (warm-start from 14/15 SOTA)

CLI is identical to launch_finetune.py (tyro on FinetuneConfig).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import tyro

_GR00T_ROOT = Path(os.environ.get("GR00T_ROOT", str(Path.home() / "work" / "Isaac-GR00T")))
if str(_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(_GR00T_ROOT))

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run
from gr00t.experiment.launch_finetune import load_modality_config


# Force `use_reentrant=False` on gradient-checkpointing (same justification as N1.6 wrapper).
def _patch_gradient_checkpointing_default():
    from transformers import PreTrainedModel

    _orig = PreTrainedModel.gradient_checkpointing_enable

    def _patched(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        gradient_checkpointing_kwargs.setdefault("use_reentrant", False)
        print(
            f"[gr00t-n17-train] patched gradient_checkpointing_enable kwargs = "
            f"{gradient_checkpointing_kwargs}",
            flush=True,
        )
        return _orig(self, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    PreTrainedModel.gradient_checkpointing_enable = _patched


_patch_gradient_checkpointing_default()


# Selective checkpoint pruning: keep multiples of KEEP_MULTIPLE permanent + last KEEP_TEMPORARY others.
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
        print(f"[gr00t-n17-train] CheckpointPruneCallback installed (keep_mult={keep_mult}, keep_temp={keep_temp})", flush=True)

    Gr00tTrainer.__init__ = _patched_trainer_init


_patch_save_pruning()


# Loss-driven checkpoint pruning: keep top-K ckpts by training loss + always keep last 1.
# AutoDL has no closed-loop sim → train_loss is the proxy signal for "which ckpt is worth keeping".
# Trade-off vs step-multiples CheckpointPruneCallback: this one is eval-driven (better signal)
# but train_loss can be noisy; multiples-based gives predictable disk usage.
# We run BOTH callbacks — multiples bound disk; loss-driven prunes the kept temp slots smarter.
def _patch_loss_driven_pruning():
    if os.environ.get("LOSS_PRUNE_DISABLE", "0") == "1":
        return
    import re, shutil
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer

    top_k = int(os.environ.get("LOSS_PRUNE_TOP_K", "5"))

    class LossDrivenPruneCallback(TrainerCallback):
        def __init__(self):
            self.ckpt_losses = {}  # step → train_loss at that save point

        def on_save(self, args, state, control, **kwargs):
            # Pull most recent training loss from log_history (HF Trainer logs after each step).
            for entry in reversed(state.log_history):
                if "loss" in entry and "step" in entry and entry["step"] == state.global_step:
                    self.ckpt_losses[state.global_step] = entry["loss"]
                    break
            if not self.ckpt_losses:
                return  # nothing to compare yet
            all_steps = sorted(self.ckpt_losses.keys())
            last_step = all_steps[-1]
            sorted_by_loss = sorted(self.ckpt_losses.items(), key=lambda x: x[1])
            keepers = {s for s, _ in sorted_by_loss[:top_k]}
            keepers.add(last_step)  # always keep most recent (in case it's the new best)
            out = args.output_dir
            if not os.path.isdir(out):
                return
            removed = []
            for d in os.listdir(out):
                m = re.match(r"^checkpoint-(\d+)$", d)
                if not m:
                    continue
                step = int(m.group(1))
                if step in keepers:
                    continue
                if step not in self.ckpt_losses:
                    continue  # don't touch ckpts we never observed (e.g. resume from disk)
                shutil.rmtree(os.path.join(out, d), ignore_errors=True)
                removed.append((step, self.ckpt_losses[step]))
            if removed:
                kept_summary = sorted([(s, self.ckpt_losses[s]) for s in keepers if s in self.ckpt_losses], key=lambda x: x[1])
                print(
                    f"[loss-prune] deleted {[(s, f'{l:.4f}') for s, l in removed]}  "
                    f"kept top{top_k}+last = {[(s, f'{l:.4f}') for s, l in kept_summary]}",
                    flush=True,
                )

    _orig_trainer_init = Gr00tTrainer.__init__

    def _patched_trainer_init(self, *args, **kwargs):
        _orig_trainer_init(self, *args, **kwargs)
        self.add_callback(LossDrivenPruneCallback())
        print(f"[gr00t-n17-train] LossDrivenPruneCallback installed (top_k={top_k})", flush=True)

    Gr00tTrainer.__init__ = _patched_trainer_init


_patch_loss_driven_pruning()


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
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
    # Match hi-space/GR00T-N1.7-3B-Pick-Orange recipe: freeze top LLM layers entirely.
    # Default upstream value (4) tunes 4 top transformer blocks; that interacts with
    # save_only_model=True to drop frozen-backbone keys → 698/1030 keys ckpt (broken).
    # Setting to 0 matches the public 14/15 SOTA ckpt structure (1030 keys, ~560M trainable).
    config.model.tune_top_llm_layers = 0
    # Keep model_name = "nvidia/Cosmos-Reason2-2B" (upstream default) — get_backbone_cls()
    # has a hardcoded whitelist that rejects local paths. For offline runs (no network),
    # populate the HF cache layout with a symlink:
    #   mkdir -p $HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B/{snapshots,refs}
    #   echo -n $SHA > .../refs/main
    #   ln -sfn /path/to/cosmos_raw .../snapshots/$SHA
    # Then transformers resolves the HF id to local files via cache, no network needed.
    config.model.use_relative_action = True

    # COLD START detect: when base_model_path is raw Cosmos backbone (model_type=qwen3_vl
    # OR HF id like "nvidia/Cosmos-Reason2-2B"), skip weight loading from checkpoint —
    # Gr00tN1d7Model auto-loads Cosmos backbone via model_name + randomly inits action head.
    # Warm start (e.g. hi-space/GR00T-N1.7-3B-Pick-Orange) keeps start_from_checkpoint.
    _is_cosmos_cold = False
    _bmp = ft_config.base_model_path or ""
    if isinstance(_bmp, str) and ("cosmos" in _bmp.lower() or "cosmos_raw" in _bmp):
        _is_cosmos_cold = True
    if os.path.isdir(_bmp):
        _cfg_json = os.path.join(_bmp, "config.json")
        if os.path.isfile(_cfg_json):
            try:
                import json as _json
                with open(_cfg_json) as _f:
                    _cfg = _json.load(_f)
                _mt = _cfg.get("model_type", "")
                if _mt.lower().startswith("qwen"):
                    _is_cosmos_cold = True
            except Exception:
                pass
    if _is_cosmos_cold:
        ft_config.skip_weight_loading = True
        # CRITICAL: also clear start_from_checkpoint so setup.py:_create_dataset takes the
        # else-branch (instantiates Gr00tN1d7Processor with set_statistics) instead of
        # AutoProcessor.from_pretrained which returns vanilla Qwen3VLProcessor lacking it.
        ft_config.base_model_path = None
        print(f"[gr00t-n17-train] COLD start from Cosmos backbone — start_from_checkpoint=None, skip_weight_loading=True", flush=True)

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    # Optimizer: small24 (4090) needs adafactor for memory; big48/big96 (PRO 6000 / H100 / A100)
    # can use adamw_torch for better convergence. Default keeps adafactor for safety.
    # paged_adamw_8bit was tried on 4090 N1.6 but bnb 0.49.2 crashes at step 501 resume.
    config.training.optim = os.environ.get("OPTIM", "adafactor")
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
    config.training.wandb_project = "finetune-gr00t-n1d7"

    # Activation checkpointing: required for 4090 24GB (trades ~30% throughput for ~half
    # activations). On big48/big96 GPUs we turn it OFF for the throughput.
    # Default True (safe); set GRADIENT_CKPT=0 to disable.
    config.training.gradient_checkpointing = os.environ.get("GRADIENT_CKPT", "1") == "1"
    print(f"[gr00t-n17-train] gradient_checkpointing={config.training.gradient_checkpointing} optim={config.training.optim}", flush=True)

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
