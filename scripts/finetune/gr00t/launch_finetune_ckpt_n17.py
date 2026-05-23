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


# CPU↔GPU pipeline parallelism: overlap H2D copy with previous step's forward/backward.
# Default HF Trainer._prepare_input uses `data.to(device)` without non_blocking=True,
# making H2D a synchronous step. With pin_memory=True (HF default) + non_blocking=True
# + a dedicated CUDA stream prefetching the NEXT batch, GPU never waits for H2D.
# See LeIsaac/docs/training/gpu_dataloader_zero_copy.html (必做 #2).
def _patch_cuda_pipeline_overlap():
    if os.environ.get("PIPELINE_OVERLAP_DISABLE", "0") == "1":
        return
    import torch
    from collections.abc import Mapping
    from transformers import Trainer

    _orig_prepare_input = Trainer._prepare_input

    def _patched_prepare_input(self, data):
        # Non-blocking H2D: needs pin_memory=True on dataloader (HF default).
        if isinstance(data, Mapping):
            return type(data)({k: _patched_prepare_input(self, v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(_patched_prepare_input(self, v) for v in data)
        elif isinstance(data, torch.Tensor):
            if data.device.type == "cpu" and self.args.device.type == "cuda":
                return data.to(self.args.device, non_blocking=True)
            return data.to(self.args.device)
        return data

    Trainer._prepare_input = _patched_prepare_input
    print("[gr00t-n17-train] patched Trainer._prepare_input → non_blocking=True (CPU↔GPU pipeline overlap)", flush=True)


_patch_cuda_pipeline_overlap()


# Bump dataloader prefetch_factor (HF default 2 → 4) so workers have more batches
# queued ahead of the trainer. Only effective when CPU has slack between worker bursts.
def _patch_prefetch_factor():
    if os.environ.get("PREFETCH_FACTOR_DISABLE", "0") == "1":
        return
    pf = int(os.environ.get("DATALOADER_PREFETCH_FACTOR", "4"))
    from gr00t.experiment.trainer import Gr00tTrainer

    _orig_get_loader = Gr00tTrainer.get_train_dataloader

    def _patched_get_loader(self):
        loader = _orig_get_loader(self)
        try:
            loader.prefetch_factor = pf
        except Exception as e:
            print(f"[gr00t-n17-train] cannot override prefetch_factor: {e}", flush=True)
        else:
            print(f"[gr00t-n17-train] dataloader prefetch_factor → {pf}", flush=True)
        return loader

    Gr00tTrainer.get_train_dataloader = _patched_get_loader


_patch_prefetch_factor()


# Optional CPU phase profiler — enable with PROFILE_PHASES=1 to print collator + get_vlm_inputs latency.
def _maybe_install_profile():
    if os.environ.get("PROFILE_PHASES", "0") != "1":
        return
    import time, statistics
    from collections import defaultdict
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator, Gr00tN1d7Processor

    times = defaultdict(list)
    _orig_collator = Gr00tN1d7DataCollator.__call__
    _orig_get_vlm = Gr00tN1d7Processor._get_vlm_inputs

    def _timed_collator(self, features):
        t0 = time.perf_counter()
        out = _orig_collator(self, features)
        times["collator(main)"].append((time.perf_counter() - t0) * 1000)
        if len(times["collator(main)"]) % 20 == 0:
            for phase, vals in times.items():
                v = vals[-50:]
                p90 = sorted(v)[max(0, int(0.9 * len(v)) - 1)]
                print(f"[profile] {phase}: n={len(vals)} mean50={statistics.mean(v):.1f}ms p90={p90:.1f}ms", flush=True)
        return out

    def _timed_get_vlm(self, image_keys, images, masks, image_transform, language):
        t0 = time.perf_counter()
        out = _orig_get_vlm(self, image_keys, images, masks, image_transform, language)
        times["get_vlm_inputs(worker)"].append((time.perf_counter() - t0) * 1000)
        return out

    Gr00tN1d7DataCollator.__call__ = _timed_collator
    Gr00tN1d7Processor._get_vlm_inputs = _timed_get_vlm
    print("[gr00t-n17-train] PROFILE_PHASES=1 → timing collator + _get_vlm_inputs", flush=True)


_maybe_install_profile()


# Heavy CPU work moved from collator (main thread) into per-worker:
# 本机 4090 实验 (2026-05-23) 证伪此优化 — CPU 已 100% 饱和时，把工作从主线程移到
# worker 等于让 4 worker 抢同一批 P-core，net 反而 GPU util 50→43%、wall +7%。
# Main-thread collator 和 GPU forward 通过 non_blocking H2D 已是天然重叠，没必要拆。
#
# **何时启用**: 当 micro_batch × n_cam ≥ 16 imgs / collator call 且主线程是单点瓶颈时
# （比如 H100 96GB 上跑 micro_batch=8, n_cam=2 — image_processor 43.8 ms / call）。
# 4090 配置 (micro_batch=2, n_cam=2 → 4 imgs/call) 不在此 sweet spot。
#
# 默认关闭；需要时设 COLLATOR_SPLIT_ENABLE=1 显式开。
def _patch_collator_split_image_proc():
    if os.environ.get("COLLATOR_SPLIT_ENABLE", "0") != "1":
        return
    import torch
    import numpy as np
    from PIL import Image
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
        Gr00tN1d7Processor,
        Gr00tN1d7DataCollator,
    )

    _orig_apply = Gr00tN1d7Processor._apply_vlm_processing

    def _patched_apply(self, images, language):
        # Per-sample worker path: do image_processor here in parallel via DataLoader workers.
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]
        conversation = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {"type": "text", "text": language},
            ],
        }]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        # Run image_processor per-sample (cheap: 0.8 ms / 1-img sample × N workers in parallel).
        image_inputs = self.processor.image_processor(images=pil_images)
        return {
            "vlm_content": {
                "text": text,
                "n_images": len(pil_images),
                "pixel_values_arr": np.asarray(image_inputs["pixel_values"]),
                "image_grid_thw_arr": np.asarray(image_inputs["image_grid_thw"]),
            }
        }

    _orig_collator = Gr00tN1d7DataCollator.__call__

    def _patched_collator(self, features):
        from transformers.feature_extraction_utils import BatchFeature
        # Fast path: features carry pre-processed pixel_values from workers.
        first_key_present = (
            "vlm_content" in features[0]
            and "pixel_values_arr" in features[0]["vlm_content"]
        )
        if not first_key_present:
            return _orig_collator(self, features)

        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))
        for key in keys:
            if key == "vlm_content":
                # Concat workers' pre-computed image features + run text replacement + tokenize.
                values = [elem[key] for elem in features if key in elem]
                proc = self.processor
                image_token = proc.image_token
                merge_length = proc.image_processor.merge_size ** 2

                texts = []
                pixel_values_list = []
                image_grid_thw_list = []
                for v in values:
                    text_i = v["text"]
                    pv = v["pixel_values_arr"]
                    grid = v["image_grid_thw_arr"]
                    # Replace each <image> placeholder with the right number of image tokens.
                    text_chars = []
                    grid_idx = 0
                    while image_token in text_i and grid_idx < grid.shape[0]:
                        n_tok = int(grid[grid_idx].prod()) // merge_length
                        text_i = text_i.replace(image_token, "<|placeholder|>" * n_tok, 1)
                        grid_idx += 1
                    text_i = text_i.replace("<|placeholder|>", image_token)
                    texts.append(text_i)
                    pixel_values_list.append(torch.as_tensor(pv))
                    image_grid_thw_list.append(torch.as_tensor(grid))

                tokenized = proc.tokenizer(text=texts, return_tensors="pt", padding=True)
                batch.update({k: v for k, v in tokenized.items()})
                batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
                batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                # Should not appear at sample level when fast path is active.
                continue
            else:
                values = [elem[key] for elem in features if key in elem]
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    Gr00tN1d7Processor._apply_vlm_processing = _patched_apply
    Gr00tN1d7DataCollator.__call__ = _patched_collator
    print("[gr00t-n17-train] patched _apply_vlm_processing (worker) + __call__ (collator) — image_proc → worker", flush=True)


_patch_collator_split_image_proc()


# torch.compile on action_head.forward — Isaac-GR00T's deployment benchmark uses this
# (`scripts/deployment/benchmark_inference.py:542`); replicating in training is low-risk.
# Expected +5-10% on top of bf16 + larger micro_batch (Codex + Opencode review consensus).
# Default ON; set COMPILE_ACTION_HEAD_DISABLE=1 to skip.
def _patch_compile_action_head():
    if os.environ.get("COMPILE_ACTION_HEAD_DISABLE", "0") == "1":
        return
    import torch
    from gr00t.experiment.trainer import Gr00tTrainer

    _orig_init = Gr00tTrainer.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            head = self.model.action_head.model
            if hasattr(head, "forward"):
                head.forward = torch.compile(
                    head.forward,
                    mode=os.environ.get("COMPILE_MODE", "reduce-overhead"),
                    dynamic=False,
                )
                print(f"[gr00t-n17-train] torch.compile(action_head.forward, mode={os.environ.get('COMPILE_MODE','reduce-overhead')})", flush=True)
        except Exception as e:
            print(f"[gr00t-n17-train] compile skipped: {e}", flush=True)

    Gr00tTrainer.__init__ = _patched_init


_patch_compile_action_head()


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
