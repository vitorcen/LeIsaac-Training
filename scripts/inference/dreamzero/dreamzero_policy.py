"""DreamZero LeIsaac inference policy.

Builds a GrootSimPolicy-equivalent that:
  - Uses our NF4+LoRA pre-loaded model (via dreamzero_inference_loader)
  - Bypasses GrootSimPolicy.__init__ (which tries FP16 full load → OOM on 4090)
  - Reuses GrootSimPolicy's apply/unapply/lazy_joint_forward_causal for transforms + AR loop
  - Exposes `.infer(obs)` matching the LeIsaac SO-101 modality format

Usage:
    policy = DreamZeroLeIsaacPolicy(ckpt_dir, cfg_dir, wan_dir)
    actions = policy.infer({
        "video.front": np.uint8 (H, W, 3),
        "video.wrist": np.uint8 (H, W, 3),
        "state.joint_pos": np.float32 (5,),
        "state.gripper_pos": np.float32 (1,),
        "task": "Pick up the orange and place it in the bowl.",
    })
    # → {"action.joint_pos": np.float32 (T, 5), "action.gripper_pos": np.float32 (T, 1)}
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, "/home/david/work/dreamzero-repo")
sys.path.insert(0, str(Path(__file__).parent))

from hydra.utils import instantiate
from tianshou.data import Batch

from groot.vla.data.schema import DatasetMetadata
from groot.vla.data.schema.embodiment_tags import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

from dreamzero_inference_loader import build_dreamzero_inference_model


logger = logging.getLogger(__name__)


def _init_single_gpu_distributed():
    """Init a single-GPU process group so GrootSimPolicy.dist.get_rank() works."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    try:
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    except Exception as e:
        # nccl may fail without GPU env; gloo is a safe fallback for single-process inference
        print(f"[policy] nccl init failed ({e}); trying gloo", flush=True)
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    torch.cuda.set_device(0)


def _build_fake_groot_sim_policy(trained_model, full_cfg, metadata_dict, embodiment_str: str = "xdof"):
    """Construct a GrootSimPolicy without going through __init__ (which would OOM on FP16 14B).
    Fill in the attributes that apply/unapply/lazy_joint_forward_causal actually need.
    """
    policy = GrootSimPolicy.__new__(GrootSimPolicy)

    # Manually call BaseTianshouPolicy.__init__ minimum, skip the heavy parent
    import torch.nn as nn
    nn.Module.__init__(policy)

    policy.trained_model = trained_model
    policy.train_cfg = full_cfg
    policy.embodiment_tag = EmbodimentTag(embodiment_str)
    policy.rank = 0
    policy.lazy_load = False
    policy.model_target = full_cfg.model._target_
    policy.model_dir = None
    policy.eval_bf16 = full_cfg.get("eval_bf16", True)
    policy.device = "cuda:0"

    # Metadata + per-embodiment slice
    meta_obj = DatasetMetadata.model_validate(metadata_dict[embodiment_str])

    # Adjust video resolution to model's expected target (if set on action_head.config)
    try:
        ah_cfg = trained_model.action_head.config
        target_h = getattr(ah_cfg, "target_video_height", None)
        target_w = getattr(ah_cfg, "target_video_width", None)
        if target_h is not None and target_w is not None and meta_obj.modalities.video:
            for key in meta_obj.modalities.video.keys():
                meta_obj.modalities.video[key].resolution = (int(target_w), int(target_h))
    except Exception:
        pass

    # Build eval transform for this embodiment
    assert embodiment_str in full_cfg.transforms, \
        f"embodiment '{embodiment_str}' not in transforms; have {list(full_cfg.transforms.keys())}"
    # Rewrite cloud paths in any DreamTransform tokenizer_path → local UMT5 cache
    from omegaconf import OmegaConf
    transforms_cfg = full_cfg.transforms[embodiment_str]
    def _rewrite_tokenizer_paths(node):
        if isinstance(node, (dict,)) or hasattr(node, "keys"):
            for k in list(node.keys()):
                if k == "tokenizer_path" and isinstance(node[k], str):
                    p = node[k]
                    is_local_ok = False
                    try:
                        is_local_ok = Path(p).exists()
                    except (PermissionError, OSError):
                        is_local_ok = False
                    if not is_local_ok:
                        local_default = "/home/david/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279/google/umt5-xxl"
                        print(f"[policy] rewrite tokenizer_path {p} → {local_default}", flush=True)
                        node[k] = local_default
                else:
                    _rewrite_tokenizer_paths(node[k])
        elif isinstance(node, list) or (hasattr(node, "__iter__") and not isinstance(node, str)):
            try:
                for item in node:
                    _rewrite_tokenizer_paths(item)
            except TypeError:
                pass
    _rewrite_tokenizer_paths(transforms_cfg)
    eval_transform = instantiate(transforms_cfg)
    eval_transform.set_metadata(meta_obj)

    # PerHorizonActionTransform stats if needed
    relative_per_horizon = full_cfg.get("relative_action_per_horizon", False)
    if relative_per_horizon:
        try:
            action_stats = meta_obj.statistics.action
            per_horizon = {}
            for action_key in action_stats:
                key_stats = action_stats[action_key]
                if hasattr(key_stats, "model_dump"):
                    per_horizon[action_key] = key_stats.model_dump()
                else:
                    per_horizon[action_key] = key_stats
            eval_transform.set_per_horizon_statistics(per_horizon)
        except Exception as e:
            print(f"[policy] per_horizon stats setup skipped: {e}")

    eval_transform.eval()
    policy.eval_transform = eval_transform

    # Modality configs
    if embodiment_str in full_cfg.modality_configs:
        policy.modality_configs = instantiate(full_cfg.modality_configs[embodiment_str])
    else:
        policy.modality_configs = instantiate(full_cfg.modality_configs)
    policy._video_delta_indices = np.array(policy.modality_configs.video.eval_delta_indices)
    policy._video_horizon = len(policy._video_delta_indices)
    if "state" in policy.modality_configs:
        policy._state_delta_indices = np.array(policy.modality_configs.state.eval_delta_indices)
        policy._state_horizon = len(policy._state_delta_indices)
    else:
        policy._state_delta_indices = None
        policy._state_horizon = None
    policy._raw_data_image_transform = None

    return policy


class DreamZeroLeIsaacPolicy:
    """Top-level inference policy for LeIsaac SO-101 PickOrange using DreamZero VLA."""

    def __init__(self, ckpt_dir, cfg_dir, wan_snap_dir, embodiment_str: str = "xdof"):
        _init_single_gpu_distributed()

        print(f"[policy] Building DreamZero VLA from ckpt={ckpt_dir}", flush=True)
        trained_model, full_cfg, metadata_dict = build_dreamzero_inference_model(
            ckpt_dir=ckpt_dir,
            experiment_cfg_dir=cfg_dir,
            wan_snap_dir=wan_snap_dir,
        )

        print(f"[policy] Wrapping in fake GrootSimPolicy (embodiment={embodiment_str})", flush=True)
        self._sim_policy = _build_fake_groot_sim_policy(
            trained_model, full_cfg, metadata_dict, embodiment_str=embodiment_str
        )

        self.embodiment_str = embodiment_str
        self._call_count = 0
        self._is_first_call = True
        self._current_session_id: str | None = None

        # Frame accumulation across calls (mirrors ARDroidRoboarenaPolicy.FRAMES_PER_CHUNK)
        # For LeIsaac SO-101 we keep last N=1 frames since num_frame_per_block=2 — handled
        # internally by action_head. Single frame per call is sufficient.
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.front": [],
            "video.wrist": [],
        }

        # Pre-warm with the standard PickOrange prompt so UMT5 (~11GB) loads ONCE before
        # Isaac Sim starts using the GPU. After this, text encoder is freed → peak drops
        # ~22GB → ~12GB, leaving headroom for sim share.
        self._prewarm("Pick up the orange and place it in the bowl.")

    def _prewarm(self, prompt: str) -> None:
        print(f"[policy] pre-warming with prompt: {prompt!r}", flush=True)
        dummy = {
            "video.front": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
            "video.wrist": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
            "state.joint_pos": np.zeros(5, dtype=np.float32),
            "state.gripper_pos": np.zeros(1, dtype=np.float32),
            "annotation.task": prompt,
        }
        _ = self.infer(dummy)
        self.reset()
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 1e9
        cur = torch.cuda.memory_allocated() / 1e9
        print(f"[policy] pre-warm done; cur={cur:.2f}GB peak_during={peak:.2f}GB", flush=True)
        torch.cuda.reset_peak_memory_stats()

    @property
    def action_horizon(self) -> int:
        return self._sim_policy.trained_model.action_head.action_horizon

    def reset(self) -> None:
        """Reset state for new episode."""
        for k in self._frame_buffers:
            self._frame_buffers[k] = []
        self._is_first_call = True
        self._current_session_id = None
        # Reset action_head autoregressive state
        ah = self._sim_policy.trained_model.action_head
        if hasattr(ah, "current_start_frame"):
            ah.current_start_frame = 0
        if hasattr(ah, "language"):
            ah.language = None
        if hasattr(ah, "clip_feas"):
            ah.clip_feas = None
        if hasattr(ah, "ys"):
            ah.ys = None
        # Invalidate our episode-scope CLIP cache so first call of next episode re-encodes
        if hasattr(ah, "_episode_clip_cache"):
            ah._episode_clip_cache = None
        print("[policy] reset", flush=True)

    def _convert_observation(self, obs: dict) -> dict:
        """Convert LeIsaac obs dict → GR00T modality batch (per-call frame accumulation).

        Input keys:
            video.front: (H, W, 3) uint8 single frame  OR  (T, H, W, 3) batched frames
            video.wrist: (H, W, 3) uint8 single frame  OR  (T, H, W, 3) batched frames
            state.joint_pos: (5,) float32
            state.gripper_pos: (1,) float32
            task / annotation.task: str
        Output (GR00T batch format):
            video.front: (T, H, W, 3)
            video.wrist: (T, H, W, 3)
            state.joint_pos: (1, 5)
            state.gripper_pos: (1, 1)
            annotation.task: [str]
        """
        out = {}
        for k in ("video.front", "video.wrist"):
            if k in obs:
                data = np.asarray(obs[k])
                if data.ndim == 4:
                    self._frame_buffers[k].extend(list(data))
                else:
                    self._frame_buffers[k].append(data)

        # Keep at 1 frame — 4-frame mode triggers a 9-frame VAE re-encode path that bloats
        # memory. Instead we monkey-patch action_head.lazy_joint_video_action's reset trigger
        # so single-frame inputs DON'T force CLIP re-encode every chunk.
        num_frames = 1
        for k, buf in self._frame_buffers.items():
            if not buf:
                continue
            frames = buf[-num_frames:]
            while len(frames) < num_frames:
                frames.insert(0, buf[0])
            out[k] = np.stack(frames, axis=0)

        # State
        joint = np.asarray(obs.get("state.joint_pos", np.zeros(5, dtype=np.float32)), dtype=np.float64)
        if joint.ndim == 1:
            joint = joint.reshape(1, -1)
        out["state.joint_pos"] = joint
        grip = np.asarray(obs.get("state.gripper_pos", np.zeros(1, dtype=np.float32)), dtype=np.float64)
        if grip.ndim == 1:
            grip = grip.reshape(1, -1)
        out["state.gripper_pos"] = grip

        # Task text
        task = obs.get("annotation.task", obs.get("task", ""))
        if isinstance(task, list):
            task = task[0] if task else ""
        out["annotation.task"] = task

        return out

    def _convert_action(self, act_dict: dict) -> dict:
        """Convert GrootSimPolicy unapply output → LeIsaac action dict (T, dim).

        Input shape per key may be (B, T, D), (T, D), or even (B,T,D) numpy ndarray;
        normalize to (T, D).
        """
        def _to_TD(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            # Drop a leading batch dim of size 1 (don't blanket squeeze — would squash horizon=1)
            if x.ndim >= 3 and x.shape[0] == 1:
                x = x[0]
            # Ensure (T, D); rank 1 → (T, 1)
            if x.ndim == 1:
                x = x[:, None]
            return x.astype(np.float32)
        return {
            "action.joint_pos": _to_TD(act_dict.get("action.joint_pos")),
            "action.gripper_pos": _to_TD(act_dict.get("action.gripper_pos")),
        }

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        """Predict an action chunk for the current obs. Caller should call `reset()` between episodes."""
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                self.reset()
            self._current_session_id = session_id

        self._call_count += 1
        converted = self._convert_observation(obs)
        batch = Batch(obs=converted)

        result_batch, video_pred = self._sim_policy.lazy_joint_forward_causal(batch)
        act_dict = {}
        for k in dir(result_batch.act):
            if k.startswith("action."):
                act_dict[k] = getattr(result_batch.act, k)
        result = self._convert_action(act_dict)

        # After the first call the prompt is cached → free UMT5 (~11GB) so we leave headroom
        # for Isaac Sim sharing this GPU. Cached embeddings keep working.
        if self._is_first_call:
            ah = self._sim_policy.trained_model.action_head
            if hasattr(ah, "free_text_encoder") and ah.text_encoder is not None:
                ah.free_text_encoder()
            self._is_first_call = False
        return result


# ----- CLI smoke test -----

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cfg-dir", required=True)
    ap.add_argument("--wan-dir", default=str(
        Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots"
    ))
    args = ap.parse_args()

    wan_dir = Path(args.wan_dir)
    if not (wan_dir / "diffusion_pytorch_model.safetensors.index.json").exists():
        wan_dir = next(wan_dir.glob("*"))

    policy = DreamZeroLeIsaacPolicy(args.ckpt_dir, args.cfg_dir, wan_dir)

    # Synthetic obs: 1 frame each of front + wrist (640x480 = native LeIsaac camera res;
    # transforms downscale to 320x176 model input)
    import time
    obs = {
        "video.front": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "video.wrist": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "state.joint_pos": np.array([0.0, -0.4, 0.5, 0.0, 0.0], dtype=np.float32),
        "state.gripper_pos": np.array([0.0], dtype=np.float32),
        "annotation.task": "Pick up the orange and place it in the bowl.",
    }
    print(f"\n[smoke] First inference (encodes everything)...", flush=True)
    t0 = time.perf_counter()
    out = policy.infer(obs)
    t1 = time.perf_counter()
    print(f"[smoke] First call took {t1-t0:.2f}s", flush=True)
    print(f"[smoke] joint_pos: shape={out['action.joint_pos'].shape if out['action.joint_pos'] is not None else None}", flush=True)
    print(f"[smoke] gripper_pos: shape={out['action.gripper_pos'].shape if out['action.gripper_pos'] is not None else None}", flush=True)
    if out["action.joint_pos"] is not None:
        a = out["action.joint_pos"]
        print(f"[smoke] joint sample: min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f} std={a.std():.3f}", flush=True)
        print(f"[smoke] NaN? joint={np.isnan(a).any()}", flush=True)

    print(f"\n[smoke] Second inference (cached caches)...", flush=True)
    t0 = time.perf_counter()
    out = policy.infer(obs)
    t1 = time.perf_counter()
    print(f"[smoke] Second call took {t1-t0:.2f}s", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[smoke] Peak VRAM: {peak:.2f} GB / 24 GB", flush=True)


if __name__ == "__main__":
    main()
