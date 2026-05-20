"""SingleArmSO101 X-VLA action_space registered into lerobot at import time.

This module is intentionally kept in our project (not in the lerobot submodule)
so the lerobot fork can be cleanly upstream-rebased without merge conflicts.

Importing this module BEFORE lerobot_train / XVLAPolicy.from_pretrained adds
``so101_single`` to ``lerobot.policies.xvla.action_hub.ACTION_REGISTRY``.

Usage from train.sh / server:
    python -c "import leisaac_xvla_actions; ..."
or via the small wrappers:
    LeIsaac/scripts/finetune/xvla/train_entry.py
    server/xvla_leisaac/server.py
both ``import action_spaces  # noqa: F401`` near the top.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from lerobot.policies.xvla.action_hub import BaseActionSpace, register_action

# Velocity-aware loss reweighting (AttenA+, 2605.13548).  Toggle via env var
# `XVLA_VELOCITY_REWEIGHT=1` so we don't need a separate registry entry.
# beta controls strength: w = exp(-beta * v/v_max).  Higher beta = more emphasis
# on low-velocity (precise / contact) frames.
_VEL_REWEIGHT = os.environ.get("XVLA_VELOCITY_REWEIGHT", "0") == "1"
_VEL_BETA = float(os.environ.get("XVLA_VELOCITY_BETA", "2.0"))

# OFT-lite (Fine-Tuning VLA 2502.19645): L1 regression on continuous actions
# instead of MSE.  OpenVLA-OFT showed +20.6 abs on LIBERO.
# Toggle via env var `XVLA_L1_LOSS=1`.  Mutually exclusive with velocity-reweight
# in compute_loss (L1 path takes precedence if both set).
_L1_LOSS = os.environ.get("XVLA_L1_LOSS", "0") == "1"


@register_action("so101_single")
class SingleArmSO101ActionSpace(BaseActionSpace):
    """
    Single-arm SO-101: 5 joints + 1 gripper.

    Layout (real robot):
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

    Real action dim: 6
    Model-facing dim: 20 (channels 6..19 are dummy zero padding to preserve the
    pretrained xvla-base head shape; gradients on those channels carry no info).

    Why a dedicated subclass (vs the generic ``auto``):
        - explicit gripper_idx=(5,) → gripper-specific loss weight,
          so the model commits to open/close rather than averaging across
          pick/place modes (the symptom we saw at ckpt-10k/20k of action_mode=
          auto: grab-but-no-place / mid-air jitter).
        - explicit JOINTS_IDX=(0..4) → joint MSE separated from gripper,
          easier to balance per-channel.
    """

    dim_action = 20
    REAL_DIM = 6
    gripper_idx = (5,)
    JOINTS_IDX = (0, 1, 2, 3, 4)
    GRIPPER_SCALE = 5.0  # bumped 3→5 per codex/opencode review
    JOINTS_SCALE = 1.0
    VELOCITY_REWEIGHT = _VEL_REWEIGHT
    VEL_BETA = _VEL_BETA
    L1_LOSS = _L1_LOSS

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        if self.VELOCITY_REWEIGHT:
            print(
                f"[so101_single] velocity-reweight ON, beta={self.VEL_BETA}",
                flush=True,
            )
        if self.L1_LOSS:
            print("[so101_single] L1 loss ON (OFT-lite)", flush=True)

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """6D → 20D zero-pad (or pass-through if already 20D)."""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"SingleArmSO101: expected last dim {self.REAL_DIM} or {self.dim_action}, "
                f"got {x.size(-1)}"
            )
        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.REAL_DIM]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self.REAL_DIM]

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape, (
            f"SingleArmSO101 shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

        if not self.VELOCITY_REWEIGHT:
            loss_fn = self.l1 if self.L1_LOSS else self.mse
            joints_loss = (
                loss_fn(
                    pred[:, :, list(self.JOINTS_IDX)],
                    target[:, :, list(self.JOINTS_IDX)],
                )
                * self.JOINTS_SCALE
            )
            gripper_loss = (
                loss_fn(
                    pred[:, :, list(self.gripper_idx)],
                    target[:, :, list(self.gripper_idx)],
                )
                * self.GRIPPER_SCALE
            )
            return {"joints_loss": joints_loss, "gripper_loss": gripper_loss}

        # Velocity-aware reweighting (AttenA+, 2605.13548).
        # target shape: (B, T, dim_action).  Compute per-step velocity from
        # target joints (excludes gripper which has its own scaling).
        tj = target[..., list(self.JOINTS_IDX)]  # (B, T, |joints|)
        vel = torch.zeros(target.shape[:2], device=target.device, dtype=target.dtype)
        vel[:, 1:] = (tj[:, 1:] - tj[:, :-1]).norm(dim=-1)  # (B, T)
        # Per-batch-element normalization → weights independent of absolute scale.
        vel_max = vel.amax(dim=1, keepdim=True).clamp(min=1e-6)
        w = torch.exp(-self.VEL_BETA * vel / vel_max)  # (B, T), low vel = high w
        # Normalize so weighted average = unweighted average on uniform-velocity inputs.
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-6)

        # Per-step joints squared error, weighted average.
        pj = pred[..., list(self.JOINTS_IDX)]
        sq_j = (pj - tj).pow(2).mean(dim=-1)  # (B, T)
        joints_loss = (sq_j * w).mean() * self.JOINTS_SCALE

        # Same weighting on gripper.
        tg = target[..., list(self.gripper_idx)]
        pg = pred[..., list(self.gripper_idx)]
        sq_g = (pg - tg).pow(2).mean(dim=-1)  # (B, T)
        gripper_loss = (sq_g * w).mean() * self.GRIPPER_SCALE

        return {"joints_loss": joints_loss, "gripper_loss": gripper_loss}

    def preprocess(self, proprio, action, mode="train"):
        # No special preprocessing — model sees zero-padded 20D directly.
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self._trim_to_real_dim(action)
