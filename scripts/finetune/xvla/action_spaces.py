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

import torch
import torch.nn as nn

from lerobot.policies.xvla.action_hub import BaseActionSpace, register_action


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

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

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

        joints_loss = (
            self.mse(
                pred[:, :, list(self.JOINTS_IDX)],
                target[:, :, list(self.JOINTS_IDX)],
            )
            * self.JOINTS_SCALE
        )
        gripper_loss = (
            self.mse(
                pred[:, :, list(self.gripper_idx)],
                target[:, :, list(self.gripper_idx)],
            )
            * self.GRIPPER_SCALE
        )

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        # No special preprocessing — model sees zero-padded 20D directly.
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self._trim_to_real_dim(action)
