"""Thin wrapper that registers our custom action_spaces, then hands control to
lerobot_train.  Use this as the module entrypoint instead of
``lerobot.scripts.lerobot_train`` so we don't have to modify the lerobot
submodule to register ``so101_single``.

Usage:
    python -m train_entry --policy.path=... [...lerobot_train flags...]

The PYTHONPATH must include the directory containing this file
(``LeIsaac/scripts/finetune/xvla/``), which our ``train.sh`` arranges.
"""
from __future__ import annotations

import runpy

# Side-effect import: registers SingleArmSO101ActionSpace into
# lerobot.policies.xvla.action_hub.ACTION_REGISTRY.
import action_spaces  # noqa: F401


if __name__ == "__main__":
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__", alter_sys=True)
