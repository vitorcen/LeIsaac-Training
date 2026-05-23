"""LeIsaac SO-101 PickOrange modality config for Isaac-GR00T N1.7 finetune.

Difference vs leisaac_config.py (N1.6):
- action delta_indices = range(40)  (was range(16))
  N1.7 Gr00tN1d7Config has action_horizon=40 hard default; data-side horizon
  must match (data loader computes horizon = max(delta)-min(delta)+1).
  hi-space/GR00T-N1.7-3B-Pick-Orange (current 14/15 SOTA) trained with this.

Same EmbodimentTag.NEW_EMBODIMENT registration — only one config can be live
at a time, so DO NOT import leisaac_config.py alongside this when running N1.7.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


leisaac_so101_n17_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(leisaac_so101_n17_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
