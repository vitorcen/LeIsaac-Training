"""LeIsaac SO-101 PickOrange modality config for Isaac-GR00T N1.6 finetune.

Registered into gr00t.configs.data.embodiment_configs at import-time so that
`launch_finetune.py --embodiment_tag NEW_EMBODIMENT` resolves to this config.

Structure mirrors examples/SO100/so100_config.py — our LeIsaac SO-101 dataset
has the exact same modality layout (5-DOF arm + 1-DOF gripper, front + wrist cams).
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


leisaac_so101_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
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

register_modality_config(leisaac_so101_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
