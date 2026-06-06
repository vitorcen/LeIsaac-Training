"""SO-101 PickOrange — data config + mixture (LeIsaac SO-101 single-arm, 6-DOF joint space).

Self-contained registry auto-discovered by
starVLA/dataloader/gr00t_lerobot/registry.py. Defines:
  * robot_type ``so101_pickorange`` -> SO101PickOrangeConfig
  * mixture    ``so101_pickorange`` -> our LeRobot dataset dir

Dataset: LeIsaac SO-101 PickOrange, LeRobot v2.1 (read via loader's v2.0 path),
6-DOF joints (shoulder_pan/lift, elbow_flex, wrist_flex/roll, gripper),
2 cameras (front=primary, wrist), task "Grab orange and place into plate".
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class SO101PickOrangeConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.shoulder_pan.pos", "state.shoulder_lift.pos", "state.elbow_flex.pos",
        "state.wrist_flex.pos", "state.wrist_roll.pos", "state.gripper.pos",
    ]
    action_keys = [
        "action.shoulder_pan.pos", "action.shoulder_lift.pos", "action.elbow_flex.pos",
        "action.wrist_flex.pos", "action.wrist_roll.pos", "action.gripper.pos",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))  # action_horizon = 16

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # All 6 joints (incl. gripper position) are continuous -> min_max normalization.
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={k: "min_max" for k in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={k: "min_max" for k in self.action_keys}),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "so101_pickorange": SO101PickOrangeConfig(),
}

# Empty: embodiment_tag is read from the DataConfig classvar by the registry.
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "so101_pickorange": [
        ("leisaac-pick-orange_old", 1.0, "so101_pickorange"),
    ],
}
