"""Modality config for BEHAVIOR-1K 2025 challenge demos (Galaxea R1 Pro).

Dataset: behavior-1k/2025-challenge-demos (LeRobot v2.1) + custom meta/modality.json
that slices the packed observation.state[256] / action[23] vectors using the
index layout from OmniGibson learning/utils/eval_utils.py (standard-track-legal keys).

Action layout (23 dims): base_vel(3) + torso(4) + left_arm(7) + left_gripper(1)
                         + right_arm(7) + right_gripper(1)
All actions are absolute joint/velocity commands as recorded by the challenge
teleop pipeline, so every key uses ActionRepresentation.ABSOLUTE.
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

_ABS_JOINT = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

b1k_r1pro_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "left_wrist", "right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "base_qvel",
            "trunk_qpos",
            "left_arm_qpos",
            "left_gripper_qpos",
            "right_arm_qpos",
            "right_gripper_qpos",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "base",
            "torso",
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        action_configs=[
            _ABS_JOINT,  # base (velocity command)
            _ABS_JOINT,  # torso
            _ABS_JOINT,  # left_arm
            _ABS_JOINT,  # left_gripper
            _ABS_JOINT,  # right_arm
            _ABS_JOINT,  # right_gripper
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(b1k_r1pro_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
