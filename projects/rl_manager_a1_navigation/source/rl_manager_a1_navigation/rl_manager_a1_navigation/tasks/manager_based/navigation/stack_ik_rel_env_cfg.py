# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
SO-ARM-101 Stack Task with Camera Observations and OpenXR Hand Tracking.

This configuration extends the camera-based environment with OpenXR VR hand tracking
for teleoperation and binary gripper control.
"""

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg
# from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass
from so_arm_mimic.source.envs.stack_ik_abs_camera_mimic_env_cfg import SO101CubeStackCameraMimicEnvCfg

from controll_scripts.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from controll_scripts.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from so_arm_mimic.source import mdp as so_arm_mimic_mdp


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Reaching Cube 1
    reaching_cube_1 = RewTerm(
        func=so_arm_mimic_mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube_1")},
        weight=1.0
    )
    # Lifting Cube 1
    lifting_cube_1 = RewTerm(
        func=so_arm_mimic_mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cube_1")},
        weight=2.0
    )
    # Stacking Success (Sparse)
    # Using the existing observation function as a reward
    stacking_success = RewTerm(
        func=so_arm_mimic_mdp.object_stacked,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "upper_object_cfg": SceneEntityCfg("cube_2"), # or cube to be stacked
            "lower_object_cfg": SceneEntityCfg("cube_1"), # or base cube
        },
        weight=10.0 # High reward for success
    )
    # Penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class SO101CubeStackRelEnvCfg(SO101CubeStackCameraMimicEnvCfg):
    """
    SO-ARM-101 Stack Task with Camera Observations and OpenXR Hand Tracking.
    
    Inherits from Camera Mimic Config but:
    - Changes teleop device to OpenXR hand tracking
    - Changes gripper action to binary (open/close)
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="adaptive",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
        self.scene.num_envs = 1

        self.rewards = RewardsCfg()
        