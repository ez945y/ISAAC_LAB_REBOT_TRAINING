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
from .stack_ik_camera_mimic_env_cfg import SO101CubeStackCameraMimicEnvCfg

from controll_scripts.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from controll_scripts.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from rl_manager.tasks.manager_based.cabinet.config.so_arm_101.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorCfg,
)

@configclass
class SO101CubeStackRelMimicEnvCfg(SO101CubeStackCameraMimicEnvCfg):
    """
    SO-ARM-101 Stack Task with Camera Observations and OpenXR Hand Tracking.
    
    Inherits from Camera Mimic Config but:
    - Changes teleop device to OpenXR hand tracking
    - Changes gripper action to binary (open/close)
    """
    feature_extractor_cfg: FeatureExtractorCfg = FeatureExtractorCfg(train=True, load_checkpoint=False)


    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls_5dof",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
        
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.25,
                    sim_device=self.sim.device,
                ),
            }
        )
        
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 1.7},
            close_command_expr={"gripper": 0.0},
        )
        
        self.datagen_config.name = "demo_src_stack_isaac_lab_rel_D0"

        # Enable debug visualization for the end-effector frame
        self.scene.ee_frame.debug_vis = False
