# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
SO-ARM-101 Stack Task with Camera Observations (No Cheating).

This configuration removes ground-truth cube positions/orientations 
and replaces them with camera observations for visual-based policy learning.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from so_arm_mimic.source import mdp as so_arm_mimic_mdp
from .stack_ik_abs_mimic_env_cfg import SO101CubeStackIKAbsMimicEnvCfg


@configclass
class SO101CubeStackCameraMimicEnvCfg(SO101CubeStackIKAbsMimicEnvCfg):
    """
    SO-ARM-101 Stack Task with Camera Observations.
    
    Inherits from IK Abs Mimic Config but:
    - Removes ground-truth cube_positions and cube_orientations (no cheating)
    - Adds wrist_camera and front_camera RGB observations
    """

    def __post_init__(self):
        super().__post_init__()
        
        # === Add Cameras to Scene ===
        # Cameras already exist in USD, so spawn=None
        
        # Hand-mounted camera (Self View) - already in USD
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/self_view_camera",
            update_period=0.1,  # 10Hz camera update
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=None,  # Camera already exists in USD
        )
        
        # Fixed camera (Full View) - already in USD
        self.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/full_view_camera",
            update_period=0.1,  # 10Hz camera update
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=None,  # Camera already exists in USD
        )
        
        self.observations.policy.concatenate_terms = True

        # self.observations.policy.cube_positions = None
        # self.observations.policy.cube_orientations = None
        

        # self.observations.policy.rgb_camera = None
        
        self.observations.policy.wrist_rgb = ObsTerm(
            func=so_arm_mimic_mdp.wrist_camera_rgb,
            params={"asset_cfg": SceneEntityCfg("wrist_camera")},
        )

        self.observations.policy.front_rgb = ObsTerm(
            func=so_arm_mimic_mdp.front_camera_rgb,
            params={"asset_cfg": SceneEntityCfg("front_camera")},
        )
        
        # Update datagen config name to reflect camera version
        self.datagen_config.name = "demo_src_stack_isaac_lab_camera_D0"