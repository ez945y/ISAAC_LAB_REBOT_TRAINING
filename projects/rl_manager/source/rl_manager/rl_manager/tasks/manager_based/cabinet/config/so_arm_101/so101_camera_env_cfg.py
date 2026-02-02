# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Cabinet Task with Camera Observations.

This configuration removes ground-truth cabinet joint positions/velocities 
and replaces them with camera observations for visual-based policy learning.
"""

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from rl_manager.tasks.manager_based.cabinet import mdp
from rl_manager.tasks.manager_based.cabinet.config.so_arm_101.mobilenet_feature_extractor import (
    MobileNetFeatureExtractor,
    MobileNetFeatureExtractorCfg,
)
from .so101_rel_env_cfg import SOArm101CabinetEnvCfg

@configclass
class CameraCfg(ObsGroup):
    """Observations for policy group."""
    front_image = ObsTerm(
        func=mdp.image_features,
        params={
            "sensor_cfg": SceneEntityCfg("front_camera"),
            "data_type": "rgb",
            "model_name": "theia-tiny-patch16-224-cddsv",
            "model_device": "cuda:0",
        },
    )
    top_image = ObsTerm(
        func=mdp.image_features,
        params={
            "sensor_cfg": SceneEntityCfg("top_camera"),
            "data_type": "rgb",
            "model_name": "theia-tiny-patch16-224-cddsv",
            "model_device": "cuda:0",
        },
    )


@configclass
class SOArm101CameraCabinetEnvCfg(SOArm101CabinetEnvCfg):
    """
    SO-ARM-101 Cabinet Task with Camera Observations using Pretrained ResNet.
    
    Inherits from SOArm101CabinetEnvCfg but:
    - Removes ground-truth cabinet_joint_pos, cabinet_joint_vel, rel_ee_drawer_distance (no cheating)
    - Adds camera CNN embedding observations using TiledCamera and PretrainedFeatureExtractor (ResNet18)
    - Uses ImageNet pretrained weights, no CNN training needed
    """

    feature_extractor_cfg = MobileNetFeatureExtractorCfg(
        freeze_backbone=True,
        embedding_dim=128,
        use_fp16=True
    )

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 256
        # self.scene.wrist_camera = TiledCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/gripper_link/self_view_camera",
        #     update_period=1/30,  # 30Hz camera update
        #     height=320,
        #     width=240,
        #     data_types=["rgb"],
        #     spawn=None,
        # )
        self.scene.top_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/top_view_camera",
            update_period=1/30,  # 30Hz camera update
            height=320,
            width=240,
            data_types=["rgb"],
            spawn=None,
        )

        # Fixed camera (Full View)
        self.scene.front_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/full_view_camera",
            update_period=1/30,  # 30Hz camera update
            height=320,
            width=240,
            data_types=["rgb"],
            spawn=None,
        )
        self.observations.camera = CameraCfg()


@configclass
class SOArm101CameraCabinetEnvCfg_PLAY(SOArm101CameraCabinetEnvCfg):
    """Play configuration with reduced environment count."""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
