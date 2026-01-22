# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Cabinet Task with Camera Observations.

This configuration removes ground-truth cabinet joint positions/velocities 
and replaces them with camera observations for visual-based policy learning.
"""

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from rl_manager.tasks.manager_based.cabinet import mdp
from rl_manager.tasks.manager_based.cabinet.config.so_arm_101.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorCfg,
)
from .so101_abs_env_cfg import SOArm101CabinetEnvCfg


@configclass
class SOArm101CameraCabinetEnvCfg(SOArm101CabinetEnvCfg):
    """
    SO-ARM-101 Cabinet Task with Camera Observations.
    
    Inherits from SOArm101CabinetEnvCfg but:
    - Removes ground-truth cabinet_joint_pos, cabinet_joint_vel, rel_ee_drawer_distance (no cheating)
    - Adds wrist_camera CNN embedding observations using TiledCamera and FeatureExtractor
    """

    # Feature extractor configuration
    feature_extractor_cfg: FeatureExtractorCfg = FeatureExtractorCfg(train=True, load_checkpoint=False)

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 400
    
        # Wrist camera - includes depth and segmentation for CNN
        self.scene.wrist_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/self_view_camera",
            update_period=0.1,  # 10Hz camera update
            height=120,  # Match FeatureExtractor expected input size
            width=120,
            data_types=["rgb", "depth", "semantic_segmentation"],
            spawn=None,
        )
        
        # # Fixed camera (Full View) - includes depth and segmentation for CNN
        # self.scene.front_camera = TiledCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/full_view_camera",
        #     update_period=0.1,  # 10Hz camera update
        #     height=120,  # Match FeatureExtractor expected input size
        #     width=120,
        #     data_types=["rgb", "depth", "semantic_segmentation"],
        #     spawn=None,  # Camera already exists in USD
        # )
        
        # Remove ground-truth observations (no cheating)
        self.observations.policy.cabinet_joint_pos = None
        self.observations.policy.cabinet_joint_vel = None
        self.observations.policy.rel_ee_drawer_distance = None
        
        # Add camera CNN embedding observations (27 dim each)
        self.observations.policy.wrist_embedding = ObsTerm(
            func=mdp.wrist_camera_embedding,
            params={"asset_cfg": SceneEntityCfg("wrist_camera")},
        )


class SOArm101CameraCabinetEnv(ManagerBasedRLEnv):
    """SO-ARM-101 Cabinet Environment with CNN-based camera observations."""
    
    cfg: SOArm101CameraCabinetEnvCfg

    def load_managers(self):
        """Load managers with feature extractor initialized first.
        
        The feature extractor must be initialized before the observation manager
        because observation functions need to access it to compute embedding shapes.
        """
        log_dir = getattr(self.cfg, 'log_dir', None)
        self.feature_extractor = FeatureExtractor(
            self.cfg.feature_extractor_cfg, 
            self.device, 
            log_dir
        )
        # Now call parent to load all managers (including ObservationManager)
        super().load_managers()


@configclass
class SOArm101CameraCabinetEnvCfg_PLAY(SOArm101CameraCabinetEnvCfg):
    # Feature extractor for inference (no training, load checkpoint)
    feature_extractor_cfg: FeatureExtractorCfg = FeatureExtractorCfg(train=False, load_checkpoint=True)
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

