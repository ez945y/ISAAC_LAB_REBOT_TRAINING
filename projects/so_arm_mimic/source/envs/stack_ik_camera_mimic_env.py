# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from .stack_ik_abs_mimic_env import SO101CubeStackIKAbsMimicEnv
from .stack_ik_camera_mimic_env_cfg import SO101CubeStackCameraMimicEnvCfg
from rl_manager.tasks.manager_based.cabinet.config.so_arm_101.feature_extractor import (
    FeatureExtractor,
)

class SO101CubeStackCameraMimicEnv(SO101CubeStackIKAbsMimicEnv):
    cfg: SO101CubeStackCameraMimicEnvCfg

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