# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# from .so_arm_stack_joint_mimic_env import SOArmStackJointMimicEnv
# from .so_arm_stack_joint_mimic_env_cfg import SOArmStackJointMimicEnvCfg
# from .so_arm_stack_camera_mimic_env_cfg import SOArmStackCameraMimicEnvCfg

from .stack_ik_abs_mimic_env import SO101CubeStackIKAbsMimicEnv
from .stack_ik_abs_mimic_env_cfg import SO101CubeStackIKAbsMimicEnvCfg
from .stack_ik_abs_camera_mimic_env_cfg import SO101CubeStackCameraMimicEnvCfg
from .stack_ik_rel_mimic_env import SO101CubeStackRelMimicEnv
from .stack_ik_rel_mimic_env_cfg import SO101CubeStackRelMimicEnvCfg

##
# SO-ARM-101 Pick and Place - Joint Control
##
gym.register(
    id="Isaac-PickPlace-SOArm-Mimic-v0",
    entry_point="so_arm_mimic.source.envs.stack_ik_abs_mimic_env:SO101CubeStackIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": "so_arm_mimic.source.envs.stack_ik_abs_mimic_env_cfg:SO101CubeStackIKAbsMimicEnvCfg",
    },
    disable_env_checker=True,
)

##
# SO-ARM-101 Pick and Place - Camera/Visual Control (No Cheating)
##
gym.register(
    id="Isaac-PickPlace-SOArm-Camera-Mimic-v0",
    entry_point="so_arm_mimic.source.envs.stack_ik_abs_mimic_env:SO101CubeStackIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": "so_arm_mimic.source.envs.stack_ik_abs_camera_mimic_env_cfg:SO101CubeStackCameraMimicEnvCfg",
    },
    disable_env_checker=True,
)

##
# SO-ARM-101 Pick and Place - Camera/Visual Control with OpenXR Hand Tracking
##
gym.register(
    id="Isaac-PickPlace-SOArm-Rel-Mimic-v0",
    entry_point="so_arm_mimic.source.envs.stack_ik_rel_mimic_env:SO101CubeStackRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": "so_arm_mimic.source.envs.stack_ik_rel_mimic_env_cfg:SO101CubeStackRelMimicEnvCfg",
    },
    disable_env_checker=True,
)

