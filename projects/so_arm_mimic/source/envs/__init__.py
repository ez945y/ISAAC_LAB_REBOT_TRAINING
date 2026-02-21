# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .stack_ik_abs_mimic_env import SO101CubeStackIKAbsMimicEnv
from .stack_ik_abs_mimic_env_cfg import SO101CubeStackIKAbsMimicEnvCfg
from .stack_ik_rel_mimic_env import SO101CubeStackRelMimicEnv
from .stack_ik_rel_mimic_env_cfg import SO101CubeStackRelMimicEnvCfg
from .move_ik_abs_mimic_env import SO101CubeMoveIKAbsMimicEnv
from .move_ik_abs_mimic_env_cfg import SO101CubeMoveIKAbsMimicEnvCfg

##
# SO-ARM-101 Pick and Place - Absolute Control
##
gym.register(
    id="Isaac-PickPlace-SOArm-Abs-Mimic-v0",
    entry_point="so_arm_mimic.source.envs.stack_ik_abs_mimic_env:SO101CubeStackIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": "so_arm_mimic.source.envs.stack_ik_abs_mimic_env_cfg:SO101CubeStackIKAbsMimicEnvCfg",
    },
    disable_env_checker=True,
)

##
# SO-ARM-101 Pick and Place - Relative Control with OpenXR Hand Tracking
##
gym.register(
    id="Isaac-PickPlace-SOArm-Rel-Mimic-v0",
    entry_point="so_arm_mimic.source.envs.stack_ik_rel_mimic_env:SO101CubeStackRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": "so_arm_mimic.source.envs.stack_ik_rel_mimic_env_cfg:SO101CubeStackRelMimicEnvCfg",
    },
    disable_env_checker=True,
)

##
# SO-ARM-101 Move Cube (Pick Right → Place Left) - Absolute Control
##
gym.register(
    id="Isaac-Move-SOArm-Abs-Mimic-v0",
    entry_point="so_arm_mimic.source.envs.move_ik_abs_mimic_env:SO101CubeMoveIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": "so_arm_mimic.source.envs.move_ik_abs_mimic_env_cfg:SO101CubeMoveIKAbsMimicEnvCfg",
    },
    disable_env_checker=True,
)