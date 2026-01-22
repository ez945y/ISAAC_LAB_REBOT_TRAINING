# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# Use custom SO-ARM-101 agent config
from . import agents

##
# Register Gym environments.
##

# ==============================================================================
# Standard RL Environments
# ==============================================================================

gym.register(
    id="Isaac-Cabinet-SOARM101-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_abs_env_cfg:SOArm101CabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101CabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-SOARM101Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_abs_env_cfg:SOArm101CabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101CabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Cabinet-SOARM101-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_rel_env_cfg:SOArm101CabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101CabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-SOARM101-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_rel_env_cfg:SOArm101CabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101CabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)

# ==============================================================================
# Camera-based RL Environments (Visual observations with CNN embeddings)
# ==============================================================================

gym.register(
    id="Isaac-Cabinet-SOARM101-Camera-v0",
    entry_point=f"{__name__}.so101_camera_env_cfg:SOArm101CameraCabinetEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_camera_env_cfg:SOArm101CameraCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101CameraPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-SOARM101-Camera-Play-v0",
    entry_point=f"{__name__}.so101_camera_env_cfg:SOArm101CameraCabinetEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_camera_env_cfg:SOArm101CameraCabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101CameraPPORunnerCfg",
    },
    disable_env_checker=True,
)