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
    id="Isaac-Cabinet-SO-ARM-101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_abs_env_cfg:SOArm101CabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101CabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-SO-ARM-101-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_abs_env_cfg:SOArm101CabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101CabinetPPORunnerCfg",
    },
    disable_env_checker=True,
)