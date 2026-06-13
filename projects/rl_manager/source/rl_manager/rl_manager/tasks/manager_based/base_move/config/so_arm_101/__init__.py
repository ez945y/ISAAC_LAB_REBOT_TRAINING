# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-BaseMove-SOARM101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_base_move_cfg:SOArm101BaseMoveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101BaseMovePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-BaseMove-SOARM101-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_base_move_cfg:SOArm101BaseMoveEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SOArm101BaseMovePPORunnerCfg",
    },
    disable_env_checker=True,
)
