# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CubeStackRelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "so_arm_101_cube_stack_rel"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Lower initial noise for IK control
        noise_std_type="log",  # Ensures std is always positive via exp()
        actor_obs_normalization=True,  # Normalize observations for smaller robot
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # Slightly lower learning rate for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  # Lower desired KL for more stable training
        max_grad_norm=1.0,
    )
