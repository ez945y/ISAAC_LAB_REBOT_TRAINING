# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for SO-ARM-101 base move task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SOArm101BaseMovePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for SO-ARM-101 base move task."""
    
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "so_arm_101_base_move"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,  # Low initial noise for simple task
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128],  # Smaller network for simpler task
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
