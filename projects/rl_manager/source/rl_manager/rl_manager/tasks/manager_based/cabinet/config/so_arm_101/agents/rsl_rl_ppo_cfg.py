# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for SO-ARM-101 cabinet manipulation task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SOArm101CabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for SO-ARM-101 opening a cabinet drawer using IK control."""
    
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "so_arm_101_open_drawer"
    
    policy = RslRlPpoActorCriticCfg(
        # Use 'log' noise type to prevent negative std values
        # This is critical for numerical stability
        init_noise_std=1.0,  # Lower initial noise for IK control
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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SOArm101CameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for SO-ARM-101 opening a cabinet drawer using IK control."""
    
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "so_arm_101_camera_open_drawer"
    
    policy = RslRlPpoActorCriticCfg(
        # Use 'log' noise type to prevent negative std values
        # This is critical for numerical stability
        init_noise_std=1.0,  # Lower initial noise for IK control
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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
