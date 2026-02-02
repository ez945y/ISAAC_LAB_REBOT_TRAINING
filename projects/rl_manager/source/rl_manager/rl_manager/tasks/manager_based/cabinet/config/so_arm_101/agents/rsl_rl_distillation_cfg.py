# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationStudentTeacherRecurrentCfg
)


@configclass
class SOArm101DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100
    save_interval = 20
    experiment_name = "so_arm_101_open_drawer"
    obs_groups = {"policy": ["policy", "camera"], "critic": ["policy", "camera"], "teacher": ["policy", "extra"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type = "log",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=20,
    )

@configclass
class SOArm101DistillationRunnerRecurrentCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100
    save_interval = 10
    experiment_name = "so_arm_101_open_drawer"
    obs_groups = {"policy": ["policy", "camera"], "critic": ["policy", "camera"], "teacher": ["policy", "extra"]}
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=0.1,
        noise_std_type = "log",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=20,
    )
