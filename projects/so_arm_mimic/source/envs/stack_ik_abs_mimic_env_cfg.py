# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass
from isaaclab.devices.device_base import DevicesCfg
from controll_scripts.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from controll_scripts.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from controll_scripts.input_devices.se3_leader_arm import Se3LeaderArmCfg
from rl_manager.tasks.manager_based.stack import stack_joint_pos_env_cfg

@configclass
class SO101CubeStackIKAbsMimicEnvCfg(stack_joint_pos_env_cfg.SO101CubeStackEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for SO-ARM-101 Cube Stack IK Abs env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        self.scene.robot.actuators["arm"].stiffness = 17.8  
        self.scene.robot.actuators["arm"].damping = 0.6
        self.scene.robot.actuators["gripper"].stiffness = 17.8
        self.scene.robot.actuators["gripper"].damping = 0.6

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls_5dof",
            ),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "leader_arm": Se3LeaderArmCfg(
                    socket_host="0.0.0.0",
                    socket_port=5359,
                    server_mode=True,
                    pos_sensitivity=1.0,
                    rot_sensitivity=1.0,
                    sim_device="cuda:0",
                ),
            }
        )

        # Override the existing values
        self.datagen_config.name = "demo_src_stack_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 2

        # The following are the subtask configurations for the stack task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=20,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="stack_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=20,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=20,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=20,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["so101"] = subtask_configs

        # Enable debug visualization for the end-effector frame
        self.scene.ee_frame.debug_vis = True
