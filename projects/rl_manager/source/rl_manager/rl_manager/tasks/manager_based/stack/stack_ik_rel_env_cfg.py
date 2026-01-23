# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
SO-ARM-101 Stack Task with Camera Observations and OpenXR Hand Tracking.

This configuration extends the camera-based environment with OpenXR VR hand tracking
for teleoperation and binary gripper control.
"""

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass
from .stack_joint_pos_env_cfg import SO101CubeStackEnvCfg
from controll_scripts.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from controll_scripts.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from . import mdp as so_arm_mimic_mdp
from isaaclab.sim import SimulationCfg, PhysxCfg

@configclass
class CommandsCfg:
    """Commands configuration for the stack environment."""
    stack_progress = so_arm_mimic_mdp.StackProgressTrackerCfg()


@configclass
class CurriculumCfg:
    difficulty = CurrTerm(
        func=so_arm_mimic_mdp.DifficultyScheduler,
        params={
            "init_difficulty": 0,
            "min_difficulty": 0,
            "max_difficulty": 50,
            "target_progress": 2,       # 目標進度 (2 = 完成堆疊)
            "success_rate": 0.8,        # 完成率門檻 (80%)
            "min_samples": 1000,        # 最少樣本數才能評估
            "promotion_only": True,     # 只升級不降級
        }
    )

    # 前期高：接近 cube_2
    reaching_cube_2_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.reaching_cube_2.weight",
            "modify_fn": so_arm_mimic_mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 3.0,   # 早期高
                "final_value": 0.5,     # 後期降低
                "difficulty_term_str": "difficulty"
            }
        }
    )

    grasp_2_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.grasp_2.weight",
            "modify_fn": so_arm_mimic_mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 2.0,   # 早期低（先學接近）
                "final_value": 12.0,    # 中期高（鼓勵精準抓）
                "difficulty_term_str": "difficulty"
            }
        }
    )

    # 中期高：夾著接近 cube_1
    approaching_cube_1_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.approaching_cube_1.weight",
            "modify_fn": so_arm_mimic_mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 0.0,   # 早期關閉（還沒抓到）
                "final_value": 8.0,     # 中後期高
                "difficulty_term_str": "difficulty"
            }
        }
    )

    # 後期高：堆疊成功
    stack_2_on_1_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.stack_2_on_1.weight",
            "modify_fn": so_arm_mimic_mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 5.0,
                "final_value": 50.0,    # 最終高
                "difficulty_term_str": "difficulty"
            }
        }
    )
    
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # --- Phase 1: Stack Cube 2 on Cube 1 ---
    
    reaching_cube_2 = RewTerm(
        func=so_arm_mimic_mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube_2")},
        weight=0.5
    )


    grasp_2 = RewTerm(
        func=so_arm_mimic_mdp.object_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube_2"),
        },
        weight=5.0
    )

    approaching_cube_1 = RewTerm(
        func=so_arm_mimic_mdp.grasped_and_approaching,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "grasped_object_cfg": SceneEntityCfg("cube_2"),
            "target_object_cfg": SceneEntityCfg("cube_1"),
        },
        weight=5.0
    )

    lifted_cube2 = RewTerm(
        func=so_arm_mimic_mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cube_2")},
        weight=3.0
    )
    
    stack_2_on_1 = RewTerm(
        func=so_arm_mimic_mdp.object_stacked,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_1"),
        },
        weight=20.0
    )

    # --- Penalties ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)
    ee_floor_penalty = RewTerm(
        func=so_arm_mimic_mdp.ee_floor_penalty,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "floor_height_threshold": 0.04,
            "penalty_strength": 1.0,
        },
        weight=-2.0
    )


@configclass
class SO101CubeStackRelEnvCfg(SO101CubeStackEnvCfg):
    """
    SO-ARM-101 Stack Task with Camera Observations and OpenXR Hand Tracking.
    
    Inherits from Camera Mimic Config but:
    - Changes teleop device to OpenXR hand tracking
    - Changes gripper action to binary (open/close)
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="adaptive",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

        self.scene.num_envs = 1024

        self.observations.policy.concatenate_terms = True

        self.commands = CommandsCfg()
        self.rewards = RewardsCfg()
        self.curriculum = CurriculumCfg()

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2
        self.sim.gpu_collision_stack_size = 4 * (2**30) - 1
        self.sim.dt = 1 / 120

        self.sim = SimulationCfg(
            dt=1 / 120,
            gravity=(0.0, 0.0, -9.81),
            physx=PhysxCfg(
                solver_type=1,
                max_position_iteration_count=192,
                max_velocity_iteration_count=1,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.00625,
                gpu_max_rigid_contact_count=2**24,                # 加大
                gpu_max_rigid_patch_count=2**24,
                gpu_collision_stack_size = 4 * (2**30) - 1,
                gpu_found_lost_pairs_capacity=1024 * 1024 * 64,
                gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 64,
                gpu_total_aggregate_pairs_capacity=128 * 1024,
                gpu_max_num_partitions=1,
            ),
        )
        