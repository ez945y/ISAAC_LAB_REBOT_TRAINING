# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from controll_scripts.input_devices.se3_leader_arm import Se3LeaderArmCfg
from rl_manager.tasks.manager_based.stack import stack_joint_pos_env_cfg


@configclass
class SO101CubeStackEnvCfg(stack_joint_pos_env_cfg.SO101CubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Disable the init_arm_pose event that resets robot to joints=0 pose
        # This prevents the robot from jumping to a different pose after reset
        # The robot will use the URDF default pose or the teleop device's last pose instead
        
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
                ik_method="dls",
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
