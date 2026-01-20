# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg, BinaryJointPositionActionCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg, FRAME_MARKER_SMALL_CFG
from isaaclab_tasks.robot_control.configs.so_arm_101 import SOArm101Config

so_arm_config = SOArm101Config()

@configclass
class SOArm101CabinetEnvCfg(CabinetEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = so_arm_config.get_articulation_cfg(for_osc=True)
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot.init_state.pos = (-0.02, 0.0, 0.6)
        self.scene.num_envs = 512

        # Scale down the cabinet for the small robot
        self.scene.cabinet.spawn.scale = (0.55, 0.55, 0.55)
        # Adjust position closer and to the left to align with the robot
        self.scene.cabinet.init_state.pos = (0.55, 0.0, 0.6)

        # Actions - SO-ARM-101 has 5 DOF, so we can only control 5 task-space axes
        # Control: X, Y, Z position (3) + pitch, yaw rotation (2) = 5 axes
        # Leave roll rotation (axis 3) free to avoid over-constrained system
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=so_arm_config.arm_joint_names,
            body_name=so_arm_config.ee_body_name,
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                # Only control 5 axes: position (x,y,z) + orientation (pitch, yaw)
                # Roll (axis index 3) is left free to match 5 DOF capability
                motion_control_axes_task=(1, 1, 1, 0, 1, 1),
                motion_stiffness_task=so_arm_config.osc_motion_stiffness,
                motion_damping_ratio_task=so_arm_config.osc_motion_damping_ratio,
                inertial_dynamics_decoupling=False,
                gravity_compensation=True,
            ),
            position_scale=1.0,
            orientation_scale=1.0,
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[so_arm_config.gripper_joint_name],
            open_command_expr={so_arm_config.gripper_joint_name: so_arm_config.gripper_open_pos},
            close_command_expr={so_arm_config.gripper_joint_name: so_arm_config.gripper_close_pos},
        )

        # Frame for observations
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{ENV_REGEX_NS}/Robot/{so_arm_config.ee_body_name}",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
                # Index 1: 左指 (固定/虛擬) - 綁定在手掌 (wrist_link)
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{ENV_REGEX_NS}/Robot/wrist_link",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.03)),
                ),
                # Index 2: 右指 (活動/真實) - 綁定在 gripper_link
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{ENV_REGEX_NS}/Robot/gripper_link",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # Rewards overrides
        self.rewards.approach_gripper_handle.params["offset"] = 0.05
        self.rewards.grasp_handle.params["open_joint_pos"] = 1.74
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = [so_arm_config.gripper_joint_name]

        # Observations
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=so_arm_config.arm_joint_names)
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=so_arm_config.arm_joint_names)

@configclass
class SOArm101CabinetEnvCfg_PLAY(SOArm101CabinetEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
