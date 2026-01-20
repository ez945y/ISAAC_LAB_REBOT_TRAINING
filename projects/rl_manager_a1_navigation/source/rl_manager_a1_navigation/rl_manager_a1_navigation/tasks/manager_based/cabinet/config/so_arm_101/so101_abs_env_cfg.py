# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

import os
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
# from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg, BinaryJointPositionActionCfg
from isaaclab.managers import SceneEntityCfg

from rl_manager_a1_navigation.tasks.manager_based.cabinet.cabinet_env_cfg import CabinetEnvCfg, FRAME_MARKER_SMALL_CFG
from controll_scripts.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from controll_scripts.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from isaaclab.sim import SimulationCfg, PhysxCfg

@configclass
class SOArm101CabinetEnvCfg(CabinetEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot
        import controll_scripts
        script_dir = os.path.dirname(os.path.abspath(controll_scripts.__file__))
        usd_path = os.path.join(script_dir, "so_arm_101", "SO-ARM101.usd")
        
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                semantic_tags=[("class", "robot")],
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.02, 0.0, 0.6),
                joint_pos={
                    "shoulder_pan": 0.0,  
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 1.658,
                    "wrist_roll": 0.0,
                    "gripper": 1.7,
                },
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                    effort_limit=10.0,
                    stiffness=17.8,
                    damping=0.6,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper"],
                    effort_limit=2.0,
                    stiffness=17.8,
                    damping=0.6,
                ),
            },
        )

        # Scale down the cabinet for the small robot
        self.scene.cabinet.spawn.scale = (0.55, 0.55, 0.55)
        # Adjust position closer and to the left to align with the robot
        self.scene.cabinet.init_state.pos = (0.55, 0.0, 0.6)

        # self.actions.arm_action = OperationalSpaceControllerActionCfg(
        #     asset_name="robot",
        #     joint_names=so_arm_config.arm_joint_names,
        #     body_name=so_arm_config.ee_body_name,
        #     controller_cfg=OperationalSpaceControllerCfg(
        #         target_types=["pose_abs"],
        #         # Only control 5 axes: position (x,y,z) + orientation (pitch, yaw)
        #         # Roll (axis index 3) is left free to match 5 DOF capability
        #         motion_control_axes_task=(1, 1, 1, 0, 1, 1),
        #         motion_stiffness_task=so_arm_config.osc_motion_stiffness,
        #         motion_damping_ratio_task=so_arm_config.osc_motion_damping_ratio,
        #         inertial_dynamics_decoupling=False,
        #         gravity_compensation=True,
        #     ),
        #     position_scale=1.0,
        #     orientation_scale=1.0,
        # )
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
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 1.7},
            close_command_expr={"gripper": 0.3},
        )

        # Frame for observations
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
                # Index 1: 左指 (固定/虛擬) - 綁定在手掌 (wrist_link)
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
                # Index 2: 右指 (活動/真實) - 綁定在 gripper_link
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )


        # Rewards overrides
        self.rewards.approach_gripper_handle.params["offset"] = 0.05
        self.rewards.grasp_handle.params["open_joint_pos"] = 1.74
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["gripper"]

        # Observations
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ])
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ])

        # self.sim.gpu_collision_stack_size = 4 * (2**30) - 1

        self.scene.num_envs = 512
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
        

@configclass
class SOArm101CabinetEnvCfg_PLAY(SOArm101CabinetEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
