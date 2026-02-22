# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

import isaaclab.sim as sim_utils
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from controll_scripts.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from controll_scripts.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from controll_scripts.input_devices.se3_leader_arm import Se3LeaderArmCfg
from rl_manager.tasks.manager_based.stack import stack_joint_pos_env_cfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
import isaaclab.envs.mdp as mdp

# ── Use our own single-cube MDP (no cube_2/cube_3 dependencies) ──
from so_arm_mimic.source.mdp import observations as move_obs
from so_arm_mimic.source.mdp import terminations as move_term
from so_arm_mimic.source.mdp import events as move_events

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.01, 0.01, 0.01)

# ── Custom USD paths ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_ROBOT_USD = os.path.join(_REPO_ROOT, "tools", "controll_scripts", "so_arm_101", "SO-ARM101v2.usd")
_CUBE_USD = os.path.join(_REPO_ROOT, "tools", "exp", "test", "bodies", "3_4.usd")
_PLATFORM_USD = os.path.join(_REPO_ROOT, "tools", "exp", "test", "bodies", "3_1.usd")


@configclass
class SO101CubeMoveIKAbsMimicEnvCfg(stack_joint_pos_env_cfg.SO101CubeStackEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config for SO-ARM-101 Cube Move (pick right → place left).
    Single cube, 2 subtasks: grasp → place.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # ── Robot: custom USD, elevated to Z=0.05 ──
        self.scene.robot.spawn.usd_path = _ROBOT_USD
        self.scene.robot.init_state.pos = (0.0, -0.005, 0.05)
        self.scene.robot.actuators["arm"].stiffness = 17.8
        self.scene.robot.actuators["arm"].damping = 0.6
        self.scene.robot.actuators["gripper"].stiffness = 17.8
        self.scene.robot.actuators["gripper"].damping = 0.6

        # ── Remove cube_2 and cube_3 from scene (only keep cube_1) ──
        self.scene.cube_2 = None
        self.scene.cube_3 = None

        # ── Cube_1: custom USD, right side ──
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_CUBE_USD,
                scale=(0.001, 0.001, 0.001),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.066),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.02,
                    rest_offset=0.0005,
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=64,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
                semantic_tags=[("class", "cube_1")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.38, -0.04, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ── Platform block: large kinematic block (target placement area) ──
        self.scene.platform_block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PlatformBlock",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_PLATFORM_USD,
                scale=(0.001, 0.001, 0.001),
                semantic_tags=[("class", "platform")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.38, 0.23, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        self.scene.cube_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/Cube_1_FrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cube_1",
                    name="cube_1_center",
                    offset=OffsetCfg(
                        pos=(-0.015, -0.0715, 0.0),
                    ),
                ),
            ],
        )
        self.scene.platform_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/PlatformBlock",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/PlatformBlock_FrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/PlatformBlock",
                    name="platform_center",
                    offset=OffsetCfg(
                        pos=(-0.115, -0.115, 0.0),
                    ),
                ),
            ],
        )
        # ── Events: only randomize 1 cube on the right side ──
        # Disable parent events that override our custom initial joint position
        self.events.init_arm_pose = None
        self.events.randomize_joint_state = None
        # Explicitly reset robot joints to our custom init_state.joint_pos on every reset
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


        # Platform Randomization ──
        self.events.randomize_platform_positions = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.36, 0.40),
                    "y": (0.21, 0.25),
                    "z": (0.0, 0.0),
                    "yaw": (-0.4, 0.4),
                },
                "min_separation": 0.0,
                "asset_cfgs": [SceneEntityCfg("platform_block")],
            },
        )

        self.events.randomize_cube_positions = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.18, 0.38),     # right side workspace
                    "y": (-0.04, -0.17),    # slightly right of center  
                    "z": (0.0, 0.0),
                    "yaw": (-0.7854, 0.7854),
                },
                "min_separation": 0.05,
                "asset_cfgs": [SceneEntityCfg("cube_1")],
            },
        )

        # ── Color Randomization ──
        self.events.randomize_cube_color = EventTerm(
            func=move_events.randomize_material_color,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube_1"),
                "color_ranges": {
                    "r": (0.1, 0.9),
                    "g": (0.1, 0.9),
                    "b": (0.1, 0.9),
                },
            },
        )
        self.events.randomize_platform_color = EventTerm(
            func=move_events.randomize_material_color,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("platform_block"),
                "color_ranges": {
                    "r": (0.1, 0.9),
                    "g": (0.1, 0.9),
                    "b": (0.1, 0.9),
                },
            },
        )


        # ── Actions: DifferentialIK absolute mode + gripper ──
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
        self.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
            use_default_offset=False,
        )

        # ── Teleop devices ──
        # self.teleop_devices = DevicesCfg(
        #     devices={
        #         "leader_arm": Se3LeaderArmCfg(
        #             socket_host="0.0.0.0",
        #             socket_port=5359,
        #             server_mode=True,
        #             pos_sensitivity=1.0,
        #             rot_sensitivity=1.0,
        #             sim_device="cuda:0",
        #         ),
        #     }
        # )

        # ── Observations: replace ALL multi-cube terms with single-cube versions ──
        self.observations.policy.object = ObsTerm(
            func=move_obs.object_obs,
            params={
                "cube_cfg": SceneEntityCfg("cube_1"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        self.observations.policy.cube_positions = ObsTerm(
            func=move_obs.cube_position_in_world_frame,
            params={"cube_cfg": SceneEntityCfg("cube_1")},
        )
        self.observations.policy.cube_orientations = ObsTerm(
            func=move_obs.cube_orientation_in_world_frame,
            params={"cube_cfg": SceneEntityCfg("cube_1")},
        )
        self.observations.policy.eef_pos = ObsTerm(func=move_obs.ee_frame_pos)
        self.observations.policy.eef_quat = ObsTerm(func=move_obs.ee_frame_quat)
        self.observations.policy.gripper_pos = ObsTerm(func=move_obs.gripper_pos)

        # ── Subtask observations: only grasp ──
        self.observations.subtask_terms.grasp_1 = ObsTerm(
            func=move_obs.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        # Remove stack-specific subtask observations
        self.observations.subtask_terms.stack_1 = None
        self.observations.subtask_terms.grasp_2 = None

        # ── Termination: cube placed on platform block ──
        self.terminations.success = DoneTerm(
            func=move_term.cube_on_platform,
            params={
                "cube_frame_name": "cube_1_frame",
                "platform_frame_name": "platform_frame",
                "xy_tolerance": 0.075,
                "min_z_above": 0.015,
            },
        )
        # Remove cube_2 and cube_3 drop terminations
        self.terminations.cube_2_dropping = None
        self.terminations.cube_3_dropping = None

        # ── Datagen config ──
        self.datagen_config.name = "demo_src_move_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 2

        # ── Subtask configs: grasp → place (terminal) ──
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="grasp",
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
                subtask_term_signal=None,  # terminal subtask
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

        # ── Debug visualization ──
        self.scene.ee_frame.debug_vis = False
