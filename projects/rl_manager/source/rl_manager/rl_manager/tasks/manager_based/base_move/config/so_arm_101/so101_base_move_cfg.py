# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM-101 configuration for base move task."""

import os
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg

from rl_manager.tasks.manager_based.base_move.base_move_env_cfg import BaseMoveEnvCfg, FRAME_MARKER_SMALL_CFG
from controll_scripts.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from controll_scripts.controllers.differential_ik_cfg import DifferentialIKControllerCfg


@configclass
class SOArm101BaseMoveEnvCfg(BaseMoveEnvCfg):
    """SO-ARM-101 configuration for base move task.
    
    Gripper facing downward (like stack task) - reduces one DOF.
    """
    
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
                    disable_gravity=True,
                ),
                semantic_tags=[("class", "robot")],
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),  # 放在桌面上
                joint_pos={
                    "shoulder_pan": 0.0,
                    "shoulder_lift": -1.0,   # 肩膀向下彎曲
                    "elbow_flex": 1.0,      # 手肘大幅彎曲
                    "wrist_flex": 1.6,       # 調整手腕補償高度
                    "wrist_roll": 0.0,
                    "gripper": 0.0,
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
        
        # Arm action with DifferentialIK
        # Use 'adaptive' method like stack task for downward gripper
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls_5dof",
            ),
            scale=0.05,  # Slow speed: 5cm max per step
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
        
        # Frame for observations (same offset as stack task)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.017, 0.0, -0.07812),
                        rot=(0.0, 0.7071068, 0.7071068, 0.0)
                    ),
                ),
            ],
        )


@configclass
class SOArm101BaseMoveEnvCfg_PLAY(SOArm101BaseMoveEnvCfg):
    """Play configuration with fewer environments."""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
