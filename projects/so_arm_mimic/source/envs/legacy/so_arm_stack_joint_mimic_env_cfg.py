# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Isaac Lab Mimic Environment Configuration (Joint Control)
"""

import os
import torch
from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg, JointEffortActionCfg, DifferentialInverseKinematicsActionCfg
)
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg


from isaaclab.envs.mimic_env_cfg import DataGenConfig, MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import Official Functionalities
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
# Success condition: all cubes stacked AND gripper open (using official function) -> see cubes_stacked_single_gripper below
from isaaclab_tasks.manager_based.manipulation.stack.mdp.terminations import cubes_stacked

# Import Se3LeaderArmCfg for teleop device registration
from controll_scripts.input_devices.se3_leader_arm import Se3LeaderArmCfg


##
# Scene definition
##
@configclass
class SOArmSceneCfg(InteractiveSceneCfg):
    """Configuration for the SO-ARM-101 scene."""

    # World
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    # Robot (configured in __post_init__)
    robot: ArticulationCfg = MISSING
    
    # End-effector frame (configured in __post_init__)
    ee_frame: FrameTransformerCfg = MISSING

    # Three cubes for stacking task (matching official Franka Stack env)
    # Cube 1: Blue (bottom cube - target for stacking)
    cube_1: RigidObjectCfg = MISSING
    # Cube 2: Red (first cube to pick and stack)
    cube_2: RigidObjectCfg = MISSING
    # Cube 3: Green (second cube to pick and stack)
    cube_3: RigidObjectCfg = MISSING

    # Cameras (Added to base scene as per refactoring request)
    # Hand-mounted camera (Self View)
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/self_view_camera",
        update_period=0.1,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=None, 
    )

    # Fixed camera (Full View)
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/full_view_camera", 
        update_period=0.1,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=None, 
    )


##
# MDP settings
##

def ee_frame_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position in world frame."""
    return env.scene[asset_cfg.name].data.target_pos_w[..., 0, :]

def ee_frame_quat(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector orientation (quaternion) in world frame."""
    return env.scene[asset_cfg.name].data.target_quat_w[..., 0, :]

# Action history
def last_action(env) -> torch.Tensor:
    """Last action taken by the agent."""
    return env.action_manager.action

# Relative joint positions/velocities
def joint_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default."""
    return env.scene[asset_cfg.name].data.joint_pos - env.scene[asset_cfg.name].data.default_joint_pos

def joint_vel_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities."""
    return env.scene[asset_cfg.name].data.joint_vel

# Object state - Three cubes
def cube_positions_in_world_frame(
    env,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of all three cubes in the world frame."""
    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    cube_3 = env.scene[cube_3_cfg.name]
    return torch.cat((cube_1.data.root_pos_w, cube_2.data.root_pos_w, cube_3.data.root_pos_w), dim=1)

def cube_orientations_in_world_frame(
    env,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The orientation of all three cubes in the world frame."""
    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    cube_3 = env.scene[cube_3_cfg.name]
    return torch.cat((cube_1.data.root_quat_w, cube_2.data.root_quat_w, cube_3.data.root_quat_w), dim=1)

def gripper_pos(env) -> torch.Tensor:
    """Gripper joint position."""
    return env.scene["robot"].data.joint_pos[:, -1:]

# ==============================================================================
# Subtask signals for stack task - Core computation helpers
# ==============================================================================

# State tracking for logging
_grasp_state = {"cube_2": False, "cube_3": False}
_stack_state = {"stack_1": False}

def reset_subtask_logging_state():
    """Reset the logging state for subtask signals. Call this at the start of each episode."""
    global _grasp_state, _stack_state
    _grasp_state = {"cube_2": False, "cube_3": False}
    _stack_state = {"stack_1": False}
    # Also reset cubes_stacked_single_gripper state
    if hasattr(cubes_stacked_single_gripper, "_last_state"):
        cubes_stacked_single_gripper._last_state = {"stack_1": False, "stack_2": False, "complete": False}


def _get_gripper_state(env, robot_cfg: SceneEntityCfg) -> tuple[torch.Tensor, float, float]:
    """Get gripper state and configuration values.
    
    Returns:
        tuple: (gripper_joint_pos, gripper_open_val, gripper_threshold)
    """
    gripper_state = env.scene[robot_cfg.name].data.joint_pos[:, -1]
    gripper_open_val = getattr(env.cfg, 'gripper_open_val', 1.75)
    gripper_threshold = getattr(env.cfg, 'gripper_threshold', 0.1)
    return gripper_state, gripper_open_val, gripper_threshold


def _compute_grasp_distance(env, ee_frame_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute distance between end-effector and object.
    
    Returns:
        torch.Tensor: Distance in meters (shape: [num_envs])
    """
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    obj_pos = env.scene[object_cfg.name].data.root_pos_w
    return torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)


def _compute_stack_distances(
    env, 
    upper_object_cfg: SceneEntityCfg, 
    lower_object_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute stacking distances between two objects.
    
    Returns:
        tuple: (xy_dist, height_dist, pos_diff_z) where:
            - xy_dist: horizontal distance (shape: [num_envs])
            - height_dist: absolute vertical distance (shape: [num_envs])
            - pos_diff_z: signed vertical difference (upper - lower), negative means upper is above
    """
    upper_object = env.scene[upper_object_cfg.name]
    lower_object = env.scene[lower_object_cfg.name]
    
    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.abs(pos_diff[:, 2])
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    
    # pos_diff[:, 2] > 0 means upper object is above lower object
    return xy_dist, height_dist, pos_diff[:, 2]


def _check_stacked_geometry(
    xy_dist: torch.Tensor, 
    height_dist: torch.Tensor, 
    pos_diff_z: torch.Tensor,
    xy_threshold: float, 
    height_threshold: float, 
    height_diff: float,
    check_direction: bool = True
) -> torch.Tensor:
    """Check if objects are geometrically stacked.
    
    Args:
        check_direction: If True, also verify upper object is above lower object.
    
    Returns:
        torch.Tensor: Boolean tensor indicating stacked state (shape: [num_envs])
    """
    stacked = torch.logical_and(
        xy_dist < xy_threshold, 
        # Single-direction check: height must be at least height_diff (like official version)
        (height_dist - height_diff) < height_threshold
    )
    if check_direction:
        # pos_diff_z > 0 means upper object is above lower object (since we compute upper - lower)
        stacked = torch.logical_and(pos_diff_z > 0.0, stacked)
    return stacked


# ==============================================================================
# Subtask signal functions (for observations/annotations)
# ==============================================================================

def object_grasped(
    env, 
    robot_cfg: SceneEntityCfg, 
    ee_frame_cfg: SceneEntityCfg, 
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.12,  # 6cm, same as official version
) -> torch.Tensor:
    """Check if object is grasped (close to EE and gripper closed)."""
    global _grasp_state
    
    # Use helper functions
    pose_diff = _compute_grasp_distance(env, ee_frame_cfg, object_cfg)
    gripper_state, gripper_open_val, gripper_threshold = _get_gripper_state(env, robot_cfg)
    
    # Check if gripper is closed (not fully open)
    is_gripping = torch.abs(gripper_state - gripper_open_val) > gripper_threshold
    grasped = torch.logical_and(pose_diff < diff_threshold, is_gripping)
    
    # Logging: print when grasp state changes
    obj_name = object_cfg.name
    is_grasped_now = grasped[0].item() if grasped.numel() > 0 else False
    if obj_name in _grasp_state:
        if is_grasped_now and not _grasp_state[obj_name]:
            cube_color = "Red" if obj_name == "cube_2" else "Green"
            signal_name = "grasp_1" if obj_name == "cube_2" else "grasp_2"
            print(f">> [SUBTASK] {signal_name}: {cube_color} cube GRASPED (pose_diff={pose_diff[0].item():.3f}m)", flush=True)
        elif not is_grasped_now and _grasp_state[obj_name]:
            cube_color = "Red" if obj_name == "cube_2" else "Green"
            signal_name = "grasp_1" if obj_name == "cube_2" else "grasp_2"
            print(f">> [SUBTASK] {signal_name}: {cube_color} cube RELEASED", flush=True)
        _grasp_state[obj_name] = is_grasped_now
    
    return grasped.float().unsqueeze(-1)


def object_stacked(
    env,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.01,
    height_diff: float = 0.039,
) -> torch.Tensor:
    """Check if upper cube is stacked on lower cube (used for subtask annotation)."""
    global _stack_state
    
    # Use helper functions
    xy_dist, height_dist, pos_diff_z = _compute_stack_distances(env, upper_object_cfg, lower_object_cfg)
    stacked = _check_stacked_geometry(
        xy_dist, height_dist, pos_diff_z, 
        xy_threshold, height_threshold, height_diff,
        check_direction=False  # Don't check direction for subtask signal
    )
    
    # Also check gripper is open (released the cube) for subtask completion
    # Using isclose for stricter check like official version (for single gripper)
    gripper_state, gripper_open_val, _ = _get_gripper_state(env, robot_cfg)
    is_open = torch.isclose(
        gripper_state,
        torch.tensor(gripper_open_val, dtype=torch.float32, device=gripper_state.device),
        atol=1.0,  # Relaxed for single gripper (SO-ARM opens to ~1.5-1.8 range)
        rtol=1e-4,
    )
    stacked = torch.logical_and(stacked, is_open)
    
    # Logging: print when stack state changes (only for stack_1: Red on Blue)
    if upper_object_cfg.name == "cube_2" and lower_object_cfg.name == "cube_1":
        is_stacked_now = stacked[0].item() if stacked.numel() > 0 else False
        if is_stacked_now and not _stack_state["stack_1"]:
            print(f">> [SUBTASK] stack_1: Red cube STACKED on Blue (xy={xy_dist[0].item():.3f}m, h={height_dist[0].item():.3f}m)", flush=True)
        elif not is_stacked_now and _stack_state["stack_1"]:
            print(f">> [SUBTASK] stack_1: Stack BROKEN (xy={xy_dist[0].item():.3f}m, h={height_dist[0].item():.3f}m)", flush=True)
        _stack_state["stack_1"] = is_stacked_now
    
    return stacked.float().unsqueeze(-1)



##
# Observation configuration
##
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        eef_pos = ObsTerm(func=ee_frame_pos, params={"asset_cfg": SceneEntityCfg("ee_frame")})
        eef_quat = ObsTerm(func=ee_frame_quat, params={"asset_cfg": SceneEntityCfg("ee_frame")})
        joint_pos = ObsTerm(func=joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=last_action)
        gripper_pos_obs = ObsTerm(func=gripper_pos)
        cube_positions = ObsTerm(func=cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=cube_orientations_in_world_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask signals - matching official Stack task."""
        
        # Grasp red cube (cube_2)
        grasp_1 = ObsTerm(
            func=object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        # Stack red cube on blue cube (cube_2 on cube_1)
        stack_1 = ObsTerm(
            func=object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        # Grasp green cube (cube_3)
        grasp_2 = ObsTerm(
            func=object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


##
# Action configuration
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # IK Control for Arm
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        body_name="gripper_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        ),
    )
    # arm_action = OperationalSpaceControllerActionCfg(
    #         asset_name="robot",
    #         joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    #         body_name="gripper_link",
    #         # If a task frame different from articulation root/base is desired, a RigidObject, e.g., "task_frame",
    #         # can be added to the scene and its relative path could provided as task_frame_rel_path
    #         # task_frame_rel_path="task_frame",
    #         controller_cfg=OperationalSpaceControllerCfg(
    #             target_types=["pose_abs"],
    #             impedance_mode="fixed", # fixed or variable_kp
    #             inertial_dynamics_decoupling=False, # True or False
    #             partial_inertial_dynamics_decoupling=False,
    #             gravity_compensation=False,
    #             motion_stiffness_task=5.0,
    #             motion_damping_ratio_task=1.0,
    #             motion_stiffness_limits_task=(0.0, 20.0),
    #             nullspace_control="none", # position or none
    #         ),
    #         nullspace_joint_pos_target="none", # center or none
    #         position_scale=1.0,
    #         orientation_scale=1.0,
    #         stiffness_scale=1.0,
    #     )

        
    # arm_action = OperationalSpaceControllerActionCfg(
    #     asset_name="robot",
    #     joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    #     body_name="gripper_link",
    #     # If a task frame different from articulation root/base is desired, a RigidObject, e.g., "task_frame",
    #     # can be added to the scene and its relative path could provided as task_frame_rel_path
    #     # task_frame_rel_path="task_frame",
    #     controller_cfg=OperationalSpaceControllerCfg(
    #         target_types=["pose_abs"],
    #         motion_control_axes_task=(1, 1, 1, 1, 1, 1),
    #         motion_stiffness_task=(150.0, 150.0, 150.0, 50.0, 50.0, 50.0),
    #         motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    #         inertial_dynamics_decoupling=False,
    #         gravity_compensation=True,
    #     ),
    #     nullspace_joint_pos_target="none", # center or none
    #     position_scale=1.0,
    #     orientation_scale=1.0,
    #     stiffness_scale=1.0,
    # )

    # Gripper joint (Position Control)
    gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=1.0, 
    )


##
# Termination configuration
##
def time_out_termination(env) -> torch.Tensor:
    """Time out termination."""
    return env.episode_length_buf >= env.max_episode_length


def root_height_below_minimum(env, asset_cfg: SceneEntityCfg, minimum_height: float = -0.05) -> torch.Tensor:
    """Check if object dropped below minimum height."""
    obj = env.scene[asset_cfg.name]
    return obj.data.root_pos_w[:, 2] < minimum_height


def cubes_stacked_single_gripper(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.01,
    height_diff: float = 0.04,  # Cube size (0.8 * 0.05)
) -> torch.Tensor:
    """Success: All cubes are stacked AND gripper is open (for single gripper robots like SO-ARM).
    
    Uses the shared helper functions for computing stacking state.
    """
    
    # Use helper functions for stack computations
    # Stack 1: cube_2 (Red) on cube_1 (Blue)
    xy_dist_c12, h_dist_c12, pos_diff_z_c12 = _compute_stack_distances(env, cube_2_cfg, cube_1_cfg)
    stack_1_ok = _check_stacked_geometry(
        xy_dist_c12, h_dist_c12, pos_diff_z_c12,
        xy_threshold, height_threshold, height_diff,
        check_direction=True  # Ensure cube_2 is above cube_1
    )

    # Stack 2: cube_3 (Green) on cube_2 (Red)
    xy_dist_c23, h_dist_c23, pos_diff_z_c23 = _compute_stack_distances(env, cube_3_cfg, cube_2_cfg)
    stack_2_ok = _check_stacked_geometry(
        xy_dist_c23, h_dist_c23, pos_diff_z_c23,
        xy_threshold, height_threshold, height_diff,
        check_direction=True  # Ensure cube_3 is above cube_2
    )

    # Check gripper is open using helper (using isclose for consistency with object_stacked)
    gripper_state, gripper_open_val, _ = _get_gripper_state(env, robot_cfg)
    gripper_is_open = torch.isclose(
        gripper_state,
        torch.tensor(gripper_open_val, dtype=torch.float32, device=gripper_state.device),
        atol=1.0,  # Relaxed for single gripper (SO-ARM opens to ~1.5-1.8 range)
        rtol=1e-4,
    )

    # Overall success: both stacks OK and gripper open
    stacked = torch.logical_and(stack_1_ok, stack_2_ok)
    success = torch.logical_and(stacked, gripper_is_open)

    # Logging for user feedback (using function attribute to store state)
    if not hasattr(cubes_stacked_single_gripper, "_last_state"):
        cubes_stacked_single_gripper._last_state = {"stack_1": False, "stack_2": False, "complete": False}
    
    # Check current state for environment 0 (assuming single env for demo recording)
    is_stack_1 = stack_1_ok[0].item()
    is_stack_2 = stack_2_ok[0].item()
    is_success = success[0].item()

    # Reset state if stack broken
    if not is_stack_1 and cubes_stacked_single_gripper._last_state["stack_1"]:
         print(">> [INFO] Stack 1 (Red on Blue) broken!", flush=True)
         cubes_stacked_single_gripper._last_state = {"stack_1": False, "stack_2": False, "complete": False}

    # Print logs on state change only
    if is_stack_1 and not cubes_stacked_single_gripper._last_state["stack_1"]:
        print("\n>> [SUCCESS] Step 1 Complete: Red Cube stacked on Blue Cube!", flush=True)
        cubes_stacked_single_gripper._last_state["stack_1"] = True

    if is_stack_2 and not cubes_stacked_single_gripper._last_state["stack_2"]:
        if is_stack_1:
            print(">> [SUCCESS] Step 2 Complete: Green Cube stacked on Red Cube!", flush=True)
            print(">> [ACTION] Now OPEN the gripper to finish!", flush=True)
        cubes_stacked_single_gripper._last_state["stack_2"] = True

    # Only print complete message once
    if is_success and not cubes_stacked_single_gripper._last_state["complete"]:
        print(">> [COMPLETE] Task Finished! Resetting environment...\n", flush=True)
        cubes_stacked_single_gripper._last_state["complete"] = True

    return success


@configclass
class TerminationsCfg:
    """Termination conditions for stack task."""
    
    time_out = DoneTerm(func=time_out_termination, time_out=True)
    
    # Check if any cube dropped (matching official config)
    cube_1_dropping = DoneTerm(
        func=root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("cube_1"), "minimum_height": -0.05},
    )
    cube_2_dropping = DoneTerm(
        func=root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("cube_2"), "minimum_height": -0.05},
    )
    cube_3_dropping = DoneTerm(
        func=root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("cube_3"), "minimum_height": -0.05},
    )
    
    # Success condition: all cubes stacked AND gripper open (single gripper version for SO-ARM)
    success = DoneTerm(func=cubes_stacked_single_gripper)


@configclass
class RewardsCfg:
    """Empty rewards config for demo recording (no RL training)."""
    pass





@configclass
class EventCfg:
    """Event configuration for SO-ARM stack task - using official functions."""
    
    # Reset robot joints to default position (defined in __post_init__)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0), # Exact default position (scale=1.0)
            "velocity_range": (0.0, 0.0), # Zero velocity
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Randomize cube positions on reset (using official function)
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.3),  # Expanded range for more front-back variation
                "y": (-0.15, 0.15),  # Wider spread
                "z": (0.0162, 0.0162),  # Scaled block height (0.8x)
                "yaw": (-0.3, 0.3),  # Less rotation
            },
            "min_separation": 0.10,  # Increased separation for easier picking
            "asset_cfgs": [
                SceneEntityCfg("cube_1"),
                SceneEntityCfg("cube_2"),
                SceneEntityCfg("cube_3"),
            ],
        },
    )


@configclass
class CommandsCfg:
    """Empty commands config."""
    pass


@configclass
class CurriculumCfg:
    """Empty curriculum config."""
    pass


##
# Main Environment Configuration
##
@configclass 
class SOArmStackJointMimicEnvCfg(ManagerBasedRLEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment configuration for SO-ARM-101 with Joint Control.
    """

    # Scene settings
    scene: SOArmSceneCfg = SOArmSceneCfg(num_envs=1, env_spacing=2.0)
    
    # Observations
    observations: ObservationsCfg = ObservationsCfg()
    
    # Actions (set in __post_init__)
    actions: ActionsCfg = ActionsCfg()
    
    # Terminations
    terminations: TerminationsCfg = TerminationsCfg()

    # Use empty configs instead of None
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization - configure robot and actions."""
        # === Basic settings ===
        # Lower decimation = faster response (actions executed every N physics steps)
        # Lower render_interval = smoother visual feedback
        self.decimation = 2  # Reduced from 5 for lower latency
        self.episode_length_s = 60.0
        self.sim.dt = 1 / 120  # 120Hz physics
        self.sim.render_interval = 1  # Reduced from 2 for smoother rendering
        
        # === Camera settings ===
        self.viewer.eye = [-0.4, -0.4, 0.4]
        self.viewer.lookat = [0.0, 0.0, 0.15]

        # === Teleop devices configuration ===
        # Register leader_arm as a teleop device
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

        # === Robot USD path ===
        # The USD file is in isaaclab_mimic/controll_scripts/so_arm_101/SO-ARM101.usd
        import controll_scripts
        script_dir = os.path.dirname(os.path.abspath(controll_scripts.__file__))
        usd_path = os.path.join(script_dir, "so_arm_101", "SO-ARM101.usd")
        
        # === Robot configuration ===
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan": 0.055,
                    "shoulder_lift": -1.65,  # Adjusted from -1.74 to avoid limit collision on reset
                    "elbow_flex": 1.665,
                    "wrist_flex": 1.233,
                    "wrist_roll": -0.077,
                    "gripper": -0.17,  # Closed gripper for start
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

        # === Gripper configuration for grasp detection ===
        self.gripper_joint_names = ["gripper"]
        self.gripper_open_val = 1.75  # Gripper open position (from se3_leader_arm.py)
        self.gripper_threshold = 0.1  # Threshold for grasp detection

        # === Three cubes for stacking task ===
        # Cube properties matching official Franka Stack env
        
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Cube 1: Blue (bottom cube - target for stacking)
        # Position in front of robot, within SO-ARM reach
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.18, 0.0, 0.0162], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )

        # Cube 2: Red (first cube to pick and stack on blue)
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.20, 0.08, 0.0162], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )

        # Cube 3: Green (second cube to pick and stack on red)
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.22, -0.08, 0.0162], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # === End-effector frame ===
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.002, 0.0, -0.07812),
                        rot=(0.0, 0.7071068, 0.7071068, 0.0)
                    ),
                ),
                # 綁定在可動關節 (wrist_link)
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(-0.01, -0.055, 0.01727),
                        rot=(-0.5, -0.5, -0.5, 0.5)
                    ),
                ),
                # 綁定在 gripper_link
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(-0.008, 0.0, -0.07812),
                        rot=(0.0, 0.7071068, 0.7071068, 0.0)
                    ),
                ),
            ],
        )

        # === Mimic Data Generation Config ===
        self.datagen_config = DataGenConfig(
            name="demo_src_stack_so_arm_joint_D0",
            generation_guarantee=True,
            generation_keep_failed=True,
            generation_num_trials=10,
            generation_select_src_per_subtask=True,
            generation_transform_first_robot_pose=False,
            generation_interpolate_from_last_target_pose=True,
            max_num_failures=25,
            seed=1,
        )
        
        # === Subtask configuration matching official Stack task ===
        # The stack task has 4 subtasks:
        # 1. Grasp red cube (cube_2)
        # 2. Stack red cube on blue cube (cube_2 on cube_1)
        # 3. Grasp green cube (cube_3)
        # 4. Stack green cube on red cube (cube_3 on cube_2)
        subtask_configs = []
        
        # Subtask 1: Grasp red cube
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp red cube",
                next_subtask_description="Stack red cube on top of blue cube",
            )
        )
        
        # Subtask 2: Stack red on blue
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="stack_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Grasp green cube",
            )
        )
        
        # Subtask 3: Grasp green cube
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Stack green cube on top of red cube",
            )
        )
        
        # Subtask 4: Stack green on red (final subtask)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal=None,  # End of final subtask
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        
        self.subtask_configs["so_arm"] = subtask_configs
