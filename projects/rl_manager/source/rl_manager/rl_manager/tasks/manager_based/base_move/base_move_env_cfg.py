# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Base move environment: learn forward/backward/left/right movement pattern."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg, PhysxCfg

import isaaclab.envs.mdp as base_mdp
from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.01, 0.01, 0.01)


##
# Scene definition
##

@configclass
class BaseMoveSceneCfg(InteractiveSceneCfg):
    """Configuration for the base move scene with a robot.
    
    This is the abstract base implementation, the exact robot is defined in derived classes.
    """
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    
    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    # Robot - to be set in derived config
    robot: ArticulationCfg = MISSING
    
    # End-effector frame sensor - to be set in derived config
    ee_frame: FrameTransformerCfg = MISSING


##
# MDP settings
##

@configclass
class CommandsCfg:
    movement_phase = mdp.MovementPhaseCommandCfg(
        move_distance=0.03,         # 3cm
        tolerance=0.008,            # 8mm
        absolute_height=0.02,       # 固定高度 2cm
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    arm_action: base_mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    
    IMPORTANT: Only proprioceptive observations for sim-to-real compatibility.
    NO ground truth targets or phase information.
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - sim-to-real compatible only."""
        
        # Joint states (from encoders on real robot)
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        
        # EE position relative to base (from FK on real robot)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_frame)
        
        # Last action (known on real robot)
        actions = ObsTerm(func=base_mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    
    Rewards use internal command state to shape learning.
    Policy only sees proprioceptive observations.
    """
    
    # Small alive reward to encourage survival
    alive = RewTerm(
        func=mdp.staying_alive,
        weight=0.1,
    )
    
    # Approach current phase target (exponential)
    approach_target = RewTerm(
        func=mdp.approach_phase_target,
        weight=4.0,
        params={"command_name": "movement_phase", "std": 0.02},
    )
    
    # Velocity towards target (encourages movement)
    velocity_towards = RewTerm(
        func=mdp.velocity_towards_target,
        weight=6.0,
        params={"command_name": "movement_phase"},
    )
    
    # Phase completion bonus
    phase_completed = RewTerm(
        func=mdp.phase_completed_bonus,
        weight=30.0,
        params={"command_name": "movement_phase"},
    )
    
    # --- Constraints & Penalties ---
    
    # Height constraint (stay at target height)
    height_constraint = RewTerm(
        func=mdp.height_constraint_penalty,
        weight=-200.0,   # Strong penalty for height deviation
    )
    
    # Velocity limit (2cm/s)
    velocity_limit = RewTerm(
        func=mdp.velocity_limit_penalty,
        weight=-100.0,   # Strong penalty for exceeding limit
        params={"max_velocity": 0.05},
    )
    
    # Smoothness penalties
    smooth_motion = RewTerm(
        func=mdp.smooth_motion,
        weight=-0.0001,
    )


@configclass
class EventCfg:
    """Configuration for events."""
    
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


##
# Environment configuration
##

@configclass
class BaseMoveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the base move environment."""
    
    # Scene settings
    scene: BaseMoveSceneCfg = BaseMoveSceneCfg(num_envs=1024, env_spacing=2.0)
    
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 20.0
        
        # Simulation settings
        self.sim = SimulationCfg(
            dt=1 / 60,
            gravity=(0.0, 0.0, -9.81),
            render_interval=self.decimation,)
