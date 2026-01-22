# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from .assets.hand_cfg import ALLEGRO_HAND_CFG
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
import torch


@configclass
class HandEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 0.1
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1000, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = ALLEGRO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot", 
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        rot=(0.0, 0.0, 0.7071, 0.7071),
    ))
    joint_gears: list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.01

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@configclass
class HandEnvCfg_PLAY(HandEnvCfg):
    """Play configuration with reduced environments for inference."""
    def __post_init__(self) -> None:
        # Make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.0


class HandEnv(DirectRLEnv):
    cfg: HandEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        # Set start_rotation to identity (no rotation) - this is the reference frame
        self.start_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)
        # self.robot.set_joint_effort_target(torch.zeros_like(forces), joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.roll,
            self.pitch,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.motor_effort_ratio,
        )
        
        # Debug: print average reward information
        if self.num_envs == 1:
            # Calculate reward components for debugging
            roll_degrees = self.roll * 180.0 / torch.pi
            pitch_degrees = self.pitch * 180.0 / torch.pi
            roll_degrees = torch.fmod(roll_degrees + 360.0, 360.0)
            pitch_degrees = torch.fmod(pitch_degrees + 360.0, 360.0)
            
            target_roll = 90.0
            roll_error = torch.abs(roll_degrees - target_roll)
            roll_reward = self.cfg.up_weight * torch.exp(-roll_error / 20.0)
            
            pitch_to_0 = torch.abs(pitch_degrees - 0.0)
            pitch_to_360 = torch.abs(pitch_degrees - 360.0)
            pitch_error = torch.min(pitch_to_0, pitch_to_360)
            pitch_reward = self.cfg.up_weight * torch.exp(-pitch_error / 20.0)
            
            actions_cost = torch.sum(self.actions**2, dim=-1)
            electricity_cost = torch.sum(
                torch.abs(self.actions * self.dof_vel * self.cfg.dof_vel_scale) * self.motor_effort_ratio.unsqueeze(0),
                dim=-1,
            )
            dof_at_limit_cost = torch.sum(self.dof_pos_scaled > 0.98, dim=-1)
            
            print(f"Avg Roll reward: {roll_reward.mean().item():.3f} | Avg Pitch reward: {pitch_reward.mean().item():.3f} | Avg Total reward: {total_reward.mean().item():.3f}")
            print(f"Avg Loss - Actions: {actions_cost.mean().item():.3f} | Energy: {electricity_cost.mean().item():.3f} | DOF: {dof_at_limit_cost.mean().item():.3f}")
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Convert angles to degrees and normalize to 0 to 360 range
        roll_degrees = self.roll * 180.0 / torch.pi
        pitch_degrees = self.pitch * 180.0 / torch.pi
        yaw_degrees = self.yaw * 180.0 / torch.pi
        
        # Normalize to 0 to 360 range
        roll_degrees = torch.fmod(roll_degrees + 360.0, 360.0)
        pitch_degrees = torch.fmod(pitch_degrees + 360.0, 360.0)
        yaw_degrees = torch.fmod(yaw_degrees + 360.0, 360.0)
        
        # Check if robot has fallen (two axes are significantly tilted)
        # Roll: 90° is normal, allow some tolerance around 90° and 270°
        # Pitch: 351.85° is normal, allow some tolerance around 0°/360°
        roll_fallen = (roll_degrees < 30.0) | (roll_degrees > 150.0)  # More tolerance for roll
        pitch_fallen = (pitch_degrees > 60.0) & (pitch_degrees < 300.0)  # More tolerance for pitch
        
        died = roll_fallen | pitch_fallen
        
        # Debug: print average conditions
        if self.num_envs == 1:
            print(f"Avg Roll: {roll_degrees.mean().item():.1f}° | Avg Pitch: {pitch_degrees.mean().item():.1f}° | Death rate: {died.float().mean().item():.2f}")
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # Convert angles to degrees for reward calculation
    roll_degrees = roll * 180.0 / torch.pi
    pitch_degrees = pitch * 180.0 / torch.pi
    
    # Normalize to 0 to 360 range
    roll_degrees = torch.fmod(roll_degrees + 360.0, 360.0)
    pitch_degrees = torch.fmod(pitch_degrees + 360.0, 360.0)
    
    # Roll reward: encourage around 90 degrees (50-130 range)
    target_roll = 90.0  # degrees
    roll_error = torch.abs(roll_degrees - target_roll)
    roll_reward = up_weight * torch.exp(-roll_error / 20.0)  # Scale factor for smoother reward
    
    # Pitch reward: encourage around 0/360 degrees (0-40 or 320-360 range)
    # Calculate distance to closest target (0 or 360)
    pitch_to_0 = torch.abs(pitch_degrees - 0.0)
    pitch_to_360 = torch.abs(pitch_degrees - 360.0)
    pitch_error = torch.min(pitch_to_0, pitch_to_360)
    pitch_reward = up_weight * torch.exp(-pitch_error / 20.0)  # Scale factor for smoother reward
    
    up_reward = roll_reward + pitch_reward

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    
    return total_reward


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
