# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Leader Arm Teleoperation Demo (Object-Oriented Refactor)

This script demonstrates teleoperation of any robot arm using a leader arm.
The leader arm data is received from a socket connection.

Key Design:
- TeleoperationRunner: Orchestrates the simulation loop, decoupled from robot specifics.
- RobotConfig: Defines robot-specific properties (joint names, limits, etc.).
- InputDevice: Provides normalized joint/EE data from external hardware.

Usage:
    python 06_teleoperate_demo.py --server-mode
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

from isaaclab.app import AppLauncher

# Add parent directory to path for controll_scripts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Leader Arm Teleoperation Demo')
parser.add_argument(
    '--controller',
    type=str,
    default='ik',
    choices=['ik', 'osc'],
    help='Controller type: ik (Differential IK) or osc (Operational Space)',
)
parser.add_argument(
    '--socket-host',
    type=str,
    default='0.0.0.0',
    help='Socket host address (server: bind address, client: target host)',
)
parser.add_argument(
    '--socket-port',
    type=int,
    default=5359,
    help='Socket port for leader arm data (default: 5359)',
)
parser.add_argument(
    '--position-scale',
    type=float,
    default=1.0,
    help='Scale factor for leader arm position (default: 1.0)',
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Auto-configure WebRTC livestream (publicIp + dynamic resize) when --livestream is set.
from tools.livestream.livestream_support import apply_livestream_defaults
apply_livestream_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Import controll_scripts library
from controll_scripts import (
    ControllerFactory,
    ControllerType,
    LeaderArmInputDevice,
    SOArm101Config,
)
from controll_scripts.configs.base import BaseRobotConfig
from controll_scripts.utils import physx_to_torch


# ========================================
# Teleoperation Runner (Decoupled from Robot)
# ========================================

class TeleoperationRunner:
    """
    Manages the teleoperation loop, decoupled from specific robot configurations.

    Attributes:
        robot_config: A BaseRobotConfig subclass instance defining the robot.
        controller_type: The type of controller to use (IK or OSC).
        input_device: The input device providing control signals.
    """

    def __init__(
        self,
        robot_config: BaseRobotConfig,
        controller_type: ControllerType,
        input_device: LeaderArmInputDevice,
        device: str = 'cuda:0',
    ):
        self.robot_config = robot_config
        self.controller_type = controller_type
        self.input_device = input_device
        self.device = device

        # Simulation objects (initialized in setup)
        self.sim = None
        self.scene = None
        self.robot = None
        self.controller = None
        self.target_marker = None

        # Joint mapping (populated from config)
        self.arm_joint_ids: Optional[List[int]] = None
        self.arm_joint_names: Optional[List[str]] = None
        self.gripper_joint_id: Optional[int] = None
        self.arm_joint_lower: Optional[torch.Tensor] = None
        self.arm_joint_upper: Optional[torch.Tensor] = None
        self.gripper_lower: float = 0.0
        self.gripper_upper: float = 1.0

        # State
        self.sim_dt: float = 0.0
        self.step_count: int = 0
        self._connection_warned: bool = False
        self._mode_announced: bool = False

    def setup(self) -> bool:
        """Initialize simulation, scene, robot, and controller."""
        # Check USD file
        if not os.path.exists(self.robot_config.usd_path):
            print(f'[ERROR]: USD file not found: {self.robot_config.usd_path}')
            return False

        # Initialize simulation
        sim_cfg = sim_utils.SimulationCfg(device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([-0.4, -0.4, 0.4], [0.0, 0.0, 0.15])
        self.sim_dt = self.sim.get_physics_dt()

        # Determine stiffness based on mode
        for_osc = self.controller_type == ControllerType.OSC
        if self.input_device.data_mode == 'joint':
            for_osc = False  # Force stiffness ON for joint mode
            print('[INFO]: JOINT mode detected. Forcing stiffness ON.')

        SceneCfg = self._create_scene_cfg(for_osc)
        scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)

        # Create target marker
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.prim_path = '/World/Visuals/target_frame'
        frame_marker_cfg.markers['frame'].scale = (0.1, 0.1, 0.1)
        self.target_marker = VisualizationMarkers(frame_marker_cfg)

        # Reset simulation
        self.sim.reset()

        # Get robot
        self.robot = self.scene['robot']
        self.robot.update(dt=self.sim_dt)

        self.controller = ControllerFactory.create(
            controller_type=self.controller_type,
            robot=self.robot,
            robot_config=self.robot_config,
            device=self.sim.device,
            num_envs=1,
        )

        # Update input device initial pose
        self.input_device.reset_target(self.controller.current_ee_pose)

        # --- Joint Mapping (from config, not hardcoded) ---
        self._setup_joint_mapping()

        self._print_info()
        return True

    def _create_scene_cfg(self, for_osc: bool) -> type:
        """Dynamically create scene configuration class."""
        robot_config = self.robot_config

        class DynamicSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path='/World/defaultGroundPlane',
                spawn=sim_utils.GroundPlaneCfg(),
            )
            dome_light = AssetBaseCfg(
                prim_path='/World/Light',
                spawn=sim_utils.DomeLightCfg(intensity=3000.0),
            )
            robot = robot_config.get_articulation_cfg(for_osc=for_osc).replace(
                prim_path='{ENV_REGEX_NS}/robot'
            )
            cube = RigidObjectCfg(
                prim_path='{ENV_REGEX_NS}/cube',
                spawn=sim_utils.CuboidCfg(
                    size=(0.03, 0.03, 0.03),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.015)),
            )

        return DynamicSceneCfg

    def _setup_joint_mapping(self):
        """Setup joint IDs and limits from robot config."""
        # Get arm joint IDs from config-defined names
        self.arm_joint_ids, self.arm_joint_names = self.robot.find_joints(
            self.robot_config.arm_joint_names
        )
        # Get gripper joint ID
        gripper_ids, _ = self.robot.find_joints([self.robot_config.gripper_joint_name])
        self.gripper_joint_id = gripper_ids[0] if gripper_ids else 0

        # Get joint limits from physics
        joint_limits = physx_to_torch(self.robot.root_physx_view.get_dof_limits())
        self.arm_joint_lower = joint_limits[0, self.arm_joint_ids, 0].to(self.sim.device)
        self.arm_joint_upper = joint_limits[0, self.arm_joint_ids, 1].to(self.sim.device)
        self.gripper_lower = joint_limits[0, self.gripper_joint_id, 0].item()
        self.gripper_upper = joint_limits[0, self.gripper_joint_id, 1].item()

    def _print_info(self):
        """Print initialization info."""
        print(f'\nRobot: {self.robot_config.name}')
        print(f'Arm Joints ({len(self.arm_joint_names)}): {self.arm_joint_names}')
        print(f'Gripper Joint: {self.robot_config.gripper_joint_name}')
        print(f'Joint Limits (lower): {self.arm_joint_lower}')
        print(f'Joint Limits (upper): {self.arm_joint_upper}')
        print('Starting control loop...\n')

    def run(self):
        """Main control loop."""
        cube: RigidObject = self.scene['cube']
        cube_initial_pose = torch.tensor(
            [[0.3, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device
        )

        while simulation_app.is_running():
            self._check_connection()

            # Update input
            target_pose, gripper_pos, reset_requested = self.input_device.update()
            data_mode = self.input_device.data_mode

            self._announce_mode(data_mode)

            # Handle reset
            if reset_requested:
                self.controller.reset()
                self.input_device.reset_target(self.controller.current_ee_pose)
                cube.write_root_pose_to_sim(cube_initial_pose)
                cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
                print('[INFO]: Scene reset')
                continue

            if data_mode == 'joint':
                self._step_joint_mode(gripper_pos)
            else:
                self._step_ee_mode(target_pose, gripper_pos)

        # Cleanup
        self.input_device.close()

    def _check_connection(self):
        """Check and log connection status."""
        if not self.input_device.is_connected:
            if not self._connection_warned:
                print('[WARNING] Not connected to leader arm sender.')
                self._connection_warned = True
        else:
            if self._connection_warned:
                print('[INFO] Connected to leader arm sender!')
                self._connection_warned = False

    def _announce_mode(self, data_mode: str):
        """Announce the control mode once."""
        if not self._mode_announced and self.input_device.is_connected:
            if data_mode == 'joint':
                print('[INFO] Running in JOINT mode - direct joint control')
            else:
                print('[INFO] Running in EE mode - using IK/OSC controller')
            self._mode_announced = True

    def _step_joint_mode(self, gripper_pos: float):
        """Execute one step in joint control mode."""
        # Get normalized joint positions from input device
        norm_j = self.input_device.joint_positions.to(self.sim.device)

        # Denormalize to actual radians using limits from physics
        arm_targets = self.arm_joint_lower + norm_j * (self.arm_joint_upper - self.arm_joint_lower)

        # Apply targets
        self.robot.set_joint_position_target_index(arm_targets, joint_ids=self.arm_joint_ids)

        # Gripper
        gripper_target = self.gripper_lower + gripper_pos * (self.gripper_upper - self.gripper_lower)
        self.robot.set_joint_position_target_index(
            torch.tensor([[gripper_target]], device=self.sim.device),
            joint_ids=[self.gripper_joint_id],
        )

        # Write and step
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.sim_dt)
        self.step_count += 1

        # Debug output
        if self.step_count % 100 == 0:
            self._print_joint_status(norm_j, arm_targets, gripper_pos, gripper_target)

    def _step_ee_mode(self, target_pose: torch.Tensor, gripper_pos: float):
        """Execute one step in end-effector control mode."""
        # Update target marker
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        import isaaclab.utils.math as math_utils
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, target_pose[:, 0:3], target_pose[:, 3:7]
        )
        self.target_marker.visualize(target_pos_w, target_quat_w)

        # Compute and apply control
        self.controller.compute(target_pose, gripper_pos)

        # Step simulation
        self.sim.step()
        self.robot.update(self.sim_dt)
        self.step_count += 1

        # Debug output
        if self.step_count % 200 == 0:
            current_pose = self.controller.current_ee_pose
            error = torch.norm(target_pose[0, 0:3] - current_pose[0, 0:3]).item()
            conn_status = 'Connected' if self.input_device.is_connected else 'Disconnected'
            print(
                f'EE Mode | Err: {error:.4f} | Grip: {gripper_pos:.2f} | {conn_status}'
            )

    def _print_joint_status(
        self,
        norm_j: torch.Tensor,
        arm_targets: torch.Tensor,
        gripper_pos: float,
        gripper_target: float,
    ):
        """Print a formatted table of joint status."""
        current_joints = self.robot.data.joint_pos[0, self.arm_joint_ids]
        current_gripper = self.robot.data.joint_pos[0, self.gripper_joint_id].item()

        joint_errors = (arm_targets - current_joints).abs()
        gripper_error = abs(gripper_target - current_gripper)
        total_error = torch.norm(joint_errors).item()

        conn_status = 'Connected' if self.input_device.is_connected else 'Disconnected'
        age = self.input_device.last_data_age

        print(f"\n{'-'*35} ROBOT STATUS {'-'*35}")
        print(f'Mode: JOINT     | Conn: {conn_status:<12} | Age: {age:5.3f}s | Error: {total_error:6.4f}')
        header = f"{'Joint Name':<15} {'Norm':>8} {'Target':>10} {'Actual':>10} {'Error':>10} {'Limits':>22}"
        print(header)
        print('-' * len(header))

        for i, name in enumerate(self.arm_joint_names):
            lim_str = f'{self.arm_joint_lower[i]:.2f} / {self.arm_joint_upper[i]:.2f}'
            print(
                f'{name:<15} {norm_j[i]:8.2f} {arm_targets[i]:10.3f} '
                f'{current_joints[i]:10.3f} {joint_errors[i]:10.4f} {lim_str:>22}'
            )

        grip_lim = f'{self.gripper_lower:.2f} / {self.gripper_upper:.2f}'
        print(
            f"{'gripper':<15} {gripper_pos:8.2f} {gripper_target:10.3f} "
            f'{current_gripper:10.3f} {gripper_error:10.4f} {grip_lim:>22}'
        )
        print('-' * 105)


# ========================================
# Main Entry Point
# ========================================

def main():
    # Select controller type
    if args_cli.controller == 'ik':
        controller_type = ControllerType.IK
    else:
        controller_type = ControllerType.OSC

    print(f"\n{'='*60}")
    print('Leader Arm Teleoperation Demo')
    print(f"{'='*60}")
    print(f'Controller: {args_cli.controller.upper()}')
    print(f'Socket: {args_cli.socket_host}:{args_cli.socket_port}')
    print(f'Position Scale: {args_cli.position_scale}')
    print(f"{'='*60}\n")

    # --- Robot Configuration (easily swappable) ---
    robot_config = SOArm101Config()

    # --- Input Device ---
    initial_pose = torch.tensor([0.25, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], device=args_cli.device)
    input_device = LeaderArmInputDevice(
        initial_pose=initial_pose,
        device=args_cli.device,
        socket_host=args_cli.socket_host,
        socket_port=args_cli.socket_port,
        server_mode=True,  # Always server mode
        position_scale=args_cli.position_scale,
    )

    runner = TeleoperationRunner(
        robot_config=robot_config,
        controller_type=controller_type,
        input_device=input_device,
        device=args_cli.device,
    )

    if runner.setup():
        runner.run()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
