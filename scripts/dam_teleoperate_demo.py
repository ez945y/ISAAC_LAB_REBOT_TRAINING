# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Leader Arm Teleoperation + DAM Safety Guard Demo

Extends 06_teleoperate_demo.py with local DAM SafetyGuard filtering.
Every joint command — whether from joint-space or EE-space control —
passes through DAM before reaching the simulated hardware.

Install:
    Install the local DAM package that provides dam.SafetyGuard.

Usage:
    python scripts/dam_teleoperate_demo.py --controller ik
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Leader Arm Teleop + DAM Safety")
parser.add_argument(
    "--controller", type=str, default="ik", choices=["ik"],
    help="Controller type. DAM joint-target interception currently supports IK.",
)
parser.add_argument("--socket-host", type=str, default="0.0.0.0")
parser.add_argument("--socket-port", type=int, default=5359)
parser.add_argument("--position-scale", type=float, default=1.0)
parser.add_argument(
    "--stackfile", type=str, default=None,
    help="DAM stackfile (default: bundled soarm_isaac_safety.yaml)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ─────────────────────────────────────────────────

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from controll_scripts import (
    ControllerFactory,
    ControllerType,
    LeaderArmInputDevice,
    SOArm101Config,
)
from controll_scripts.safety import DAMSafetyWrapper


# ── SafeTeleoperationRunner ─────────────────────────────────────────────

class SafeTeleoperationRunner:
    """Teleoperation runner with DAM safety guard on every control step."""

    def __init__(
        self,
        robot_config: SOArm101Config,
        controller_type: ControllerType,
        input_device: LeaderArmInputDevice,
        dam: DAMSafetyWrapper,
        device: str = "cuda:0",
    ):
        self.robot_config = robot_config
        self.controller_type = controller_type
        self.input_device = input_device
        self.dam = dam
        self.device = device

        self.sim = None
        self.scene = None
        self.robot = None
        self.controller = None
        self.step_count = 0

    def setup(self) -> bool:
        if not os.path.exists(self.robot_config.usd_path):
            print(f"[ERROR] USD not found: {self.robot_config.usd_path}")
            return False

        sim_cfg = sim_utils.SimulationCfg(device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([-0.4, -0.4, 0.4], [0.0, 0.0, 0.15])
        self.sim_dt = self.sim.get_physics_dt()

        for_osc = False

        SceneCfg = self._create_scene_cfg(for_osc)
        self.scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))

        self.sim.reset()
        self.robot = self.scene["robot"]
        self.robot.update(dt=self.sim_dt)

        self.controller = ControllerFactory.create(
            controller_type=self.controller_type,
            robot=self.robot,
            robot_config=self.robot_config,
            device=self.sim.device,
            num_envs=1,
        )
        self.dam.attach_isaac_controller(self.robot, self.controller, self.robot_config)
        self.input_device.reset_target(self.controller.current_ee_pose)

        # Joint mapping
        self.arm_joint_ids, self.arm_joint_names = self.robot.find_joints(
            self.robot_config.arm_joint_names,
        )
        gripper_ids, _ = self.robot.find_joints([self.robot_config.gripper_joint_name])
        self.gripper_joint_id = gripper_ids[0]

        joint_limits = self.robot.root_physx_view.get_dof_limits()
        self.arm_lower = joint_limits[0, self.arm_joint_ids, 0].to(self.sim.device)
        self.arm_upper = joint_limits[0, self.arm_joint_ids, 1].to(self.sim.device)
        self.grip_lower = joint_limits[0, self.gripper_joint_id, 0].item()
        self.grip_upper = joint_limits[0, self.gripper_joint_id, 1].item()

        print(f"\n{'=' * 60}")
        print(f"  Teleop + DAM Safety | Controller: {self.controller_type.name}")
        print(f"  Arm joints: {self.arm_joint_names}")
        print(f"{'=' * 60}\n")
        return True

    def run(self):
        cube: RigidObject = self.scene["cube"]
        cube_init = torch.tensor(
            [[0.3, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device,
        )

        while simulation_app.is_running():
            target_pose, gripper_pos, reset = self.input_device.update()

            if reset:
                self.controller.reset()
                self.input_device.reset_target(self.controller.current_ee_pose)
                cube.write_root_pose_to_sim(cube_init)
                cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
                print("[INFO] Reset")
                continue

            data_mode = self.input_device.data_mode

            if data_mode == "joint":
                self._step_joint_safe(gripper_pos)
            else:
                self._step_ee_safe(target_pose, gripper_pos)

        self.input_device.close()
        self.dam.close()
        print(f"\n[DAM] {self.dam.step_count} steps, {self.dam.clamp_rate:.1%} clamped")

    def _step_joint_safe(self, gripper_pos: float):
        """Joint-space control with DAM safety filter."""
        norm_j = self.input_device.joint_positions.to(self.sim.device)
        arm_targets = self.arm_lower + norm_j * (self.arm_upper - self.arm_lower)
        current_pos = self.robot.data.joint_pos[0, self.arm_joint_ids]

        gripper_target = self.grip_lower + gripper_pos * (self.grip_upper - self.grip_lower)
        current_grip = self.robot.data.joint_pos[0, self.gripper_joint_id].item()

        # ── DAM filter ──
        safe_targets = self.dam.filter(
            arm_targets, current_pos,
            gripper_action=gripper_target,
            gripper_obs=current_grip,
        )

        self.robot.set_joint_position_target(safe_targets.unsqueeze(0), joint_ids=self.arm_joint_ids)
        self.robot.set_joint_position_target(
            torch.tensor([[self.dam.last_safe_gripper]], device=self.sim.device),
            joint_ids=[self.gripper_joint_id],
        )
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.sim_dt)
        self.step_count += 1

        if self.step_count % 100 == 0:
            tag = "CLAMP" if self.dam.last_clamped else "PASS "
            conn = "OK" if self.input_device.is_connected else "--"
            print(f"[DAM {tag}] joint mode | conn={conn} | clamp_rate={self.dam.clamp_rate:.1%}")

    def _step_ee_safe(self, target_pose: torch.Tensor, gripper_pos: float):
        """EE-space control through DAM resolver-backed filtering."""
        current_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        gripper_target = self.grip_lower + gripper_pos * (self.grip_upper - self.grip_lower)
        current_grip = self.robot.data.joint_pos[:, self.gripper_joint_id].item()

        safe_targets = self.dam.filter_ee(
            target_pose, current_pos,
            gripper_action=gripper_target,
            gripper_obs=current_grip,
        )

        self.robot.set_joint_position_target(safe_targets, self.arm_joint_ids)
        self.robot.set_joint_position_target(
            torch.tensor([[self.dam.last_safe_gripper]], device=self.sim.device),
            joint_ids=[self.gripper_joint_id],
        )
        self.robot.write_data_to_sim()

        self.sim.step()
        self.robot.update(self.sim_dt)
        self.step_count += 1

        if self.step_count % 200 == 0:
            error = torch.norm(target_pose[0, :3] - self.controller.current_ee_pose[0, :3]).item()
            tag = "CLAMP" if self.dam.last_clamped else "PASS "
            conn = "OK" if self.input_device.is_connected else "--"
            print(f"[DAM {tag}] ee mode | err={error:.4f} | conn={conn} | clamp_rate={self.dam.clamp_rate:.1%}")

    def _create_scene_cfg(self, for_osc: bool) -> type:
        rc = self.robot_config

        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(),
            )
            dome_light = AssetBaseCfg(
                prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0),
            )
            robot = rc.get_articulation_cfg(for_osc=for_osc).replace(
                prim_path="{ENV_REGEX_NS}/robot",
            )
            cube = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.03, 0.03, 0.03),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.015)),
            )

        return SceneCfg


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    controller_type = ControllerType.IK

    robot_config = SOArm101Config()

    initial_pose = torch.tensor([0.25, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], device=args_cli.device)
    input_device = LeaderArmInputDevice(
        initial_pose=initial_pose,
        device=args_cli.device,
        socket_host=args_cli.socket_host,
        socket_port=args_cli.socket_port,
        server_mode=True,
        position_scale=args_cli.position_scale,
    )

    stackfile = args_cli.stackfile or "soarm_isaac_safety.yaml"
    dam = DAMSafetyWrapper(
        stackfile=stackfile,
        robot_config=robot_config,
        device=args_cli.device,
        task="default",
    )

    runner = SafeTeleoperationRunner(
        robot_config=robot_config,
        controller_type=controller_type,
        input_device=input_device,
        dam=dam,
        device=args_cli.device,
    )

    if runner.setup():
        runner.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
