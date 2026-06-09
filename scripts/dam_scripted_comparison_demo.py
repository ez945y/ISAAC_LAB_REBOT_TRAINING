# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Scripted SO-ARM-101 DAM comparison demo.

This demo is built for recording, not manual teleoperation. It replays the same
end-effector target trajectory with DAM bypassed and with DAM enabled so viewers
can see why the safety layer matters.

Usage:
    python scripts/dam_scripted_comparison_demo.py --mode compare
    python scripts/dam_scripted_comparison_demo.py --mode dam --unsafe-scale 1.4
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Scripted DAM on/off comparison demo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", choices=["raw", "dam", "compare"], default="compare")
parser.add_argument("--steps", type=int, default=720, help="Steps per replay segment")
parser.add_argument("--hold-steps", type=int, default=120, help="Settling steps between compare segments")
parser.add_argument("--unsafe-scale", type=float, default=1.25, help="Scale for the scripted unsafe reach")
parser.add_argument("--log-every", type=int, default=60)
parser.add_argument(
    "--stackfile",
    type=str,
    default=None,
    help="DAM stackfile (default: bundled soarm_isaac_safety.yaml)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    parser.error("Scripted DAM comparison currently supports --num_envs 1 only.")
if args_cli.steps <= 0:
    parser.error("--steps must be positive.")
if args_cli.hold_steps < 0:
    parser.error("--hold-steps must be non-negative.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from controll_scripts import ControllerFactory, ControllerType, SOArm101Config
from controll_scripts.safety import DAMSafetyWrapper


def create_scene_cfg(robot_config: SOArm101Config) -> type:

    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )
        robot = robot_config.get_articulation_cfg(for_osc=False).replace(
            prim_path="{ENV_REGEX_NS}/robot",
        )
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.32, 0.0, 0.015)),
        )

    return SceneCfg


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _scripted_pose(initial_pose: torch.Tensor, step: int, total_steps: int, unsafe_scale: float) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    target = initial_pose.clone()

    if progress < 0.25:
        local = _smoothstep(progress / 0.25)
        target[:, 0] += 0.04 * local
        target[:, 1] += 0.03 * math.sin(progress * 8.0)
    elif progress < 0.70:
        local = _smoothstep((progress - 0.25) / 0.45)
        target[:, 0] += unsafe_scale * (0.04 + 0.18 * local)
        target[:, 1] += unsafe_scale * (0.02 + 0.12 * math.sin(local * math.pi))
        target[:, 2] -= unsafe_scale * (0.03 + 0.10 * local)
    else:
        local = _smoothstep((progress - 0.70) / 0.30)
        target[:, 0] += unsafe_scale * 0.22 * (1.0 - local)
        target[:, 1] += unsafe_scale * 0.02 * (1.0 - local)
        target[:, 2] -= unsafe_scale * 0.13 * (1.0 - local)

    target[:, 2] = torch.clamp(target[:, 2], min=0.02)
    return target


def _decision_color(decision: str) -> str:
    colors = {
        "PASS": "\033[92m",
        "CLAMP": "\033[93m",
        "REJECT": "\033[91m",
        "FAULT": "\033[95m",
        "RAW": "\033[91m",
    }
    color = colors.get(decision, "\033[90m")
    return f"{color}{decision:<6}\033[0m"


class ScriptedComparisonDemo:
    def __init__(self) -> None:
        self.robot_config = SOArm101Config()
        self.stackfile = args_cli.stackfile or "soarm_isaac_safety.yaml"
        self.sim = None
        self.scene = None
        self.robot = None
        self.controller = None
        self.dam = None
        self.marker = None
        self.sim_dt = 0.0

    def setup(self) -> bool:
        if not os.path.exists(self.robot_config.usd_path):
            print(f"[ERROR] USD not found: {self.robot_config.usd_path}")
            return False

        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([0.45, 0.38, 0.36], [0.05, 0.0, 0.12])
        self.sim_dt = self.sim.get_physics_dt()

        scene_cfg = create_scene_cfg(self.robot_config)(num_envs=1, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/World/Visuals/scripted_target_frame"
        marker_cfg.markers["frame"].scale = (0.09, 0.09, 0.09)
        self.marker = VisualizationMarkers(marker_cfg)

        self.sim.reset()
        self.robot = self.scene["robot"]
        self.robot.update(dt=self.sim_dt)

        self.controller = ControllerFactory.create(
            controller_type=ControllerType.IK,
            robot=self.robot,
            robot_config=self.robot_config,
            device=self.sim.device,
            num_envs=1,
        )
        self.dam = DAMSafetyWrapper(
            stackfile=self.stackfile,
            robot_config=self.robot_config,
            device=self.sim.device,
            task="default",
        )
        self.dam.attach_isaac_controller(self.robot, self.controller, self.robot_config)

        print("\n" + "=" * 72)
        print("  Scripted DAM Comparison Demo")
        print(f"  Mode:       {args_cli.mode.upper()}")
        print(f"  Stackfile:  {self.stackfile}")
        print("  Story: same scripted unsafe EE command, RAW first, DAM protected second.")
        print("=" * 72 + "\n")
        return True

    def run(self) -> None:
        modes = ["raw", "dam"] if args_cli.mode == "compare" else [args_cli.mode]
        for index, mode in enumerate(modes):
            self._reset_scene()
            self._hold(f"{mode.upper()} settling", args_cli.hold_steps)
            self._run_segment(mode)
            if index < len(modes) - 1:
                self._hold("transition", args_cli.hold_steps)

        self.dam.close()
        print(f"\n[DAM] {self.dam.step_count} protected steps, {self.dam.clamp_rate:.1%} clamped")

    def _reset_scene(self) -> None:
        cube: RigidObject = self.scene["cube"]
        cube_pose = torch.tensor([[0.32, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device)
        cube.write_root_pose_to_sim(cube_pose)
        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
        self.controller.reset()
        self.robot.update(self.sim_dt)

    def _hold(self, label: str, steps: int) -> None:
        if steps <= 0:
            return
        print(f"[{label}] holding for {steps} steps")
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            self.sim.step()
            self.robot.update(self.sim_dt)

    def _run_segment(self, mode: str) -> None:
        initial_pose = self.controller.current_ee_pose.clone()
        print(f"\n{'RAW COMMAND' if mode == 'raw' else 'DAM ON'} segment")

        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                return
            target_pose = _scripted_pose(initial_pose, step, args_cli.steps, args_cli.unsafe_scale)
            self._visualize_target(target_pose)

            if mode == "raw":
                self.controller.compute(target_pose, gripper_pos=0.75)
                decision = "RAW"
            else:
                self._apply_dam_target(target_pose, gripper_pos=0.75)
                decision = self.dam.last_decision

            self.sim.step()
            self.robot.update(self.sim_dt)

            if step % args_cli.log_every == 0 or step == args_cli.steps - 1:
                current_pose = self.controller.current_ee_pose
                error = torch.norm(target_pose[0, :3] - current_pose[0, :3]).item()
                tag = _decision_color(decision)
                print(
                    f"[{tag}] step={step:04d} "
                    f"target=({target_pose[0,0]:+.3f},{target_pose[0,1]:+.3f},{target_pose[0,2]:+.3f}) "
                    f"err={error:.4f} clamp_rate={self.dam.clamp_rate:.1%}"
                )

    def _apply_dam_target(self, target_pose: torch.Tensor, gripper_pos: float) -> None:
        arm_joint_ids = self.controller._arm_joint_ids
        gripper_joint_ids = self.controller._gripper_joint_ids
        current_pos = self.robot.data.joint_pos[:, arm_joint_ids]
        current_gripper = self.robot.data.joint_pos[:, gripper_joint_ids[0]].item()
        gripper_target = (
            self.controller._gripper_lower
            + gripper_pos * (self.controller._gripper_upper - self.controller._gripper_lower)
        )

        safe_targets = self.dam.filter_ee(
            target_pose,
            current_pos,
            gripper_action=gripper_target,
            gripper_obs=current_gripper,
        )
        self.robot.set_joint_position_target(safe_targets, arm_joint_ids)
        self.robot.set_joint_position_target(
            torch.tensor([[self.dam.last_safe_gripper]], device=self.sim.device),
            gripper_joint_ids,
        )
        self.robot.write_data_to_sim()

    def _visualize_target(self, target_pose: torch.Tensor) -> None:
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w,
            root_quat_w,
            target_pose[:, 0:3],
            target_pose[:, 3:7],
        )
        self.marker.visualize(target_pos_w, target_quat_w)


def main() -> None:
    demo = ScriptedComparisonDemo()
    if demo.setup():
        demo.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
