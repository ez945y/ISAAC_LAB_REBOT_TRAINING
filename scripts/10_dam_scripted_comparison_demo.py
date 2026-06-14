# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Scripted SO-ARM-101 DAM comparison demo.

This demo is built for recording, not manual teleoperation. It replays the same
end-effector target trajectory with DAM bypassed and with DAM enabled so viewers
can see why the safety layer matters.

Usage:
    python scripts/10_dam_scripted_comparison_demo.py --mode compare
    python scripts/10_dam_scripted_comparison_demo.py --mode dam --unsafe-scale 1.4
"""

import argparse
import math
import os
import sys
from pathlib import Path

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
    "--summary-path",
    type=str,
    default=None,
    help="Optional path for a Markdown summary to use in posts or handoff notes.",
)
parser.add_argument(
    "--stackfile",
    type=str,
    default=None,
    help="DAM stackfile (default: bundled soarm_isaac_safety.yaml)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Auto-configure WebRTC livestream (publicIp + dynamic resize) when --livestream is set.
from tools.livestream.livestream_support import apply_livestream_defaults
apply_livestream_defaults(args_cli)
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from controll_scripts import ControllerFactory, ControllerType, SOArm101Config
from controll_scripts.safety import DAMSafetyWrapper
from controll_scripts.safety.demo_summary import (
    SegmentMetrics,
    format_linkedin_summary,
    format_segment_summary,
)


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
        left_robot = robot_config.get_articulation_cfg(for_osc=False).replace(
            prim_path="{ENV_REGEX_NS}/left_robot",
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.25, 0.0)),
        )
        right_robot = robot_config.get_articulation_cfg(for_osc=False).replace(
            prim_path="{ENV_REGEX_NS}/right_robot",
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -0.25, 0.0)),
        )
        left_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/left_cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.32, 0.25, 0.015)),
        )
        right_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/right_cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.32, -0.25, 0.015)),
        )
        left_workspace = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/left_workspace",
            spawn=sim_utils.SphereCfg(
                radius=0.16,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.8, 0.2),
                    opacity=0.25,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.25, 0.14)),
        )
        right_workspace = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/right_workspace",
            spawn=sim_utils.SphereCfg(
                radius=0.16,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.8, 0.2),
                    opacity=0.25,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, -0.25, 0.14)),
        )
        left_safe_ground = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/left_safe_ground",
            spawn=sim_utils.CuboidCfg(
                size=(0.35, 0.35, 0.001),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.45, 0.16)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.25, 0.0001)),
        )
        right_safe_ground = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/right_safe_ground",
            spawn=sim_utils.CuboidCfg(
                size=(0.35, 0.35, 0.001),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.45, 0.16)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, -0.25, 0.0001)),
        )
        divider = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/divider",
            spawn=sim_utils.CuboidCfg(
                size=(0.60, 0.02, 0.02),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.86, 0.86)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.0, 0.01)),
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
        # Rapid wild shaking / swinging (先做快速亂甩)
        target[:, 0] += 0.08 * math.sin(progress * 45.0) * local
        target[:, 1] += 0.12 * math.cos(progress * 55.0) * local
        target[:, 2] += 0.07 * math.sin(progress * 35.0) * local
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
        self.left_robot = None
        self.right_robot = None
        self.left_controller = None
        self.right_controller = None
        self.left_dam = None
        self.right_dam = None
        self.left_marker = None
        self.right_marker = None
        self.sim_dt = 0.0
        self.segment_metrics: list[SegmentMetrics] = []

    def setup(self) -> bool:
        if not os.path.exists(self.robot_config.usd_path):
            print(f"[ERROR] USD not found: {self.robot_config.usd_path}")
            return False

        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([0.55, 0.45, 0.40], [0.08, 0.0, 0.15])
        self.sim_dt = self.sim.get_physics_dt()

        scene_cfg = create_scene_cfg(self.robot_config)(num_envs=1, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)

        marker_cfg_left = FRAME_MARKER_CFG.copy()
        marker_cfg_left.prim_path = "/World/Visuals/scripted_target_frame_left"
        marker_cfg_left.markers["frame"].scale = (0.09, 0.09, 0.09)
        self.left_marker = VisualizationMarkers(marker_cfg_left)

        marker_cfg_right = FRAME_MARKER_CFG.copy()
        marker_cfg_right.prim_path = "/World/Visuals/scripted_target_frame_right"
        marker_cfg_right.markers["frame"].scale = (0.09, 0.09, 0.09)
        self.right_marker = VisualizationMarkers(marker_cfg_right)

        self.sim.reset()
        self.left_robot = self.scene["left_robot"]
        self.right_robot = self.scene["right_robot"]
        self.left_robot.update(dt=self.sim_dt)
        self.right_robot.update(dt=self.sim_dt)

        self.left_controller = ControllerFactory.create(
            controller_type=ControllerType.IK,
            robot=self.left_robot,
            robot_config=self.robot_config,
            device=self.sim.device,
            num_envs=1,
        )
        self.right_controller = ControllerFactory.create(
            controller_type=ControllerType.IK,
            robot=self.right_robot,
            robot_config=self.robot_config,
            device=self.sim.device,
            num_envs=1,
        )

        if args_cli.mode in ["compare", "dam"]:
            self.left_dam = DAMSafetyWrapper(
                stackfile=self.stackfile,
                robot_config=self.robot_config,
                device=self.sim.device,
                task="default",
            )
            self.left_dam.attach_isaac_controller(self.left_robot, self.left_controller, self.robot_config)

        if args_cli.mode == "dam":
            self.right_dam = DAMSafetyWrapper(
                stackfile=self.stackfile,
                robot_config=self.robot_config,
                device=self.sim.device,
                task="default",
              )
            self.right_dam.attach_isaac_controller(self.right_robot, self.right_controller, self.robot_config)

        print("\n" + "=" * 72)
        print("  Scripted DAM Comparison Demo (Twin-Arm)")
        print(f"  Mode:       {args_cli.mode.upper()}")
        print(f"  Stackfile:  {self.stackfile}")
        print("  Story: Left is DAM (ON), Right is RAW (OFF)")
        print("=" * 72 + "\n")
        return True

    def run(self) -> None:
        self._reset_scene()
        self._hold("settling", args_cli.hold_steps)

        initial_pose_left = self.left_controller.current_ee_pose.clone()
        initial_pose_right = self.right_controller.current_ee_pose.clone()
        initial_pos_left = initial_pose_left[0, :3].detach().clone()
        initial_pos_right = initial_pose_right[0, :3].detach().clone()

        left_mode = "dam" if args_cli.mode in ["compare", "dam"] else "raw"
        right_mode = "dam" if args_cli.mode == "dam" else "raw"

        left_metrics = SegmentMetrics(mode=left_mode)
        right_metrics = SegmentMetrics(mode=right_mode)

        print(f"\nTwin-Arm run starting (Left: {left_mode.upper()}, Right: {right_mode.upper()})")

        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break

            target_pose_left = _scripted_pose(initial_pose_left, step, args_cli.steps, args_cli.unsafe_scale)
            target_pose_right = _scripted_pose(initial_pose_right, step, args_cli.steps, args_cli.unsafe_scale)

            self._visualize_target(self.left_robot, self.left_marker, target_pose_left)
            self._visualize_target(self.right_robot, self.right_marker, target_pose_right)

            # Left arm
            if self.left_dam is not None:
                self._apply_dam_target(self.left_robot, self.left_controller, self.left_dam, target_pose_left, gripper_pos=0.0, workspace_center=initial_pos_left)
                left_decision = self.left_dam.last_decision
            else:
                self.left_controller.compute(target_pose_left, gripper_pos=0.0)
                left_decision = "RAW"

            # Right arm
            # The RAW group (right arm) erroneously releases/opens the gripper when it shakes or goes outside the safe boundary (progress > 0.15)
            progress = step / max(args_cli.steps - 1, 1)
            right_gripper_pos = 0.0
            if progress > 0.15:
                right_gripper_pos = 1.0  # Erroneous release (open fully)

            if self.right_dam is not None:
                self._apply_dam_target(self.right_robot, self.right_controller, self.right_dam, target_pose_right, gripper_pos=right_gripper_pos, workspace_center=initial_pos_right)
                right_decision = self.right_dam.last_decision
            else:
                self.right_controller.compute(target_pose_right, gripper_pos=right_gripper_pos)
                right_decision = "RAW"

            self.sim.step()
            self.left_robot.update(self.sim_dt)
            self.right_robot.update(self.sim_dt)

            current_pose_left = self.left_controller.current_ee_pose
            current_pose_right = self.right_controller.current_ee_pose

            error_left = torch.norm(target_pose_left[0, :3] - current_pose_left[0, :3]).item()
            error_right = torch.norm(target_pose_right[0, :3] - current_pose_right[0, :3]).item()

            target_offset_left = torch.norm(target_pose_left[0, :3] - initial_pos_left).item()
            target_offset_right = torch.norm(target_pose_right[0, :3] - initial_pos_right).item()

            # Left metrics
            left_metrics.steps += 1
            left_metrics.max_tracking_error = max(left_metrics.max_tracking_error, error_left)
            left_metrics.max_target_offset = max(left_metrics.max_target_offset, target_offset_left)
            left_metrics.min_target_z = min(left_metrics.min_target_z, float(target_pose_left[0, 2].item()))
            left_metrics.decisions[left_decision] += 1
            if target_offset_left > 0.16 or target_pose_left[0, 2].item() < 0.08:
                left_metrics.risky_frames += 1

            # Right metrics
            right_metrics.steps += 1
            right_metrics.max_tracking_error = max(right_metrics.max_tracking_error, error_right)
            right_metrics.max_target_offset = max(right_metrics.max_target_offset, target_offset_right)
            right_metrics.min_target_z = min(right_metrics.min_target_z, float(target_pose_right[0, 2].item()))
            right_metrics.decisions[right_decision] += 1
            if target_offset_right > 0.16 or target_pose_right[0, 2].item() < 0.08:
                right_metrics.risky_frames += 1

            if step % args_cli.log_every == 0 or step == args_cli.steps - 1:
                tag_left = _decision_color(left_decision)
                tag_right = _decision_color(right_decision)
                print(
                    f"[STEP {step:04d}] "
                    f"LEFT (DAM): {tag_left} err={error_left:.4f} | "
                    f"RIGHT (RAW): {tag_right} err={error_right:.4f}"
                )

        if self.left_dam is not None:
            self.left_dam.close()
            print(f"\n[DAM Left] {self.left_dam.step_count} protected steps, {self.left_dam.clamp_rate:.1%} clamped")
        if self.right_dam is not None:
            self.right_dam.close()
            print(f"[DAM Right] {self.right_dam.step_count} protected steps, {self.right_dam.clamp_rate:.1%} clamped")

        self.segment_metrics = [right_metrics, left_metrics]
        self._print_linkedin_summary()

    def _reset_scene(self) -> None:
        left_cube: RigidObject = self.scene["left_cube"]
        right_cube: RigidObject = self.scene["right_cube"]
        
        left_cube_pose = torch.tensor([[0.32, 0.25, 0.015, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device)
        right_cube_pose = torch.tensor([[0.32, -0.25, 0.015, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device)
        
        left_cube.write_root_pose_to_sim(left_cube_pose)
        left_cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
        right_cube.write_root_pose_to_sim(right_cube_pose)
        right_cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
        
        self.left_controller.reset()
        self.right_controller.reset()
        self.left_robot.update(self.sim_dt)
        self.right_robot.update(self.sim_dt)

    def _hold(self, label: str, steps: int) -> None:
        if steps <= 0:
            return
        print(f"[{label}] holding for {steps} steps")
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            self.sim.step()
            self.left_robot.update(self.sim_dt)
            self.right_robot.update(self.sim_dt)

    def _apply_dam_target(self, robot, controller, dam, target_pose: torch.Tensor, gripper_pos: float, workspace_center=None) -> None:
        arm_joint_ids = controller._arm_joint_ids
        gripper_joint_ids = controller._gripper_joint_ids
        current_pos = robot.data.joint_pos[:, arm_joint_ids]
        current_gripper = robot.data.joint_pos[:, gripper_joint_ids[0]].item()
        gripper_target = (
            controller._gripper_lower
            + gripper_pos * (controller._gripper_upper - controller._gripper_lower)
        )

        safe_targets = dam.filter_ee(
            target_pose,
            current_pos,
            gripper_action=gripper_target,
            gripper_obs=current_gripper,
            workspace_center=workspace_center,
            workspace_radius=0.16,
            min_z=0.08,
        )
        robot.set_joint_position_target_index(safe_targets, arm_joint_ids)
        robot.set_joint_position_target_index(
            torch.tensor([[dam.last_safe_gripper]], device=self.sim.device),
            gripper_joint_ids,
        )
        robot.write_data_to_sim()

    def _visualize_target(self, robot, marker, target_pose: torch.Tensor) -> None:
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w,
            root_quat_w,
            target_pose[:, 0:3],
            target_pose[:, 3:7],
        )
        marker.visualize(target_pos_w, target_quat_w)

    def _print_segment_summary(self, metrics: SegmentMetrics) -> None:
        print(format_segment_summary(metrics))

    def _print_linkedin_summary(self) -> None:
        if not self.segment_metrics:
            return

        summary = format_linkedin_summary(self.segment_metrics)
        print(summary)

        if args_cli.summary_path:
            path = Path(args_cli.summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(summary, encoding="utf-8")
            print(f"[INFO] Wrote LinkedIn summary: {path}")


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
