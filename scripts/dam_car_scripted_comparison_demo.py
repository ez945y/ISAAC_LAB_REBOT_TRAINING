# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Scripted Jetbot RAW vs safety-boundary comparison demo.

This is the vehicle version of the LinkedIn-ready safety story. It replays the
same high-speed drive command twice: raw wheel velocity control, then a
runtime safety shell that slows, steers, or stops near an obstacle.

Usage:
    python scripts/dam_car_scripted_comparison_demo.py --mode compare
    python scripts/dam_car_scripted_comparison_demo.py --mode safe --unsafe-speed 7.0
"""

import argparse
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Scripted Jetbot RAW vs safety comparison demo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", choices=["raw", "safe", "compare"], default="compare")
parser.add_argument("--steps", type=int, default=720, help="Steps per replay segment")
parser.add_argument("--hold-steps", type=int, default=90, help="Settling steps between replay segments")
parser.add_argument("--unsafe-speed", type=float, default=6.0, help="Raw forward speed command")
parser.add_argument("--turn-command", type=float, default=2.0, help="Small scripted yaw command")
parser.add_argument("--obstacle-x", type=float, default=1.35, help="Obstacle x position in meters")
parser.add_argument("--log-every", type=int, default=60)
parser.add_argument(
    "--hold-open",
    action="store_true",
    help="After the demo finishes, keep stepping (and streaming) instead of closing. "
    "Auto-enabled when --livestream is set so you can watch over the WebRTC client.",
)
parser.add_argument(
    "--summary-path",
    type=str,
    default=None,
    help="Optional path for a Markdown summary to use in posts or handoff notes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    parser.error("Scripted car safety comparison currently supports --num_envs 1 only.")
if args_cli.steps <= 0:
    parser.error("--steps must be positive.")
if args_cli.hold_steps < 0:
    parser.error("--hold-steps must be non-negative.")

# Auto-configure WebRTC livestream (publicIp + dynamic resize + no-window) when --livestream is set.
from livestream_support import apply_livestream_defaults

apply_livestream_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


JETBOT_TRACK_WIDTH = 0.12
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)


def create_scene_cfg() -> type:

    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
        obstacle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.22, 0.55, 0.35),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(args_cli.obstacle_x, 0.0, 0.175)),
        )
        safe_gate = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/SafetyGate",
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.75, 0.03),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.0)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(args_cli.obstacle_x - 0.45, 0.0, 0.015)),
        )

    return SceneCfg


@dataclass
class CarSegmentMetrics:
    mode: str
    steps: int = 0
    risky_frames: int = 0
    min_obstacle_distance: float = float("inf")
    max_raw_speed: float = 0.0
    max_safe_speed: float = 0.0
    decisions: Counter = field(default_factory=Counter)

    @property
    def interventions(self) -> int:
        return sum(self.decisions[name] for name in ("SLOW", "STEER", "STOP"))


def _scripted_drive_command(step: int, total_steps: int) -> tuple[float, float]:
    progress = step / max(total_steps - 1, 1)
    if progress < 0.18:
        speed = args_cli.unsafe_speed * progress / 0.18
    elif progress < 0.78:
        speed = args_cli.unsafe_speed
    else:
        speed = args_cli.unsafe_speed * max(0.0, 1.0 - (progress - 0.78) / 0.22)
    omega = args_cli.turn_command * math.sin(progress * math.pi * 2.0)
    return speed, omega


def _filter_vehicle_command(raw_speed: float, raw_omega: float, obstacle_distance: float) -> tuple[float, float, str]:
    if obstacle_distance < 0.25:
        return 0.0, 0.0, "STOP"
    if obstacle_distance < 0.45:
        return min(raw_speed, 0.75), raw_omega + 5.0, "STEER"
    if obstacle_distance < 0.70:
        return min(raw_speed, 1.5), raw_omega, "SLOW"
    return raw_speed, raw_omega, "PASS"


def _wheel_targets(speed: float, omega: float, device: str) -> torch.Tensor:
    left = speed - omega * JETBOT_TRACK_WIDTH / 2.0
    right = speed + omega * JETBOT_TRACK_WIDTH / 2.0
    return torch.tensor([[left, right]], dtype=torch.float32, device=device)


def _decision_color(decision: str) -> str:
    colors = {
        "RAW": "\033[91m",
        "PASS": "\033[92m",
        "SLOW": "\033[93m",
        "STEER": "\033[95m",
        "STOP": "\033[91m",
    }
    color = colors.get(decision, "\033[90m")
    return f"{color}{decision:<6}\033[0m"


class CarSafetyComparisonDemo:
    def __init__(self) -> None:
        self.sim = None
        self.scene = None
        self.jetbot = None
        self.obstacle = None
        self.sim_dt = 0.0
        self.metrics: list[CarSegmentMetrics] = []

    def setup(self) -> bool:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([2.2, -1.2, 1.0], [0.7, 0.0, 0.1])
        self.sim_dt = self.sim.get_physics_dt()

        scene_cfg = create_scene_cfg()(num_envs=1, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.jetbot = self.scene["Jetbot"]
        self.obstacle = self.scene["obstacle"]
        self.jetbot.update(dt=self.sim_dt)

        print("\n" + "=" * 72)
        print("  Scripted Car Safety Comparison Demo")
        print(f"  Mode:          {args_cli.mode.upper()}")
        print(f"  Unsafe speed:  {args_cli.unsafe_speed:.2f}")
        print(f"  Obstacle x:    {args_cli.obstacle_x:.2f}")
        print("  Story: same vehicle command, RAW first, safety boundary second.")
        print("=" * 72 + "\n")
        return True

    def run(self) -> None:
        modes = ["raw", "safe"] if args_cli.mode == "compare" else [args_cli.mode]
        for index, mode in enumerate(modes):
            self._reset_scene()
            self._hold(f"{mode.upper()} settling", args_cli.hold_steps)
            self.metrics.append(self._run_segment(mode))
            if index < len(modes) - 1:
                self._hold("transition", args_cli.hold_steps)

        summary = self._format_linkedin_summary()
        print(summary)
        if args_cli.summary_path:
            path = Path(args_cli.summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(summary, encoding="utf-8")
            print(f"[INFO] Wrote car summary: {path}")

    def _reset_scene(self) -> None:
        root_pose = self.jetbot.data.default_root_pose.torch.clone()
        root_pose[:, :3] += self.scene.env_origins
        self.jetbot.write_root_pose_to_sim_index(root_pose=root_pose)
        self.jetbot.write_root_velocity_to_sim_index(root_velocity=self.jetbot.data.default_root_vel.torch.clone())
        self.jetbot.write_joint_position_to_sim_index(position=self.jetbot.data.default_joint_pos.torch.clone())
        self.jetbot.write_joint_velocity_to_sim_index(velocity=self.jetbot.data.default_joint_vel.torch.clone())
        obstacle_pose = torch.tensor([[args_cli.obstacle_x, 0.0, 0.175, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device)
        self.obstacle.write_root_pose_to_sim_index(root_pose=obstacle_pose)
        self.obstacle.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.sim.device))
        self.scene.reset()
        self.jetbot.update(dt=self.sim_dt)

    def _hold(self, label: str, steps: int) -> None:
        if steps <= 0:
            return
        print(f"[{label}] holding for {steps} steps")
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

    def _run_segment(self, mode: str) -> CarSegmentMetrics:
        metrics = CarSegmentMetrics(mode=mode)
        print(f"\n{'RAW COMMAND' if mode == 'raw' else 'SAFETY ON'} segment")
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                return metrics

            raw_speed, raw_omega = _scripted_drive_command(step, args_cli.steps)
            distance = self._obstacle_distance()
            if mode == "raw":
                speed, omega, decision = raw_speed, raw_omega, "RAW"
            else:
                speed, omega, decision = _filter_vehicle_command(raw_speed, raw_omega, distance)

            self.jetbot.set_joint_velocity_target_index(target=_wheel_targets(speed, omega, self.sim.device))
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

            metrics.steps += 1
            metrics.min_obstacle_distance = min(metrics.min_obstacle_distance, distance)
            metrics.max_raw_speed = max(metrics.max_raw_speed, abs(raw_speed))
            metrics.max_safe_speed = max(metrics.max_safe_speed, abs(speed))
            metrics.decisions[decision] += 1
            if distance < 0.70:
                metrics.risky_frames += 1

            if step % args_cli.log_every == 0 or step == args_cli.steps - 1:
                tag = _decision_color(decision)
                print(
                    f"[{tag}] step={step:04d} "
                    f"raw_v={raw_speed:+.2f} safe_v={speed:+.2f} "
                    f"dist={distance:.3f}m interventions={metrics.interventions}"
                )

        self._print_segment_summary(metrics)
        return metrics

    def _obstacle_distance(self) -> float:
        car_pos = self.jetbot.data.root_pos_w[0, :2]
        obstacle_pos = self.obstacle.data.root_pos_w[0, :2]
        return max(0.0, torch.norm(obstacle_pos - car_pos).item() - 0.18)

    def _print_segment_summary(self, metrics: CarSegmentMetrics) -> None:
        decision_text = ", ".join(
            f"{name}={metrics.decisions[name]}"
            for name in ("RAW", "PASS", "SLOW", "STEER", "STOP")
            if metrics.decisions[name]
        ) or "none"
        print(
            f"[SUMMARY {metrics.mode.upper()}] "
            f"steps={metrics.steps} risky_frames={metrics.risky_frames} "
            f"min_obstacle_distance={metrics.min_obstacle_distance:.3f}m "
            f"max_raw_speed={metrics.max_raw_speed:.2f} "
            f"max_safe_speed={metrics.max_safe_speed:.2f} "
            f"decisions={decision_text}"
        )

    def _format_linkedin_summary(self) -> str:
        raw = next((item for item in self.metrics if item.mode == "raw"), None)
        safe = next((item for item in self.metrics if item.mode == "safe"), None)
        risky_frames = max((item.risky_frames for item in self.metrics), default=0)
        interventions = safe.interventions if safe is not None else 0
        safe_steps = safe.steps if safe is not None else 0
        intervention_rate = interventions / safe_steps if safe_steps else 0.0

        lines = [
            "",
            "LINKEDIN CAR DEMO SUMMARY",
            "Problem: autonomous and teleop vehicles can receive risky speed commands near obstacles.",
            "Demo: replay the same drive command twice, first raw and then through a runtime safety boundary.",
            f"Risky proximity frames: {risky_frames}",
            f"Safety interventions: {interventions} ({intervention_rate:.1%} of SAFE frames)",
        ]
        if raw is not None:
            lines.append(
                f"RAW closest obstacle distance: {raw.min_obstacle_distance:.3f}m; "
                f"max speed command: {raw.max_raw_speed:.2f}"
            )
        if safe is not None:
            lines.append(
                f"SAFE decisions: PASS={safe.decisions['PASS']} SLOW={safe.decisions['SLOW']} "
                f"STEER={safe.decisions['STEER']} STOP={safe.decisions['STOP']}"
            )
        if safe is not None and interventions == 0:
            lines.append(
                "Demo tuning note: no safety intervention was observed; increase --unsafe-speed "
                "or move --obstacle-x closer before recording."
            )
        lines.extend(
            [
                "Caption angle: same command, safer vehicle. Safety belongs in the control path, not only in logs.",
                "",
            ]
        )
        return "\n".join(lines)


def main() -> None:
    from livestream_support import is_livestreaming

    demo = CarSafetyComparisonDemo()
    if demo.setup():
        demo.run()
        # Keep stepping so the final framed scene stays visible to the WebRTC
        # client instead of vanishing the instant the demo ends.
        if args_cli.hold_open or is_livestreaming(args_cli):
            print("[demo] holding open for livestream -- press Ctrl+C to exit.", flush=True)
            while simulation_app.is_running():
                demo.sim.step()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
