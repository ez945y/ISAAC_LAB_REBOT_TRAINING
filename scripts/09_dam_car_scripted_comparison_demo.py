# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Twin-lane Jetbot DAM intervention demo.

Two Jetbots receive the exact same scripted wheel-velocity command stream:

* RAW lane: action goes straight into Isaac.
* DAM lane: action is intercepted by robot-dam's SafetyGuard first.

This is intentionally a visual A/B test, not a terminal-only replay.

Usage:
    python scripts/09_dam_car_scripted_comparison_demo.py
    python scripts/09_dam_car_scripted_comparison_demo.py --unsafe-speed 7.0 --turn-command 5.0
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

parser = argparse.ArgumentParser(description="Twin-lane Jetbot RAW vs DAM demo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=900, help="Scripted replay length")
parser.add_argument("--unsafe-speed", type=float, default=6.5, help="Scripted raw forward speed")
parser.add_argument("--turn-command", type=float, default=4.5, help="Scripted raw yaw command")
parser.add_argument("--log-every", type=int, default=60)
parser.add_argument(
    "--stackfile",
    type=str,
    default=None,
    help="DAM stackfile (default: bundled jetbot_lane_safety.yaml)",
)
parser.add_argument(
    "--summary-path",
    type=str,
    default=None,
    help="Optional Markdown summary path.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    parser.error("Twin-lane Jetbot demo currently supports --num_envs 1 only.")
if args_cli.steps <= 0:
    parser.error("--steps must be positive.")

from tools.livestream.livestream_support import apply_livestream_defaults

apply_livestream_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from controll_scripts.safety import JetbotDAMWrapper
from controll_scripts.utils import physx_to_torch


JETBOT_TRACK_WIDTH = 0.12
RAW_LANE_Y = 0.62
DAM_LANE_Y = -0.62
START_X = -0.55
OBSTACLE_X = 1.32
LANE_LENGTH = 3.2
LANE_WIDTH = 0.82
CRASH_RADIUS = 0.26

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)


def _cuboid(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    collision: bool = False,
) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


def create_scene_cfg() -> type:
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3600.0, color=(0.85, 0.85, 0.85)),
        )

        raw_lane = AssetBaseCfg(
            prim_path="/World/RawLane/Floor",
            spawn=_cuboid((LANE_LENGTH, LANE_WIDTH, 0.015), (0.45, 0.06, 0.04)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.72, RAW_LANE_Y, 0.007)),
        )
        dam_lane = AssetBaseCfg(
            prim_path="/World/DamLane/Floor",
            spawn=_cuboid((LANE_LENGTH, LANE_WIDTH, 0.015), (0.03, 0.34, 0.12)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.72, DAM_LANE_Y, 0.007)),
        )
        divider = AssetBaseCfg(
            prim_path="/World/Divider",
            spawn=_cuboid((LANE_LENGTH, 0.045, 0.055), (0.95, 0.95, 0.95)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.72, 0.0, 0.027)),
        )

        raw_start = AssetBaseCfg(
            prim_path="/World/RawLane/StartLine",
            spawn=_cuboid((0.035, LANE_WIDTH, 0.045), (0.9, 0.9, 0.9)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(START_X, RAW_LANE_Y, 0.03)),
        )
        dam_start = AssetBaseCfg(
            prim_path="/World/DamLane/StartLine",
            spawn=_cuboid((0.035, LANE_WIDTH, 0.045), (0.9, 0.9, 0.9)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(START_X, DAM_LANE_Y, 0.03)),
        )
        raw_danger = AssetBaseCfg(
            prim_path="/World/RawLane/DangerZone",
            spawn=_cuboid((0.34, LANE_WIDTH, 0.035), (1.0, 0.0, 0.0)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(OBSTACLE_X, RAW_LANE_Y, 0.026)),
        )
        dam_danger = AssetBaseCfg(
            prim_path="/World/DamLane/DangerZone",
            spawn=_cuboid((0.34, LANE_WIDTH, 0.035), (1.0, 0.0, 0.0)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(OBSTACLE_X, DAM_LANE_Y, 0.026)),
        )
        raw_gate = AssetBaseCfg(
            prim_path="/World/RawLane/CrashGate",
            spawn=_cuboid((0.075, 0.62, 0.22), (1.0, 0.18, 0.13), collision=True),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(OBSTACLE_X + 0.1, RAW_LANE_Y, 0.11)),
        )
        dam_gate = AssetBaseCfg(
            prim_path="/World/DamLane/CrashGate",
            spawn=_cuboid((0.075, 0.62, 0.22), (1.0, 0.18, 0.13), collision=True),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(OBSTACLE_X + 0.1, DAM_LANE_Y, 0.11)),
        )
        raw_label = AssetBaseCfg(
            prim_path="/World/Labels/RawDirect",
            spawn=_cuboid((0.55, 0.09, 0.08), (1.0, 0.08, 0.05)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.3, RAW_LANE_Y, 0.12)),
        )
        dam_label = AssetBaseCfg(
            prim_path="/World/Labels/DamGuarded",
            spawn=_cuboid((0.55, 0.09, 0.08), (0.05, 0.95, 0.25)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.3, DAM_LANE_Y, 0.12)),
        )
        dam_intervention_beacon = AssetBaseCfg(
            prim_path="/World/DamLane/InterventionBeacon",
            spawn=_cuboid((0.2, 0.2, 0.2), (1.0, 0.82, 0.0)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(OBSTACLE_X - 0.28, DAM_LANE_Y, 0.1)),
        )

        raw_jetbot = JETBOT_CONFIG.replace(prim_path="/World/RawLane/Jetbot")
        dam_jetbot = JETBOT_CONFIG.replace(prim_path="/World/DamLane/Jetbot")

    return SceneCfg


@dataclass
class TwinLaneMetrics:
    steps: int = 0
    raw_crash_frames: int = 0
    dam_crash_frames: int = 0
    raw_min_gate_distance: float = float("inf")
    dam_min_gate_distance: float = float("inf")
    max_raw_wheel_speed: float = 0.0
    max_dam_wheel_speed: float = 0.0
    max_guard_delta: float = 0.0
    decisions: Counter = field(default_factory=Counter)

    @property
    def interventions(self) -> int:
        return sum(self.decisions[name] for name in ("CLAMP", "REJECT", "FAULT"))


def _scripted_policy_wheels(step: int, total_steps: int, device: str) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    if progress < 0.12:
        speed = args_cli.unsafe_speed * progress / 0.12
    elif progress < 0.72:
        speed = args_cli.unsafe_speed
    else:
        speed = args_cli.unsafe_speed * max(0.0, 1.0 - (progress - 0.72) / 0.28)

    weave = math.sin(progress * math.pi * 3.0)
    late_snap = 1.0 if 0.42 < progress < 0.58 else 0.0
    omega = args_cli.turn_command * (0.55 * weave + 0.75 * late_snap)
    left = speed - omega * JETBOT_TRACK_WIDTH / 2.0
    right = speed + omega * JETBOT_TRACK_WIDTH / 2.0
    return torch.tensor([[left, right]], dtype=torch.float32, device=device)


def _decision_color(decision: str) -> str:
    colors = {
        "RAW": "\033[91m",
        "PASS": "\033[92m",
        "CLAMP": "\033[93m",
        "REJECT": "\033[91m",
        "FAULT": "\033[95m",
    }
    return f"{colors.get(decision, '\033[90m')}{decision:<6}\033[0m"


class TwinLaneDAMDemo:
    def __init__(self) -> None:
        self.sim = None
        self.scene = None
        self.raw_jetbot = None
        self.dam_jetbot = None
        self.dam_guard: JetbotDAMWrapper | None = None
        self.sim_dt = 0.0
        self.metrics = TwinLaneMetrics()
        self.stackfile = args_cli.stackfile or "jetbot_lane_safety.yaml"

    def setup(self) -> bool:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([2.45, -2.15, 1.8], [0.72, 0.0, 0.05])
        self.sim_dt = self.sim.get_physics_dt()

        scene_cfg = create_scene_cfg()(num_envs=1, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.raw_jetbot = self.scene["raw_jetbot"]
        self.dam_jetbot = self.scene["dam_jetbot"]
        self.dam_guard = JetbotDAMWrapper(self.stackfile, device=self.sim.device)
        self._reset_robots()

        print("\n" + "=" * 78)
        print("  Twin-Lane Jetbot DAM Intervention Demo")
        print(f"  Stackfile:     {self.stackfile}")
        print(f"  Raw speed:     {args_cli.unsafe_speed:.2f}")
        print(f"  Turn command:  {args_cli.turn_command:.2f}")
        print("  RAW lane: action goes directly to Isaac.")
        print("  DAM lane: same action must pass through robot-dam SafetyGuard.")
        print("=" * 78 + "\n")
        return True

    def run(self) -> None:
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                return
            raw_action = _scripted_policy_wheels(step, args_cli.steps, self.sim.device)
            obs = self._dam_wheel_obs()
            dam_action = self.dam_guard.filter(raw_action, obs)

            self.raw_jetbot.set_joint_velocity_target_index(target=raw_action)
            self.dam_jetbot.set_joint_velocity_target_index(target=dam_action)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

            self._record_step(raw_action, dam_action)
            if step % args_cli.log_every == 0 or step == args_cli.steps - 1:
                self._print_step(step, raw_action, dam_action)

        summary = self._format_summary()
        print(summary)
        if args_cli.summary_path:
            path = Path(args_cli.summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(summary, encoding="utf-8")
            print(f"[INFO] Wrote car summary: {path}")

    def close(self) -> None:
        if self.dam_guard is not None:
            self.dam_guard.close()

    def _reset_robots(self) -> None:
        self._reset_robot(self.raw_jetbot, RAW_LANE_Y)
        self._reset_robot(self.dam_jetbot, DAM_LANE_Y)
        self.scene.reset()
        self.raw_jetbot.update(dt=self.sim_dt)
        self.dam_jetbot.update(dt=self.sim_dt)

    def _reset_robot(self, robot, lane_y: float) -> None:
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, 0] = START_X
        root_pose[:, 1] = lane_y
        root_pose[:, 2] = 0.05
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.default_root_vel.torch))
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_vel.torch))

    def _dam_wheel_obs(self) -> torch.Tensor:
        vel = physx_to_torch(self.dam_jetbot.data.joint_vel)
        return torch.as_tensor(vel, dtype=torch.float32, device=self.sim.device)[:, :2]

    def _record_step(self, raw_action: torch.Tensor, dam_action: torch.Tensor) -> None:
        raw_dist = self._gate_distance(self.raw_jetbot, RAW_LANE_Y)
        dam_dist = self._gate_distance(self.dam_jetbot, DAM_LANE_Y)
        decision = self.dam_guard.last_decision

        self.metrics.steps += 1
        self.metrics.raw_min_gate_distance = min(self.metrics.raw_min_gate_distance, raw_dist)
        self.metrics.dam_min_gate_distance = min(self.metrics.dam_min_gate_distance, dam_dist)
        self.metrics.max_raw_wheel_speed = max(self.metrics.max_raw_wheel_speed, torch.max(torch.abs(raw_action)).item())
        self.metrics.max_dam_wheel_speed = max(self.metrics.max_dam_wheel_speed, torch.max(torch.abs(dam_action)).item())
        self.metrics.max_guard_delta = max(self.metrics.max_guard_delta, self.dam_guard.last_delta)
        self.metrics.decisions[decision] += 1
        if raw_dist < CRASH_RADIUS:
            self.metrics.raw_crash_frames += 1
        if dam_dist < CRASH_RADIUS:
            self.metrics.dam_crash_frames += 1

    def _gate_distance(self, robot, lane_y: float) -> float:
        pos = physx_to_torch(robot.data.root_pos_w)[0, :2]
        gate = torch.tensor([OBSTACLE_X + 0.1, lane_y], dtype=pos.dtype, device=pos.device)
        return torch.norm(gate - pos).item()

    def _print_step(self, step: int, raw_action: torch.Tensor, dam_action: torch.Tensor) -> None:
        tag = _decision_color(self.dam_guard.last_decision)
        raw_l, raw_r = raw_action[0].tolist()
        dam_l, dam_r = dam_action[0].tolist()
        print(
            f"[DAM {tag}] step={step:04d} "
            f"raw=[{raw_l:+.2f},{raw_r:+.2f}] "
            f"dam=[{dam_l:+.2f},{dam_r:+.2f}] "
            f"delta={self.dam_guard.last_delta:.2f} "
            f"interventions={self.metrics.interventions}"
        )

    def _format_summary(self) -> str:
        decisions = " ".join(
            f"{name}={self.metrics.decisions[name]}"
            for name in ("PASS", "CLAMP", "REJECT", "FAULT")
        )
        return "\n".join(
            [
                "",
                "TWIN-LANE DAM DEMO SUMMARY",
                "Same scripted wheel command stream; RAW applies it directly, DAM applies SafetyGuard output.",
                f"Steps: {self.metrics.steps}",
                f"DAM decisions: {decisions}",
                f"DAM interventions: {self.metrics.interventions} ({self.dam_guard.intervention_rate:.1%})",
                f"Max raw wheel command: {self.metrics.max_raw_wheel_speed:.2f} rad/s",
                f"Max DAM wheel command: {self.metrics.max_dam_wheel_speed:.2f} rad/s",
                f"Max DAM correction: {self.metrics.max_guard_delta:.2f} rad/s",
                f"RAW closest gate distance: {self.metrics.raw_min_gate_distance:.3f}m",
                f"DAM closest gate distance: {self.metrics.dam_min_gate_distance:.3f}m",
                f"RAW crash frames: {self.metrics.raw_crash_frames}",
                f"DAM crash frames: {self.metrics.dam_crash_frames}",
                "",
            ]
        )


def main() -> None:
    demo = TwinLaneDAMDemo()
    try:
        if demo.setup():
            demo.run()
            print("[demo] replay complete; keeping the visualizer loop alive. Press Ctrl+C to exit.", flush=True)
            while demo.sim.is_headless_or_exist_active_visualizer():
                demo.sim.step()
    finally:
        demo.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
