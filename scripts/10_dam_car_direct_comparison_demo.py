# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001  -- Isaac modules imported only after SimulationApp starts.

"""
Twin-lane Jetbot DAM boundary demo — DIRECT edition (NATIVE isaacsim).

Native-app sibling of ``09_dam_car_scripted_comparison_demo.py``. Same twin-lane
story (RAW vs DAM) and the SAME in-process safety path as 09 — the DAM car's
``[v, omega]`` command goes straight through ``JetbotDAMWrapper.filter()`` and is
applied with direct ``set_joint_velocities`` calls. No ROS 2, no OmniGraph bridge,
no external guard node.

    Isaac (this script)
    ───────────────────
    RAW car:  scripted target -> [v,w] -> wheels                 (applied directly)
    DAM car:  scripted target -> [v,w] -> Guardrail.filter() -> wheels  (applied directly)

This replaces the legacy ROS-bridge variant
(``10_dam_car_ros_comparison_demo_legacy.py``), which routed the DAM car over a
ROS2SubscribeTwist -> DifferentialController graph fed by a separate guard-node
process. The only reason that variant ran on the native app was the OmniGraph ROS
bridge (it fails to build under Isaac Lab's minimal kit); with the bridge gone the
direct path here is all that is needed.

WHY STILL NATIVE (not Isaac Lab AppLauncher): this demo keeps the native WebRTC
streaming path (tools/livestream.native_livestream_argv) so it stays a drop-in for
the demo-10 slot. Demo 09 is the Isaac Lab AppLauncher equivalent of the same story.

Scene parity with demo 09: the arena bands, divider, goal flags, optional walking
worker (--worker), cars and the stream camera (eye/target) use identical world
coordinates and the warehouse uses the same scale (0.25). The warehouse ORIENTATION quaternion intentionally differs:
demo 09's (w,x,y,z)=(0,0,-0.7071,0.7071) is a 180° flip the asset needs under Isaac
Lab; the natively-referenced asset is already upright, so we keep identity — applying
09's quat here would lay the warehouse on its side and remove the floor.

Run (one terminal — no ROS, no guard node):

    source ~/IsaacLab/env_isaaclab/bin/activate
    python scripts/10_dam_car_direct_comparison_demo.py --livestream 2
"""

import argparse
import math
import os
import sys
from collections import Counter

from isaacsim import SimulationApp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

parser = argparse.ArgumentParser(description="Twin-lane Jetbot RAW vs DAM demo (direct, native)")
parser.add_argument("--steps", type=int, default=960, help="Scripted replay length")
parser.add_argument("--target-scale", type=float, default=1.15, help="Raw target aggressiveness")
parser.add_argument("--drive-gain", type=float, default=4.2, help="Target follower drive gain")
parser.add_argument("--log-every", type=int, default=60)
parser.add_argument("--stackfile", type=str, default="examples/stackfiles/jetbot_lane_safety.yaml",
                    help="DAM stackfile (default resolves to the bundled jetbot_lane_safety.yaml).")
parser.add_argument("--livestream", type=int, default=0, help="1/2 = enable WebRTC livestream.")
parser.add_argument("--loop", action="store_true", default=True,
                    help="Loop the scripted run forever (default on, for streaming).")
parser.add_argument("--no-loop", dest="loop", action="store_false",
                    help="Run --steps once then hold instead of looping.")
parser.add_argument("--cam-eye", type=str, default="3.0, -1.5, 1.2",
                    help="Stream camera eye 'x,y,z' (default matches demo 09).")
parser.add_argument("--cam-target", type=str, default="-1.8, 1.3, 0",
                    help="Stream camera look-at target 'x,y,z' (default matches demo 09).")
parser.add_argument("--worker", action="store_true",
                    help="Spawn a walking construction worker in each lane (like demo 09); the "
                    "RAW car drives into it during the opening surge while the DAM car is clamped short.")
parser.add_argument("--worker-scale", type=float, default=0.2,
                    help="Worker size (1.0 = ~1.8m human; 0.2 ~ 0.36m).")
parser.add_argument("--worker-enter-step", type=int, default=40,
                    help="Phase offset (steps) shifting where the worker is in its back-and-forth cycle.")
parser.add_argument("--worker-yaw", type=float, default=90.0,
                    help="Extra yaw (deg) to correct the worker model's facing (try 90/180/-90).")
args_cli = parser.parse_args()
if args_cli.steps <= 0:
    parser.error("--steps must be positive.")
_CAM_EYE = [float(v) for v in args_cli.cam_eye.split(",")]
_CAM_TARGET = [float(v) for v in args_cli.cam_target.split(",")]

# Streaming: native WebRTC, the same proven settings as 09 — injected as Kit args
# BEFORE SimulationApp so they take effect at launch (see tools/livestream).
if args_cli.livestream >= 1:
    from livestream.livestream_support import native_livestream_argv

    sys.argv += native_livestream_argv()

simulation_app = SimulationApp({"headless": True})

# ── Enable just the native core API (World + articulation). No graph / ROS bridge
# extensions — the direct path needs none of them. ──
import omni.kit.app  # noqa: E402
from omni.ext import ExtensionPathType  # noqa: E402

import isaacsim  # noqa: E402

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_ext_mgr.add_path(
    os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"),
    ExtensionPathType.COLLECTION,
)
_ext_mgr.set_extension_enabled_immediate("isaacsim.core.api", True)
simulation_app.update()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.core.api.robots.robot import Robot  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from controll_scripts.safety import AckermannSolver, JetbotDAMWrapper  # noqa: E402


JETBOT_TRACK_WIDTH = 0.12
JETBOT_WHEEL_RADIUS = 0.03
RAW_LANE_Y = 0.72
DAM_LANE_Y = -0.72
LANE_CENTER_X = 0.42
ARENA_WIDTH = 1.20
BOTTOM_BAND_X = -0.51
SAFE_X_MIN = -0.35
SAFE_X_MAX = 1.40
SAFE_Y_LIMIT = 0.48
ARENA_WIDTH_HALF = ARENA_WIDTH / 2.0
SIDE_BAND_WIDTH = 0.12
START_X = 0.18
STAND_Z = 0.05

DAM_JETBOT_PRIM = "/World/DamArena/Jetbot"
RAW_JETBOT_PRIM = "/World/RawArena/Jetbot"
WHEEL_JOINTS = ["left_wheel_joint", "right_wheel_joint"]

# Guard math runs on CPU — the per-step tensors are tiny and this keeps the
# wrapper independent of the render device.
DAM_DEVICE = "cpu"

RED = np.array([0.92, 0.04, 0.04])
GREEN = np.array([0.03, 0.45, 0.16])
WHITE = np.array([0.86, 0.86, 0.86])
GREY = np.array([0.7, 0.7, 0.7])
FLAG_RED = np.array([1.0, 0.08, 0.05])
FLAG_GREEN = np.array([0.08, 0.95, 0.22])

# Worker walk paths (world XY) — identical layout to demo 09: each worker walks
# ALONG the outer red forbidden band (constant y = lane ± 0.54, sweeping x), so
# only the un-guarded RAW car drives into it during the opening surge.
RED_BAND_Y = ARENA_WIDTH_HALF - SIDE_BAND_WIDTH / 2.0  # 0.54
RAW_WORKER_START = (0.10, RAW_LANE_Y + RED_BAND_Y)
RAW_WORKER_END = (1.30, RAW_LANE_Y + RED_BAND_Y)
DAM_WORKER_START = (0.10, DAM_LANE_Y + RED_BAND_Y)
DAM_WORKER_END = (1.30, DAM_LANE_Y + RED_BAND_Y)


def _raw_target(step: int, total_steps: int, lane_y: float) -> tuple[float, float]:
    """Scripted 2D target — identical to demo 09.

    ``step`` wraps modulo ``total_steps`` so the trajectory loops forever, keeping
    the stream alive instead of freezing."""
    step = step % max(total_steps, 1)
    progress = step / max(total_steps - 1, 1)
    loop = progress * math.pi * 2.0
    x = 0.42 + args_cli.target_scale * (0.64 * math.sin(loop * 0.86) - 0.35 * math.sin(loop * 1.7))
    local_y = args_cli.target_scale * (0.36 * math.sin(loop * 1.23))
    if progress < 0.24:
        surge = math.sin(progress / 0.24 * math.pi)
        local_y += args_cli.target_scale * 0.40 * surge
        x += args_cli.target_scale * 0.34 * surge
    if 0.58 < progress < 0.78:
        x -= args_cli.target_scale * 0.55
    return x, lane_y + local_y


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _decision_color(decision: str) -> str:
    colors = {"PASS": "\033[92m", "CLAMP": "\033[93m", "REJECT": "\033[91m", "FAULT": "\033[95m"}
    reset = "\033[0m"
    color = colors.get(decision, "\033[90m")
    return f"{color}{decision:<6}{reset}"


class TwinLaneDAMDirectDemo:
    def __init__(self) -> None:
        self.world: World | None = None
        self.raw_robot: Robot | None = None
        self.dam_robot: Robot | None = None
        self.dam_guard: JetbotDAMWrapper | None = None
        self.raw_wheel_idx: list[int] = []
        self.dam_wheel_idx: list[int] = []
        self.steps_done = 0
        self.raw_forbidden = 0
        self.dam_forbidden = 0
        self.raw_min_margin = float("inf")
        self.dam_min_margin = float("inf")
        self.max_delta = 0.0
        self.decisions: Counter = Counter()
        self.workers: list = []

    def setup(self) -> bool:
        self.world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        assets_root = get_assets_root_path()
        if assets_root is None:
            print("[ERROR] Isaac assets root not found", flush=True)
            return False

        add_reference_to_stage(
            assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd", "/World/Warehouse"
        )
        # Match demo 09's warehouse SIZE (scale 0.25). The orientation quaternion
        # differs on purpose — see the module docstring: identity here = upright on
        # the native app, same look as 09's flipped quat under Isaac Lab.
        self._set_xform("/World/Warehouse", scale=0.25, quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        jetbot_usd = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
        add_reference_to_stage(jetbot_usd, RAW_JETBOT_PRIM)
        add_reference_to_stage(jetbot_usd, DAM_JETBOT_PRIM)

        self._build_boundaries()
        if args_cli.worker:
            self._spawn_workers(assets_root)

        self.raw_robot = self.world.scene.add(Robot(prim_path=RAW_JETBOT_PRIM, name="raw_jetbot"))
        self.dam_robot = self.world.scene.add(Robot(prim_path=DAM_JETBOT_PRIM, name="dam_jetbot"))

        # In-process DAM guard — the SAME safety layer demo 09 uses.
        self.dam_guard = JetbotDAMWrapper(
            args_cli.stackfile,
            device=DAM_DEVICE,
            solver=AckermannSolver(
                track_width=JETBOT_TRACK_WIDTH, wheel_radius=JETBOT_WHEEL_RADIUS,
                max_v=2.2, max_omega=6.0,
            ),
        )

        self.world.reset()
        self.raw_robot.initialize()
        self.dam_robot.initialize()
        self.raw_wheel_idx = [self.raw_robot.get_dof_index(n) for n in WHEEL_JOINTS]
        self.dam_wheel_idx = [self.dam_robot.get_dof_index(n) for n in WHEEL_JOINTS]

        self._place(self.raw_robot, RAW_LANE_Y)
        self._place(self.dam_robot, DAM_LANE_Y)
        self._set_stream_camera()

        print("\n" + "=" * 78)
        print("  Twin-Lane Jetbot DAM Boundary Demo — DIRECT (native isaacsim)")
        print(f"  Stackfile:  {args_cli.stackfile}")
        print("  RAW lane:   applies nominal [v, omega] directly")
        print("  DAM lane:   applies Guardrail-validated [v, omega] directly (in-process)")
        print("=" * 78 + "\n", flush=True)
        return True

    def _build_boundaries(self) -> None:
        def band(path, pos, scale, color):
            VisualCuboid(prim_path=path, name=path.split("/")[-1].lower(),
                         position=np.array(pos), scale=np.array(scale), color=color, size=1.0)

        sx = SAFE_X_MAX - SAFE_X_MIN
        cx = (SAFE_X_MIN + SAFE_X_MAX) / 2.0
        for lane, tag in ((RAW_LANE_Y, "Raw"), (DAM_LANE_Y, "Dam")):
            band(f"/World/{tag}Arena/SafeRegion", (cx, lane, 0.0001), (sx, SAFE_Y_LIMIT * 2.0, 0.001), GREEN)
            band(f"/World/{tag}Arena/LeftForbidden",
                 (cx, lane + ARENA_WIDTH / 2.0 - SIDE_BAND_WIDTH / 2.0, 0.0002),
                 (sx, SIDE_BAND_WIDTH, 0.001), RED)
            band(f"/World/{tag}Arena/RightForbidden",
                 (cx, lane - ARENA_WIDTH / 2.0 + SIDE_BAND_WIDTH / 2.0, 0.0002),
                 (sx, SIDE_BAND_WIDTH, 0.001), RED)
            band(f"/World/{tag}Arena/BottomForbidden", (BOTTOM_BAND_X, lane, 0.0002),
                 (0.32, ARENA_WIDTH, 0.001), RED)
        band("/World/Divider", (LANE_CENTER_X, 0.0, 0.0225), (3.0, 0.045, 0.045), WHITE)

        # Goal flags, matching demo 09: a grey pole + a coloured flag in each lane
        # (red for RAW, green for DAM) marking the far end of the working area.
        for lane, tag, flag_color, sign in (
            (RAW_LANE_Y, "Raw", FLAG_RED, -1.0),
            (DAM_LANE_Y, "Dam", FLAG_GREEN, 1.0),
        ):
            fy = lane + sign * 0.5
            band(f"/World/{tag}Arena/Flagpole", (1.55, fy, 0.20), (0.02, 0.02, 0.40), GREY)
            band(f"/World/{tag}Arena/Flag", (1.485, fy, 0.36), (0.15, 0.01, 0.08), flag_color)

    def _spawn_workers(self, assets_root: str) -> None:
        """Spawn a walking worker in each lane (demo 09 parity, native edition).

        Each worker prim is a plain parent Xform that the SlidingWorker drives
        (translate/orient in world space); the actual character USD is referenced
        as a CHILD carrying only the uniform scale, so scaling never distorts the
        world-space slide path. Import is lazy + guarded: a missing People asset or
        Isaac Lab import must not abort the demo."""
        try:
            import omni.usd
            from pxr import Gf, UsdGeom

            from controll_scripts.scene import SlidingWorker, WorkerPath
        except Exception as exc:  # noqa: BLE001 -- worker is optional eye-candy
            print(f"[demo10] --worker unavailable, skipping ({exc})", flush=True)
            return

        worker_usd = (
            assets_root + "/Isaac/People/Characters/"
            "original_male_adult_construction_05/male_adult_construction_05.usd"
        )
        stage = omni.usd.get_context().get_stage()
        sc = args_cli.worker_scale
        common = dict(
            enter_step=args_cli.worker_enter_step,
            walk_steps=max(args_cli.steps // 3, 200),
            yaw_offset_deg=args_cli.worker_yaw,
            loop=True,
        )
        specs = [
            ("/World/RawArena/Worker", RAW_WORKER_START, RAW_WORKER_END),
            ("/World/DamArena/Worker", DAM_WORKER_START, DAM_WORKER_END),
        ]
        for prim_path, start, end in specs:
            # Three levels so scale never pollutes the world-space slide:
            #   <prim_path>        Xform — SlidingWorker drives translate/orient (world)
            #   <prim_path>/Scale  Xform — carries ONLY the uniform scale
            #   <prim_path>/Scale/Model  referenced character USD
            # The character USD root already ships its own xformOps (translate/scale/
            # rotate), so we must NOT AddScaleOp on it (the op already exists); a fresh
            # intermediate Xform gives us a clean scale op instead.
            UsdGeom.Xform.Define(stage, prim_path)
            scale_path = prim_path + "/Scale"
            UsdGeom.Xform.Define(stage, scale_path)
            UsdGeom.Xformable(stage.GetPrimAtPath(scale_path)).AddScaleOp().Set(
                Gf.Vec3d(sc, sc, sc)
            )
            add_reference_to_stage(worker_usd, scale_path + "/Model")
            worker = SlidingWorker(prim_path, WorkerPath(start_xy=start, end_xy=end, **common))
            if worker.build():
                self.workers.append(worker)
        print(f"[demo10] sliding workers active: {len(self.workers)}", flush=True)

    def _place(self, robot: Robot, lane_y: float) -> None:
        robot.set_world_pose(position=np.array([START_X, lane_y, STAND_Z]))
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))

    @staticmethod
    def _set_stream_camera() -> None:
        """Frame the streamed viewport like demo 09 (same eye/target). Without an
        explicit camera the WebRTC stream shows Kit's default pose."""
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=_CAM_EYE, target=_CAM_TARGET)
        except Exception as exc:  # noqa: BLE001 -- camera is cosmetic, never fatal
            print(f"[demo10] could not set stream camera ({exc})", flush=True)

    @staticmethod
    def _set_xform(prim_path: str, *, scale: float, quat_wxyz: tuple) -> None:
        """Set a uniform scale + orientation on a referenced prim (USD authoring)."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
        xf.ClearXformOpOrder()
        xf.AddOrientOp().Set(Gf.Quatf(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
        xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    def run(self) -> None:
        step = 0
        # Loop the scripted trajectory forever (default) so the stream never freezes;
        # --no-loop runs --steps once then prints the summary and holds.
        while simulation_app.is_running():
            for worker in self.workers:
                worker.update(step)

            # RAW car: nominal target follower applied directly (the contrast).
            tx, ty = _raw_target(step, args_cli.steps, RAW_LANE_Y)
            left, right = self._wheels_from_command(*self._target_to_command(self.raw_robot, tx, ty))
            self.raw_robot.set_joint_velocities(np.array([left, right]), joint_indices=self.raw_wheel_idx)

            # DAM car: same nominal command, but filtered through the Guardrail and
            # converted to wheels by the guard's own solver — the exact path demo 09
            # uses, so the action source is identical end-to-end.
            dtx, dty = _raw_target(step, args_cli.steps, DAM_LANE_Y)
            v, omega = self._target_to_command(self.dam_robot, dtx, dty)
            dam_wheels = self.dam_guard.command_to_wheels(self._dam_filter(self.dam_robot, v, omega))
            dleft, dright = float(dam_wheels[0, 0]), float(dam_wheels[0, 1])
            self.dam_robot.set_joint_velocities(np.array([dleft, dright]), joint_indices=self.dam_wheel_idx)

            self.world.step(render=True)

            self._record(step)
            if step % args_cli.log_every == 0:
                self._print(step)

            step += 1
            if not args_cli.loop and step >= args_cli.steps:
                print(self._summary(), flush=True)
                break

        # --no-loop: hold the app alive after the single run so you can keep watching.
        while not args_cli.loop and simulation_app.is_running():
            self.world.step(render=True)

    def _target_to_command(self, robot: Robot, tx: float, ty: float) -> tuple[float, float]:
        """Nominal [v, omega] target follower (matches demo 09's gains/clamps)."""
        pos, quat = robot.get_world_pose()
        x, y = float(pos[0]), float(pos[1])
        yaw = _yaw_from_quat(quat)
        dx, dy = tx - x, ty - y
        desired = math.atan2(dy, dx)
        heading_err = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
        distance = math.hypot(dx, dy)
        v = max(-2.2, min(2.2, args_cli.drive_gain * distance * math.cos(heading_err)))
        omega = max(-5.0, min(5.0, 5.0 * heading_err))
        return v, omega

    def _dam_filter(self, robot: Robot, v: float, omega: float) -> "torch.Tensor":
        """Run [v, omega] through the in-process DAM guard; return the safe (1, 2)
        [v, omega] tensor (fed straight into the guard's command_to_wheels, like 09).

        State is the lane-local pose [x, y_local, yaw] (y measured from DAM_LANE_Y),
        exactly as demo 09 feeds it — the stackfile boundaries are authored in the
        lane frame."""
        pos, quat = robot.get_world_pose()
        state = torch.tensor(
            [[float(pos[0]), float(pos[1]) - DAM_LANE_Y, _yaw_from_quat(quat)]],
            dtype=torch.float32, device=DAM_DEVICE,
        )
        command = torch.tensor([[v, omega]], dtype=torch.float32, device=DAM_DEVICE)
        safe = self.dam_guard.filter(command, state)
        self.decisions[self.dam_guard.last_decision] += 1
        self.max_delta = max(self.max_delta, self.dam_guard.last_delta)
        return safe

    @staticmethod
    def _wheels_from_command(v: float, omega: float) -> tuple[float, float]:
        left = (v - omega * JETBOT_TRACK_WIDTH / 2.0) / JETBOT_WHEEL_RADIUS
        right = (v + omega * JETBOT_TRACK_WIDTH / 2.0) / JETBOT_WHEEL_RADIUS
        return left, right

    def _record(self, step: int) -> None:
        rp, _ = self.raw_robot.get_world_pose()
        dp, _ = self.dam_robot.get_world_pose()
        raw_margin = self._margin(float(rp[0]), float(rp[1]) - RAW_LANE_Y)
        dam_margin = self._margin(float(dp[0]), float(dp[1]) - DAM_LANE_Y)
        self.steps_done += 1
        self.raw_min_margin = min(self.raw_min_margin, raw_margin)
        self.dam_min_margin = min(self.dam_min_margin, dam_margin)
        if raw_margin < 0.0:
            self.raw_forbidden += 1
        if dam_margin < 0.0:
            self.dam_forbidden += 1

    @staticmethod
    def _margin(x: float, local_y: float) -> float:
        return min(x - SAFE_X_MIN, SAFE_X_MAX - x, SAFE_Y_LIMIT - abs(local_y))

    @property
    def interventions(self) -> int:
        return sum(self.decisions[name] for name in ("CLAMP", "REJECT", "FAULT"))

    def _print(self, step: int) -> None:
        rp, _ = self.raw_robot.get_world_pose()
        dp, _ = self.dam_robot.get_world_pose()
        tag = _decision_color(self.dam_guard.last_decision)
        print(
            f"[step {step:04d}] [DAM {tag}] "
            f"RAW pos=({rp[0]:+.2f},{rp[1]:+.2f}) min_margin={self.raw_min_margin:+.3f} | "
            f"DAM pos=({dp[0]:+.2f},{dp[1]:+.2f}) min_margin={self.dam_min_margin:+.3f} "
            f"interventions={self.interventions}",
            flush=True,
        )

    def _summary(self) -> str:
        decisions = " ".join(
            f"{name}={self.decisions[name]}" for name in ("PASS", "CLAMP", "REJECT", "FAULT")
        )
        return "\n".join([
            "",
            "TWIN-LANE DAM BOUNDARY SUMMARY (direct, native)",
            "Same target stream; RAW applies nominal [v, omega], DAM applies Guardrail-validated [v, omega].",
            f"Steps: {self.steps_done}",
            f"DAM decisions: {decisions}",
            f"DAM interventions: {self.interventions} ({self.dam_guard.intervention_rate:.1%})",
            f"Max DAM command correction: {self.max_delta:.2f}",
            f"RAW forbidden-zone frames: {self.raw_forbidden}",
            f"DAM forbidden-zone frames: {self.dam_forbidden}",
            f"RAW min safe margin: {self.raw_min_margin:.3f}m",
            f"DAM min safe margin: {self.dam_min_margin:.3f}m",
            "",
        ])

    def close(self) -> None:
        if self.dam_guard is not None:
            self.dam_guard.close()


def main() -> None:
    demo = TwinLaneDAMDirectDemo()
    try:
        if demo.setup():
            # run() loops the scripted trajectory forever (default) and only returns
            # when the app is closing, so nothing "ends" mid-stream. Ctrl+C to exit.
            demo.run()
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
