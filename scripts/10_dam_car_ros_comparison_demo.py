# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001  -- Isaac modules imported only after SimulationApp starts.

"""
Twin-lane Jetbot DAM boundary demo — ROS 2 bridge edition (NATIVE isaacsim).

ROS version of ``09_dam_car_scripted_comparison_demo.py``. Same twin-lane story
(RAW vs DAM), but the DAM car is driven over ROS 2 through Isaac's OmniGraph ROS 2
bridge — never by direct ``set_joint_*`` calls:

    Isaac (this script)                         DAM guard node (separate process)
    ───────────────────                         ─────────────────────────────────
    publish DAM car pose  ──/dam_jetbot/odom──▶  follow scripted target -> [v,w]
                                                 JetbotDAMWrapper.filter() (guard)
    drive DAM car wheels ◀──/dam_jetbot/cmd_vel── publish safe [v,w] (Twist)

The DAM car graph is ROS2SubscribeTwist -> DifferentialController ->
ArticulationController; its pose goes out via ROS2PublishOdometry. The RAW car is
driven directly in-process (the contrast).

WHY NATIVE (not Isaac Lab AppLauncher): OmniGraph graph creation fails under Isaac
Lab's minimal kit experience ("Unable to create prim for graph"); the ROS 2 bridge
graph only builds under the native isaacsim app. Streaming is therefore done with
the native WebRTC path (tools/livestream.native_livestream_argv) — the SAME proven
settings 09 uses, just injected at native launch instead of via AppLauncher.

Run (two terminals, both ROS 2 sourced):

    # terminal 1 — simulator
    source ~/IsaacLab/env_isaaclab/bin/activate
    source /opt/ros/jazzy/setup.bash
    export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
    python scripts/10_dam_car_ros_comparison_demo.py --livestream 2

    # terminal 2 — DAM guardrail node
    source ~/IsaacLab/env_isaaclab/bin/activate
    source /opt/ros/jazzy/setup.bash
    python tools/ros/dam_jetbot_guard_node.py
"""

import argparse
import math
import os
import sys

from isaacsim import SimulationApp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

parser = argparse.ArgumentParser(description="Twin-lane Jetbot RAW vs DAM demo (ROS 2 bridge, native)")
parser.add_argument("--steps", type=int, default=960, help="Scripted replay length")
parser.add_argument("--target-scale", type=float, default=1.15, help="Raw target aggressiveness")
parser.add_argument("--drive-gain", type=float, default=4.2, help="RAW target follower drive gain")
parser.add_argument("--log-every", type=int, default=60)
parser.add_argument("--cmd-topic", type=str, default="/dam_jetbot/cmd_vel",
                    help="Twist topic the DAM car is driven from (subscribed via the bridge).")
parser.add_argument("--odom-topic", type=str, default="/dam_jetbot/odom",
                    help="Odometry topic the DAM car pose is published on (for the guard node).")
parser.add_argument("--livestream", type=int, default=0, help="1/2 = enable WebRTC livestream.")
parser.add_argument("--loop", action="store_true", default=True,
                    help="Loop the scripted run forever (default on, for streaming).")
parser.add_argument("--no-loop", dest="loop", action="store_false",
                    help="Run --steps once then hold instead of looping.")
parser.add_argument("--cam-eye", type=str, default="3.0, -1.5, 1.2",
                    help="Stream camera eye 'x,y,z'.")
parser.add_argument("--cam-target", type=str, default="-1.8, 1.3, 0",
                    help="Stream camera look-at target 'x,y,z'.")
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

# ── Enable the native extensions we need (graph nodes + ROS 2 bridge + the
# deprecated core.api World). Same bootstrap as scripts/smoke/_go2_smoke.py. ──
import omni.kit.app  # noqa: E402
from omni.ext import ExtensionPathType  # noqa: E402

import isaacsim  # noqa: E402

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_ext_mgr.add_path(
    os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"),
    ExtensionPathType.COLLECTION,
)
for _ext in (
    "isaacsim.core.api",
    "isaacsim.robot.wheeled_robots",
    "omni.graph.action",
    "omni.graph.nodes",
    "isaacsim.core.nodes",
    "isaacsim.robot.wheeled_robots.nodes",
    "isaacsim.ros2.bridge",
):
    _ext_mgr.set_extension_enabled_immediate(_ext, True)
simulation_app.update()

import numpy as np  # noqa: E402
import omni.graph.core as og  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.core.api.robots.robot import Robot  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402


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
SIDE_BAND_WIDTH = 0.12
START_X = 0.18
STAND_Z = 0.05

DAM_JETBOT_PRIM = "/World/DamArena/Jetbot"
RAW_JETBOT_PRIM = "/World/RawArena/Jetbot"
WHEEL_JOINTS = ["left_wheel_joint", "right_wheel_joint"]

RED = np.array([0.92, 0.04, 0.04])
GREEN = np.array([0.03, 0.45, 0.16])
WHITE = np.array([0.86, 0.86, 0.86])


def _raw_target(step: int, total_steps: int, lane_y: float) -> tuple[float, float]:
    """Scripted 2D target — identical to 09 (and mirrored in the guard node).

    ``step`` wraps modulo ``total_steps`` so the trajectory loops forever (the
    guard node mirrors this), keeping the stream alive instead of freezing."""
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


class TwinLaneDAMRosDemo:
    def __init__(self) -> None:
        self.world: World | None = None
        self.raw_robot: Robot | None = None
        self.dam_robot: Robot | None = None
        self.raw_wheel_idx: list[int] = []
        self.steps_done = 0
        self.raw_forbidden = 0
        self.dam_forbidden = 0
        self.raw_min_margin = float("inf")
        self.dam_min_margin = float("inf")

    def setup(self) -> bool:
        self.world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        assets_root = get_assets_root_path()
        if assets_root is None:
            print("[ERROR] Isaac assets root not found", flush=True)
            return False

        add_reference_to_stage(
            assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd", "/World/Warehouse"
        )
        # Match demo 09's warehouse SIZE (scale 0.25). NOTE: demo 09's rotation
        # (w,x,y,z)=(0,0,-0.7071,0.7071) is a 180° flip needed under Isaac Lab; the
        # natively-referenced asset is already upright, so applying it here lays the
        # warehouse on its side (maps +z up -> -y) and removes the floor -> the cars
        # fall and jam. Scale only.
        self._set_xform("/World/Warehouse", scale=0.25, quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        jetbot_usd = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
        add_reference_to_stage(jetbot_usd, RAW_JETBOT_PRIM)
        add_reference_to_stage(jetbot_usd, DAM_JETBOT_PRIM)

        self._build_boundaries()

        self.raw_robot = self.world.scene.add(Robot(prim_path=RAW_JETBOT_PRIM, name="raw_jetbot"))
        self.dam_robot = self.world.scene.add(Robot(prim_path=DAM_JETBOT_PRIM, name="dam_jetbot"))

        # Build the ROS bridge graph that drives the DAM car (validated to work on
        # the native app). Done before reset, while the timeline is stopped.
        self._build_ros_graph()

        self.world.reset()
        self.raw_robot.initialize()
        self.dam_robot.initialize()
        self.raw_wheel_idx = [self.raw_robot.get_dof_index(n) for n in WHEEL_JOINTS]

        self._place(self.raw_robot, RAW_LANE_Y)
        self._place(self.dam_robot, DAM_LANE_Y)
        self._set_stream_camera()

        print("\n" + "=" * 78)
        print("  Twin-Lane Jetbot DAM Boundary Demo — ROS 2 bridge (native isaacsim)")
        print("  RAW lane:  applies nominal [v, omega] directly (in-process)")
        print(f"  DAM lane:  driven over ROS 2 bridge from '{args_cli.cmd_topic}'")
        print(f"  DAM pose:  published on '{args_cli.odom_topic}'")
        print("  Start the guard node: python tools/ros/dam_jetbot_guard_node.py")
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

    def _build_ros_graph(self) -> None:
        keys = og.Controller.Keys
        try:
            og.Controller.edit(
                {"graph_path": "/DamRosControlGraph", "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnPlaybackTick"),
                        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("ReadTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                        ("BreakLin", "omni.graph.nodes.BreakVector3"),
                        ("BreakAng", "omni.graph.nodes.BreakVector3"),
                        ("DiffCtl", "isaacsim.robot.wheeled_robots.DifferentialController"),
                        ("ArtCtl", "isaacsim.core.nodes.IsaacArticulationController"),
                        ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                        ("PubOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                        ("PubClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "SubTwist.inputs:execIn"),
                        ("SubTwist.outputs:execOut", "DiffCtl.inputs:execIn"),
                        ("SubTwist.outputs:linearVelocity", "BreakLin.inputs:tuple"),
                        ("SubTwist.outputs:angularVelocity", "BreakAng.inputs:tuple"),
                        ("BreakLin.outputs:x", "DiffCtl.inputs:linearVelocity"),
                        ("BreakAng.outputs:z", "DiffCtl.inputs:angularVelocity"),
                        ("OnTick.outputs:tick", "ArtCtl.inputs:execIn"),
                        ("DiffCtl.outputs:velocityCommand", "ArtCtl.inputs:velocityCommand"),
                        ("OnTick.outputs:tick", "ComputeOdom.inputs:execIn"),
                        ("ComputeOdom.outputs:execOut", "PubOdom.inputs:execIn"),
                        ("ComputeOdom.outputs:position", "PubOdom.inputs:position"),
                        ("ComputeOdom.outputs:orientation", "PubOdom.inputs:orientation"),
                        ("ComputeOdom.outputs:linearVelocity", "PubOdom.inputs:linearVelocity"),
                        ("ComputeOdom.outputs:angularVelocity", "PubOdom.inputs:angularVelocity"),
                        ("ReadTime.outputs:simulationTime", "PubOdom.inputs:timeStamp"),
                        ("OnTick.outputs:tick", "PubClock.inputs:execIn"),
                        ("ReadTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
                        ("Context.outputs:context", "SubTwist.inputs:context"),
                        ("Context.outputs:context", "PubOdom.inputs:context"),
                        ("Context.outputs:context", "PubClock.inputs:context"),
                    ],
                    keys.SET_VALUES: [
                        ("SubTwist.inputs:topicName", args_cli.cmd_topic),
                        ("DiffCtl.inputs:wheelRadius", JETBOT_WHEEL_RADIUS),
                        ("DiffCtl.inputs:wheelDistance", JETBOT_TRACK_WIDTH),
                        ("DiffCtl.inputs:maxLinearSpeed", 2.2),
                        ("DiffCtl.inputs:maxAngularSpeed", 6.0),
                        ("ArtCtl.inputs:robotPath", DAM_JETBOT_PRIM),
                        ("ArtCtl.inputs:jointNames", WHEEL_JOINTS),
                        ("ComputeOdom.inputs:chassisPrim", [DAM_JETBOT_PRIM]),
                        ("PubOdom.inputs:topicName", args_cli.odom_topic),
                        ("PubOdom.inputs:odomFrameId", "odom"),
                        ("PubOdom.inputs:chassisFrameId", "dam_jetbot"),
                        ("PubClock.inputs:topicName", "/clock"),
                    ],
                },
            )
            print("[ros] DAM control graph built (/DamRosControlGraph).", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[ros][ERROR] failed to build control graph: {exc}", flush=True)
            raise

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

            # Configurable via --cam-eye / --cam-target (the warehouse is scaled to
            # 0.25 so it is small; tune from the CLI instead of editing code).
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
        dam_start = self.dam_robot.get_world_pose()[0][:2].copy()
        warned = False
        step = 0
        # Loop the scripted trajectory forever (default) so the stream never freezes;
        # --no-loop runs --steps once then prints the summary and holds.
        while simulation_app.is_running():
            # RAW car: nominal target follower applied directly (the contrast).
            tx, ty = _raw_target(step, args_cli.steps, RAW_LANE_Y)
            left, right = self._target_to_wheels(self.raw_robot, tx, ty)
            self.raw_robot.set_joint_velocities(np.array([left, right]), joint_indices=self.raw_wheel_idx)

            # DAM car: driven entirely by the ROS bridge graph (cmd_vel from the
            # guard node). We only step the sim.
            self.world.step(render=True)

            self._record(step)
            if step % args_cli.log_every == 0:
                self._print(step)

            step += 1
            # One-shot guard-not-running hint, checked once the trajectory has had
            # time to move the DAM car (it only moves on guard cmd_vel).
            if not warned and step == 200:
                dam_now = self.dam_robot.get_world_pose()[0][:2]
                if float(np.linalg.norm(dam_now - dam_start)) < 0.05:
                    print(
                        f"[WARN] DAM car not moving — no commands on '{args_cli.cmd_topic}'. "
                        "Is the guard node running? (python tools/ros/dam_jetbot_guard_node.py)",
                        flush=True,
                    )
                warned = True

            if not args_cli.loop and step >= args_cli.steps:
                print(self._summary(), flush=True)
                break

        # --no-loop: hold the app alive after the single run so you can keep watching.
        while not args_cli.loop and simulation_app.is_running():
            self.world.step(render=True)

    def _target_to_wheels(self, robot: Robot, tx: float, ty: float) -> tuple[float, float]:
        pos, quat = robot.get_world_pose()
        x, y = float(pos[0]), float(pos[1])
        yaw = _yaw_from_quat(quat)
        dx, dy = tx - x, ty - y
        desired = math.atan2(dy, dx)
        heading_err = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
        distance = math.hypot(dx, dy)
        v = max(-2.2, min(2.2, args_cli.drive_gain * distance * math.cos(heading_err)))
        omega = max(-5.0, min(5.0, 5.0 * heading_err))
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

    def _print(self, step: int) -> None:
        rp, _ = self.raw_robot.get_world_pose()
        dp, _ = self.dam_robot.get_world_pose()
        print(
            f"[step {step:04d}] RAW pos=({rp[0]:+.2f},{rp[1]:+.2f}) min_margin={self.raw_min_margin:+.3f} | "
            f"DAM pos=({dp[0]:+.2f},{dp[1]:+.2f}) min_margin={self.dam_min_margin:+.3f}",
            flush=True,
        )

    def _summary(self) -> str:
        return "\n".join([
            "",
            "TWIN-LANE DAM BOUNDARY SUMMARY (ROS 2 bridge, native)",
            "RAW applies nominal [v,omega] directly; DAM is driven over ROS from the guard node.",
            f"Steps: {self.steps_done}",
            f"RAW forbidden-zone frames: {self.raw_forbidden}",
            f"DAM forbidden-zone frames: {self.dam_forbidden}",
            f"RAW min safe margin: {self.raw_min_margin:.3f}m",
            f"DAM min safe margin: {self.dam_min_margin:.3f}m",
            "(DAM PASS/CLAMP/REJECT decisions are logged by the guard node.)",
            "",
        ])


def main() -> None:
    demo = TwinLaneDAMRosDemo()
    if demo.setup():
        # run() loops the scripted trajectory forever (default) and only returns
        # when the app is closing, so nothing "ends" mid-stream. Ctrl+C to exit.
        demo.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
