# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""DAM Jetbot guardrail — ROS 2 node (companion to scripts/10_dam_car_ros_comparison_demo_legacy.py).

This is the "brain" of the ROS version of demo 09: it runs OUTSIDE the simulator
and talks to it only over ROS 2, so the sim never calls ``filter()`` directly.

    subscribe  /dam_jetbot/odom  (nav_msgs/Odometry)   -- the DAM car's pose
        |  follow the same scripted target demo 09 uses  -> nominal [v, omega]
        |  JetbotDAMWrapper.filter(nominal, state)        -> safe   [v, omega]
        v
    publish    /dam_jetbot/cmd_vel (geometry_msgs/Twist) -- the guarded command

The sim's OmniGraph bridge subscribes that Twist and drives the DAM car
(ROS2SubscribeTwist -> DifferentialController -> ArticulationController).

Run (ROS 2 sourced; robot-dam on PYTHONPATH via the Isaac env):
    source ~/IsaacLab/env_isaaclab/bin/activate
    source /opt/ros/jazzy/setup.bash
    python tools/ros/dam_jetbot_guard_node.py

Constants below MIRROR the sim (scripts/10_dam_car_ros_comparison_demo_legacy.py) so the
DAM car follows the same target stream as the RAW car.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

SIM_DT = 1.0 / 60.0  # matches the sim's World physics_dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controll_scripts.safety import AckermannSolver, JetbotDAMWrapper  # noqa: E402

# ── Constants mirrored from the sim demo ────────────────────────────────────
DAM_LANE_Y = -0.72
START_X = 0.18
JETBOT_TRACK_WIDTH = 0.12
JETBOT_WHEEL_RADIUS = 0.03
_DEFAULT_STACKFILE = os.path.join(
    os.path.expanduser("~/DAM"), "examples", "stackfiles", "jetbot_lane_safety.yaml"
)


def _scripted_target(step: int, total_steps: int, lane_y: float, target_scale: float) -> tuple[float, float]:
    """Scripted 2D target — identical to the sim's ``_raw_target`` (loops modulo
    total_steps so it matches the sim's looping trajectory)."""
    step = step % max(total_steps, 1)
    progress = step / max(total_steps - 1, 1)
    loop = progress * math.pi * 2.0
    x = 0.42 + target_scale * (0.64 * math.sin(loop * 0.86) - 0.35 * math.sin(loop * 1.7))
    local_y = target_scale * (0.36 * math.sin(loop * 1.23))
    if progress < 0.24:
        surge = math.sin(progress / 0.24 * math.pi)
        local_y += target_scale * 0.40 * surge
        x += target_scale * 0.34 * surge
    if 0.58 < progress < 0.78:
        x -= target_scale * 0.55
    return x, lane_y + local_y


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class DamJetbotGuardNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dam_jetbot_guard")
        self._args = args
        self._device = "cpu"
        self._step = 0
        # Latest reconstructed pose: (local_x, local_y, yaw). None until first odom.
        self._pose: tuple[float, float, float] | None = None
        self._logged_first_odom = False
        self._sim_time: float | None = None  # from /clock; keeps DAM on the same
        #                                       scripted timeline as the RAW car.

        self.guard = JetbotDAMWrapper(
            args.stackfile,
            device=self._device,
            solver=AckermannSolver(
                track_width=JETBOT_TRACK_WIDTH, wheel_radius=JETBOT_WHEEL_RADIUS,
                max_v=2.2, max_omega=6.0,
            ),
        )

        self._pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self._sub = self.create_subscription(Odometry, args.odom_topic, self._on_odom, 10)
        self._clock_sub = self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self._timer = self.create_timer(1.0 / args.rate, self._on_tick)
        self.get_logger().info(
            f"DAM guard up: odom='{args.odom_topic}' -> cmd='{args.cmd_topic}' "
            f"@ {args.rate:.0f} Hz, stackfile='{args.stackfile}'"
        )

    def _on_odom(self, msg: Odometry) -> None:
        # IsaacComputeOdometry reports the chassis pose in the world frame (its
        # reference is the world origin, where the prim was referenced before being
        # placed). So odom (x, y) IS the world pose; the DAM lane is centred at
        # DAM_LANE_Y, so sim-local coords are (world_x, world_y - DAM_LANE_Y).
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        local_x = p.x
        local_y = p.y - DAM_LANE_Y
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        self._pose = (local_x, local_y, yaw)
        if not self._logged_first_odom:
            self.get_logger().info(
                f"first odom: world=({p.x:+.3f},{p.y:+.3f}) -> local=({local_x:+.3f},{local_y:+.3f})"
            )
            self._logged_first_odom = True

    def _on_clock(self, msg: Clock) -> None:
        self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_tick(self) -> None:
        if self._pose is None:
            return  # wait for the first odometry sample
        local_x, local_y, yaw = self._pose

        # Drive the scripted target off SIM time (/clock) so the DAM car follows the
        # exact same timeline as the RAW car regardless of ROS/sim rate or startup
        # delay; fall back to the local tick count until the first /clock arrives.
        step = int(self._sim_time / SIM_DT) if self._sim_time is not None else self._step

        # World pose of the DAM car (lane_y offset back in).
        world_x, world_y = local_x, local_y + DAM_LANE_Y
        tx, ty = _scripted_target(step, self._args.steps, DAM_LANE_Y, self._args.target_scale)

        # Nominal target follower (mirrors the sim's _target_to_command).
        dx, dy = tx - world_x, ty - world_y
        desired_heading = math.atan2(dy, dx)
        heading_error = math.atan2(math.sin(desired_heading - yaw), math.cos(desired_heading - yaw))
        distance = math.hypot(dx, dy)
        # Cap the DAM car's approach speed BELOW the RAW car's (2.2): over ROS the
        # safe command returns a few frames late, so a slower approach means a much
        # smaller overshoot past the boundary -> the guard can hold/redirect it
        # (ride the edge, turn) instead of overshooting and reversing (front/back judder).
        forward = max(-self._args.max_v, min(self._args.max_v,
                                              self._args.drive_gain * distance * math.cos(heading_error)))
        omega = max(-5.0, min(5.0, 5.0 * heading_error))

        nominal = torch.tensor([[forward, omega]], dtype=torch.float32, device=self._device)
        state = torch.tensor([[local_x, local_y, yaw]], dtype=torch.float32, device=self._device)
        safe = self.guard.filter(nominal, state)
        safe_v, safe_omega = float(safe[0, 0]), float(safe[0, 1])

        twist = Twist()
        twist.linear.x = safe_v
        twist.angular.z = safe_omega
        self._pub.publish(twist)

        if self._step % self._args.log_every == 0:
            self.get_logger().info(
                f"[{self.guard.last_decision:<6}] sim_step={step:04d} "
                f"nominal=({forward:+.2f},{omega:+.2f}) safe=({safe_v:+.2f},{safe_omega:+.2f}) "
                f"pose=({world_x:+.2f},{world_y:+.2f})"
            )
        self._step += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="DAM Jetbot guardrail ROS 2 node")
    parser.add_argument("--odom-topic", type=str, default="/dam_jetbot/odom")
    parser.add_argument("--cmd-topic", type=str, default="/dam_jetbot/cmd_vel")
    parser.add_argument("--steps", type=int, default=960, help="Scripted replay length (match the sim).")
    parser.add_argument("--target-scale", type=float, default=1.15, help="Match the sim's --target-scale.")
    parser.add_argument("--drive-gain", type=float, default=4.2, help="Match the sim's --drive-gain.")
    parser.add_argument("--max-v", type=float, default=1.2,
                        help="DAM approach speed cap (m/s); keep below the RAW car's 2.2 so the "
                             "ROS-delayed safe command doesn't overshoot the boundary.")
    parser.add_argument("--rate", type=float, default=60.0, help="Control rate (Hz).")
    parser.add_argument("--log-every", type=int, default=60)
    parser.add_argument("--stackfile", type=str, default=_DEFAULT_STACKFILE)
    args = parser.parse_args()

    rclpy.init()
    node = DamJetbotGuardNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.guard.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
