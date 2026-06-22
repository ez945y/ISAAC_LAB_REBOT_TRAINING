# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""ROS transport adapter for squad dispatch/control.

This module defines a compact JSON command contract for product integrations
(e.g. web scheduler -> ROS -> simulator/robots) and a thin ROS 2 bridge that
translates ROS String messages into normalized command dictionaries.

Design goals:
- Keep scheduler/mission logic transport-agnostic
- Avoid custom ROS msg build steps for fast iteration
- Provide deterministic validation/normalization for incoming commands
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RosTopics:
    command_topic: str = "/squad/dispatch_cmd"
    status_topic: str = "/squad/status"
    action_topic: str = "/squad/action_cmd"
    marker_topic: str = "/squad/markers"


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Yaw (rad about +z) -> (x, y, z, w) quaternion."""
    import math

    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def _expect_number(obj: dict[str, Any], key: str) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def normalize_dispatch_command(payload: str | dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one dispatch command.

    Contract (JSON object):
      - verb: dispatch | regroup | patrol | recall | halt | quit | help
      - dispatch fields:
          target: {"x": float, "y": float}
          group: optional str, "ALL" by default
          formation: optional "wedge"|"row"|"column"
          spacing: optional float
      - regroup fields:
          group_size: int (1..64)
      - patrol fields:
          enabled: bool
    """
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    verb = str(data.get("verb", "")).strip().lower()
    if not verb:
        raise ValueError("missing verb")

    out: dict[str, Any] = {"verb": verb}
    if verb == "dispatch":
        target = data.get("target")
        if not isinstance(target, dict):
            raise ValueError("dispatch.target must be an object with x/y")
        out["target"] = {
            "x": _expect_number(target, "x"),
            "y": _expect_number(target, "y"),
        }
        out["group"] = str(data.get("group", "ALL")).strip() or "ALL"
        if "formation" in data and data["formation"] is not None:
            out["formation"] = str(data["formation"]).strip().lower()
        if "spacing" in data and data["spacing"] is not None:
            spacing = float(data["spacing"])
            if spacing <= 0.0:
                raise ValueError("spacing must be > 0")
            out["spacing"] = spacing
    elif verb == "regroup":
        size = int(data.get("group_size", 0))
        if size <= 0 or size > 64:
            raise ValueError("group_size must be in [1, 64]")
        out["group_size"] = size
    elif verb == "patrol":
        if "enabled" not in data:
            raise ValueError("patrol.enabled is required")
        out["enabled"] = bool(data["enabled"])
    elif verb in {"recall", "halt", "quit", "help"}:
        pass
    else:
        raise ValueError(f"unsupported verb: {verb}")

    return out


class SquadRosBridge:
    """ROS 2 String-topic bridge for squad command ingress and telemetry egress."""

    def __init__(self, *, node_name: str, topics: RosTopics) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from std_msgs.msg import String
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "rclpy/std_msgs unavailable. Source ROS 2 environment before using --ros-control."
            ) from exc

        self._rclpy = rclpy
        self._String = String
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self._spin_thread: threading.Thread | None = None

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node: Node = Node(node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)

        self._pub_status = self.node.create_publisher(String, topics.status_topic, 10)
        self._pub_action = self.node.create_publisher(String, topics.action_topic, 50)
        self._sub_cmd = self.node.create_subscription(String, topics.command_topic, self._on_cmd, 50)

        # Optional rviz visualization (visualization_msgs is a desktop dep; degrade
        # gracefully if it isn't there so the core command/status path still works).
        self._Marker = None
        self._MarkerArray = None
        self._pub_markers = None
        try:
            from visualization_msgs.msg import Marker, MarkerArray

            self._Marker = Marker
            self._MarkerArray = MarkerArray
            self._pub_markers = self.node.create_publisher(MarkerArray, topics.marker_topic, 10)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warning(f"markers disabled (no visualization_msgs): {exc}")

    def start(self) -> None:
        if self._spin_thread is not None:
            return

        def _spin() -> None:
            while not self._stop.is_set() and self._rclpy.ok():
                self._executor.spin_once(timeout_sec=0.05)

        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()

    def _on_cmd(self, msg) -> None:
        try:
            cmd = normalize_dispatch_command(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warning(f"invalid command payload: {exc}")
            return
        self._queue.put(cmd)

    def poll_commands(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    def publish_status(self, payload: dict[str, Any]) -> None:
        msg = self._String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._pub_status.publish(msg)

    def publish_action(self, payload: dict[str, Any]) -> None:
        msg = self._String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._pub_action.publish(msg)

    def publish_markers(self, items: list[dict[str, Any]], *, frame: str = "map") -> None:
        """Publish rviz markers from plain dicts (keeps script ROS-msg-agnostic).

        Each item: ns, id, shape ("arrow"|"cylinder"|"sphere"|"text"), x, y, z, yaw,
        sx/sy/sz scale, r/g/b/a color, optional text. Coordinates are in ``frame``
        (the dog world == the /map frame, so they line up on the occupancy grid)."""
        if self._pub_markers is None:
            return
        _SHAPE = {
            "arrow": self._Marker.ARROW,
            "cylinder": self._Marker.CYLINDER,
            "sphere": self._Marker.SPHERE,
            "text": self._Marker.TEXT_VIEW_FACING,
        }
        stamp = self.node.get_clock().now().to_msg()
        arr = self._MarkerArray()
        for it in items:
            m = self._Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = str(it.get("ns", "squad"))
            m.id = int(it["id"])
            m.type = _SHAPE.get(it.get("shape", "sphere"), self._Marker.SPHERE)
            m.action = self._Marker.ADD
            m.pose.position.x = float(it.get("x", 0.0))
            m.pose.position.y = float(it.get("y", 0.0))
            m.pose.position.z = float(it.get("z", 0.0))
            qx, qy, qz, qw = _yaw_to_quat(float(it.get("yaw", 0.0)))
            m.pose.orientation.x, m.pose.orientation.y = qx, qy
            m.pose.orientation.z, m.pose.orientation.w = qz, qw
            m.scale.x = float(it.get("sx", 0.3))
            m.scale.y = float(it.get("sy", 0.3))
            m.scale.z = float(it.get("sz", 0.3))
            m.color.r = float(it.get("r", 1.0))
            m.color.g = float(it.get("g", 1.0))
            m.color.b = float(it.get("b", 1.0))
            m.color.a = float(it.get("a", 1.0))
            if "text" in it:
                m.text = str(it["text"])
            arr.markers.append(m)
        self._pub_markers.publish(arr)

    def close(self) -> None:
        self._stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
        try:
            self._executor.remove_node(self.node)
        except Exception:  # noqa: BLE001
            pass
        self.node.destroy_node()
        # Do not globally shutdown rclpy if another node owns the context.
        time.sleep(0.01)
