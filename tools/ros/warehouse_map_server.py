# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal static-map server: publish a baked ROS occupancy grid on /map.

nav2_map_server isn't installed on this box, and the squad codebase deliberately
keeps its ROS surface tiny (see tools/controll_scripts/squad/ros_interface.py), so
this is a ~1-file stand-in: read the warehouse.pgm/.yaml that scripts/_warehouse_
occupancy.py bakes, build one nav_msgs/OccupancyGrid, and latch it on /map with
transient_local QoS so late subscribers (rviz, the web console bridge, every dog)
all receive the same shared map. The map is the environment — one publisher, all
robots consume it.

Run (source ROS first):
    python tools/ros/warehouse_map_server.py [--map tools/ros/maps/warehouse.yaml]
                                             [--topic /map] [--frame map]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def _parse_yaml(path: str) -> dict:
    """Tiny flat-YAML reader for the map_server fields we write (no pyyaml dep)."""
    out: dict[str, object] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, val = (s.strip() for s in line.split(":", 1))
            if val.startswith("[") and val.endswith("]"):
                out[key] = [float(x) for x in val[1:-1].split(",") if x.strip()]
            else:
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    return out


def _read_pgm(path: str) -> np.ndarray:
    """Read an 8-bit binary (P5) PGM into a (h, w) uint8 array, row 0 = top."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5", "only binary P5 PGM supported"
        # width height (may be on one or two lines); then maxval
        dims: list[int] = []
        while len(dims) < 2:
            dims += [int(t) for t in f.readline().split()]
        f.readline()  # maxval
        w, h = dims[0], dims[1]
        return np.frombuffer(f.read(w * h), np.uint8).reshape(h, w)


def _to_occupancy(grid: np.ndarray, negate: int, occ_th: float, free_th: float) -> np.ndarray:
    """Map pixels -> ROS occupancy [0..100], -1 unknown, then flip to ROS row order.

    ROS convention: p_occ = (255 - p)/255 when negate==0 (else p/255). data[0] is
    the (min_x, min_y) bottom-left cell, rows increasing upward — so flip the
    top-first PGM vertically.
    """
    p = grid.astype(np.float32)
    occ = p / 255.0 if negate else (255.0 - p) / 255.0
    out = np.full(grid.shape, -1, dtype=np.int8)        # unknown
    out[occ > occ_th] = 100                              # occupied
    out[occ < free_th] = 0                               # free
    return np.flipud(out)                                # top-first PGM -> bottom-first ROS


class MapServer(Node):
    def __init__(self, msg: OccupancyGrid, topic: str, period: float) -> None:
        super().__init__("warehouse_map_server")
        # Latched map: transient_local so anyone who subscribes later still gets it.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(OccupancyGrid, topic, qos)
        self._msg = msg
        self._publish()
        # Re-stamp + re-publish occasionally (belt-and-suspenders alongside latching).
        self.create_timer(period, self._publish)
        self.get_logger().info(
            f"latched {msg.info.width}x{msg.info.height} @ {msg.info.resolution}m map on {topic}"
        )

    def _publish(self) -> None:
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._msg)


def build_msg(yaml_path: str, frame: str) -> OccupancyGrid:
    meta = _parse_yaml(yaml_path)
    pgm_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), str(meta["image"]))
    grid = _read_pgm(pgm_path)
    data = _to_occupancy(
        grid,
        int(meta.get("negate", 0)),
        float(meta.get("occupied_thresh", 0.65)),
        float(meta.get("free_thresh", 0.196)),
    )

    msg = OccupancyGrid()
    msg.header.frame_id = frame
    msg.info.resolution = float(meta["resolution"])
    msg.info.height, msg.info.width = grid.shape
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    pose = Pose()
    pose.position.x, pose.position.y = float(origin[0]), float(origin[1])
    pose.orientation.w = 1.0  # yaw assumed 0 (our baker writes origin yaw 0)
    msg.info.origin = pose
    msg.data = data.flatten().tolist()
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a baked occupancy grid on /map")
    parser.add_argument("--map", default="tools/ros/maps/warehouse.yaml", help="map yaml path")
    parser.add_argument("--topic", default="/map")
    parser.add_argument("--frame", default="map")
    parser.add_argument("--period", type=float, default=2.0, help="re-publish period (s)")
    args = parser.parse_args()

    msg = build_msg(args.map, args.frame)
    rclpy.init()
    node = MapServer(msg, args.topic, args.period)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
