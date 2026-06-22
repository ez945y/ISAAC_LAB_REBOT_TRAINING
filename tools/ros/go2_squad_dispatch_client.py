# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""CLI publisher for the squad ROS dispatch API.

This is a lightweight product-integration helper: web backends can mirror the
same JSON payload contract on /squad/dispatch_cmd.

Examples:
    python tools/ros/go2_squad_dispatch_client.py dispatch --x 5.0 --y 2.0 --group ALL --formation wedge
    python tools/ros/go2_squad_dispatch_client.py regroup --group-size 3
    python tools/ros/go2_squad_dispatch_client.py patrol --enabled true
    python tools/ros/go2_squad_dispatch_client.py halt
"""

from __future__ import annotations

import argparse
import json

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String


def _rclpy_init() -> None:
    """Init rclpy WITHOUT rclpy's own signal handlers. The interactive loop puts
    the terminal in raw mode (``_getch``); rclpy's default SIGINT handler in that
    state could shut the context down behind the loop's back -> 'publisher's context
    is invalid' on the next publish. We handle Ctrl+C ourselves instead."""
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)


def _parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send JSON dispatch commands to Go2 squad ROS API")
    parser.add_argument("--topic", default="/squad/dispatch_cmd", help="Command topic")
    parser.add_argument("--interactive", action="store_true", help="Start interactive keyboard mode")
    sub = parser.add_subparsers(dest="verb", required=False)

    p_dispatch = sub.add_parser("dispatch")
    p_dispatch.add_argument("--x", type=float, required=True)
    p_dispatch.add_argument("--y", type=float, required=True)
    p_dispatch.add_argument("--group", default="ALL")
    p_dispatch.add_argument("--formation", default=None, choices=["wedge", "row", "column"])
    p_dispatch.add_argument("--spacing", type=float, default=None)

    p_regroup = sub.add_parser("regroup")
    p_regroup.add_argument("--group-size", type=int, required=True)

    p_patrol = sub.add_parser("patrol")
    p_patrol.add_argument("--enabled", type=_parse_bool, required=True)

    sub.add_parser("recall")
    sub.add_parser("halt")
    return parser


def _payload_from_args(args: argparse.Namespace) -> dict:
    verb = args.verb
    if verb == "dispatch":
        payload = {
            "verb": "dispatch",
            "target": {"x": args.x, "y": args.y},
            "group": args.group,
        }
        if args.formation is not None:
            payload["formation"] = args.formation
        if args.spacing is not None:
            payload["spacing"] = args.spacing
        return payload
    if verb == "regroup":
        return {"verb": "regroup", "group_size": args.group_size}
    if verb == "patrol":
        return {"verb": "patrol", "enabled": bool(args.enabled)}
    return {"verb": verb}


class DispatchClient(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("go2_squad_dispatch_client")
        self._pub = self.create_publisher(String, topic, 10)

    def wait_for_subscriber(self, timeout_s: float = 8.0) -> bool:
        """Block until the sim's subscription is discovered, so a one-shot publish
        isn't dropped during DDS discovery (each CLI invocation is a fresh node, so
        it must re-discover every time). Returns False on timeout."""
        import time

        deadline = time.time() + timeout_s
        while self._pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._pub.get_subscription_count() > 0

    def publish(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.05)  # flush the publication
        self.get_logger().info(f"published: {msg.data}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    # Interactive keyboard mode
    if args.interactive:
        _interactive_mode(args.topic)
        return

    if args.verb is None:
        parser.error("give a subcommand (dispatch/regroup/patrol/recall/halt) or --interactive")
    payload = _payload_from_args(args)

    _rclpy_init()
    node = DispatchClient(args.topic)
    try:
        if not node.wait_for_subscriber():
            node.get_logger().warn(
                f"no subscriber on '{args.topic}' — is the sim running with --ros-control? "
                "Publishing anyway (may be dropped)."
            )
        node.publish(payload)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _getch() -> str:
    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _print_help() -> None:
    print("Interactive keyboard commands:")
    print("  d -> dispatch (prompts for x,y,group,formation)")
    print("  r -> regroup (prompts for group_size)")
    print("  p -> patrol (prompts enabled true/false)")
    print("  h -> halt")
    print("  c -> recall")
    print("  q -> quit")
    print("  ? -> show this help")


def _interactive_mode(topic: str) -> None:
    import sys

    _rclpy_init()
    node = DispatchClient(topic)
    print(f"Interactive client publishing to {topic}")
    if node.wait_for_subscriber():
        print("connected to the sim (subscriber found).")
    else:
        print("WARNING: no subscriber yet — is the sim running with --ros-control?")
    _print_help()
    try:
        while True:
            print("Press key (? for help):", end=" ", flush=True)
            ch = _getch()
            print(ch)
            payload = None
            if ch == "?":
                _print_help()
            elif ch == "q":
                print("quit")
                break
            elif ch == "d":
                try:
                    x = float(input("target x: "))
                    y = float(input("target y: "))
                except ValueError:
                    print("invalid coordinates")
                    continue
                group = input("group [ALL]: ") or "ALL"
                # Number/name picker: 1=wedge 2=row 3=column (enter = keep current).
                _FORMATIONS = {"1": "wedge", "2": "row", "3": "column",
                               "wedge": "wedge", "row": "row", "column": "column"}
                f_in = input("formation [1=wedge 2=row 3=column] (enter for none): ").strip().lower()
                formation = _FORMATIONS.get(f_in)
                if f_in and formation is None:
                    print(f"unknown formation '{f_in}', keeping current")
                spacing_txt = input("spacing m [enter=default 1.2]: ")
                payload = {"verb": "dispatch", "target": {"x": x, "y": y}, "group": group}
                if formation:
                    payload["formation"] = formation
                if spacing_txt.strip():
                    payload["spacing"] = float(spacing_txt)
            elif ch == "r":
                try:
                    payload = {"verb": "regroup", "group_size": int(input("group_size: "))}
                except ValueError:
                    print("invalid group_size")
                    continue
            elif ch == "p":
                try:
                    payload = {"verb": "patrol", "enabled": _parse_bool(input("enabled (true/false): "))}
                except argparse.ArgumentTypeError:
                    print("invalid bool")
                    continue
            elif ch == "h":
                payload = {"verb": "halt"}
            elif ch == "c":
                payload = {"verb": "recall"}
            else:
                print("unknown key, press ? for help")

            if payload is not None:
                node.publish(payload)
    except KeyboardInterrupt:
        print("\nquit (Ctrl+C)")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
