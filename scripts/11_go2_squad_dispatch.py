# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001 -- Isaac modules imported only after SimulationApp starts.

"""Unitree Go2 squad dispatch in a warehouse (Isaac Sim 6.0).

Six Go2 quadrupeds you DRIVE live from the keyboard: dispatch the squad in
formation to named warehouse zones, hot-swap the formation shape, regroup on the
fly (1x6 / 2x3 / 3x2), or toggle an auto-patrol — all while watching the WebRTC
stream. Pass ``--auto`` for the original hands-off scripted 2x3 -> 3x2 run.

Architecture (see tools/controll_scripts/squad/):
    formation/scheduler/mission/robot_agent  — pure-logic, Isaac-free, tested in
        tests/test_squad_dispatch.py
    teleop.KeyboardConsole                    — operator interface: keys -> ABI
        intents (Isaac-free; reads the terminal, watch the stream)
    locomotion.Go2Locomotion                  — the only Go2-bound piece (RL policy)

The keyboard console turns keystrokes into squad-level Commands; the runtime
applies them to the Dispatcher/Squad ABI. The scheduler/missions never see Isaac,
the Go2, or the keyboard — swapping embodiment or input is a leaf change.

Boots the NATIVE isaacsim SimulationApp (not Isaac Lab's AppLauncher, whose
PhysxManager breaks the native Go2 policy). Headless by default; pass --livestream 2
for the WebRTC stream and drive from the launching terminal.

Run:  python scripts/11_go2_squad_dispatch.py [--livestream 2] [--auto] [--max-seconds N]
      (or scripts/demo11_go2_squad_ros.sh — sim + interactive dispatch client)
"""

import argparse
import math
import os
import sys

from isaacsim import SimulationApp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

parser = argparse.ArgumentParser(description="Go2 warehouse squad dispatch demo")
parser.add_argument("--max-seconds", type=float, default=600.0, help="Wall budget for the run.")
parser.add_argument("--arrive-tol", type=float, default=0.6, help="Slot arrival tolerance (m).")
parser.add_argument("--livestream", type=int, default=0, help="1/2 = enable WebRTC livestream.")
parser.add_argument("--auto", action="store_true", help="Run the scripted 2x3->3x2 demo (no keyboard).")
parser.add_argument("--capture", default="", help="Save framed PNG(s) then exit (smoke test).")
parser.add_argument("--capture-secs", default="", help="Comma sim-times to snapshot, e.g. 0,6,15,30 (with --auto).")
parser.add_argument("--ground", action="store_true", help="Use the default ground plane instead of the warehouse (gait A/B test).")
parser.add_argument("--ros-control", action="store_true", help="Enable ROS command/status/action interface.")
parser.add_argument("--ros-node-name", type=str, default="go2_squad_demo", help="ROS node name for squad interface.")
parser.add_argument("--ros-cmd-topic", type=str, default="/squad/dispatch_cmd", help="ROS command topic (std_msgs/String JSON).")
parser.add_argument("--ros-status-topic", type=str, default="/squad/status", help="ROS status topic (std_msgs/String JSON).")
parser.add_argument("--ros-action-topic", type=str, default="/squad/action_cmd", help="ROS action topic (std_msgs/String JSON).")
args_cli = parser.parse_args()

# WebRTC livestream: the Go2 RL policy needs the native SimulationApp, which cannot
# use Isaac Lab's AppLauncher (its PhysxManager breaks the native World), so the
# AppLauncher-based apply_livestream_defaults does not apply. Instead inject the
# SAME proven streaming kit args into argv BEFORE SimulationApp — it forwards them
# to Kit at launch (no-window/publicIp/resize must be set at launch, not after).
_launch_config: dict = {"headless": True}
if args_cli.livestream >= 1:
    from livestream.livestream_support import native_livestream_argv  # noqa: E402

    sys.argv += native_livestream_argv()
    # Keep the UI visible under headless so the dispatch console panel composites
    # into the WebRTC stream (the documented SimulationApp `hide_ui` override "when
    # live streaming"). We then enable a MINIMAL set of editor-UI extensions on the
    # lean base experience below — NOT the FULL streaming experience, which on this
    # aarch64/GB10 box drags in omni.anim.behavior.core and crashes (bad_variant_access)
    # during boot. Native app + native World throughout, so the Go2 policy is unaffected.
    _launch_config["hide_ui"] = False

# The Go2 RL policy is a NATIVE-isaacsim component and does NOT run under Isaac
# Lab's AppLauncher (it installs its own PhysxManager as SimulationManager, whose
# missing set_backend even breaks native World construction). Boot the native
# SimulationApp directly — same harness as scripts/smoke/_go2_smoke.py.
simulation_app = SimulationApp(_launch_config)

# ── Enable the native extensions the base app does not auto-load ──
# `isaacsim.core.api` World lives in extsDeprecated; the policy needs
# `isaacsim.robot.policy.examples`. (See scripts/smoke/_go2_smoke.py — same bootstrap.)
import omni.kit.app  # noqa: E402
from omni.ext import ExtensionPathType  # noqa: E402

import isaacsim  # noqa: E402

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
# Register BOTH collection paths up front — the in-repo dispatch console path must
# be known before any enable/solve, else enabling other exts first leaves the
# registry unaware of it ("No versions ... satisfies").
_ext_mgr.add_path(
    os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"),
    ExtensionPathType.COLLECTION,
)
_ext_mgr.add_path(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "isaac_ext"),
    ExtensionPathType.COLLECTION,
)
# In-repo dispatch console extension: UI panel + viewport overlay (shows in the
# WebRTC stream — no rviz needed). Lives under tools/isaac_ext/.
_boot_exts = ["isaacsim.core.api", "isaacsim.robot.policy.examples", "devicenexus.squad.dispatch"]
if args_cli.livestream >= 1:
    # Minimal editor UI so omni.ui windows (the dispatch console panel) composite
    # into the streamed frame — the lean base experience loads none of these. We
    # hand-pick the few we need instead of a full experience to avoid the
    # omni.anim.behavior crash (see the SimulationApp comment above).
    _boot_exts += ["omni.kit.uiapp", "omni.kit.mainwindow"]
for _ext in _boot_exts:
    try:
        _ext_mgr.set_extension_enabled_immediate(_ext, True)
    except Exception as _exc:  # noqa: BLE001 -- console is cosmetic, never fatal
        print(f"[demo] could not enable {_ext}: {_exc}", flush=True)
simulation_app.update()

# ── Isaac imports after the app is up ───────────────────────────────────────
import carb  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from controll_scripts.squad import (  # noqa: E402
    Command, Dispatcher, Formation, KeyboardConsole,
    RobotAgent, RosTopics, Squad, SquadRosBridge, help_text, split_evenly,
)
from controll_scripts.squad.locomotion import Go2Locomotion  # noqa: E402

PHYSICS_DT = 0.005          # 200 Hz, matches policy training
RENDER_EVERY = 4            # render at ~50 Hz
STAND_Z = 0.5

# 6 dogs start as a 2x3 block on the warehouse floor, just "west" of HOME.
START_XY = [
    (-1.5, 1.2), (-1.5, 0.0), (-1.5, -1.2),
    (-3.0, 1.2), (-3.0, 0.0), (-3.0, -1.2),
]
HOME = (0.0, 0.0)
# Named warehouse zones the operator dispatches to (kept inside the open lanes;
# the dogs have no obstacle avoidance, so keep targets off the shelving).
ZONES: list[tuple[str, tuple[float, float]]] = [
    ("DOCK-A", (5.0, 2.5)),
    ("DOCK-B", (5.0, -2.5)),
    ("AISLE-W", (-5.0, 0.0)),
    ("STAGE-N", (0.0, 3.5)),
    ("CHARGE-S", (0.0, -3.5)),
]
SHAPES = [Formation.WEDGE, Formation.ROW, Formation.COLUMN]
GROUP_SIZES = [6, 3, 2]      # 1x6 -> 2x3 -> 3x2 (group OF size)

# Distinct colors so you can track which group goes where in rviz.
GROUP_PALETTE = [
    (0.20, 0.60, 1.00), (1.00, 0.55, 0.10), (0.30, 0.90, 0.40),
    (0.90, 0.30, 0.90), (0.95, 0.85, 0.15), (0.40, 0.90, 0.90),
]


def _group_color(gid: str | None) -> tuple[float, float, float]:
    """Stable color for a group id (trailing int -> palette), white if ungrouped."""
    if not gid:
        return (0.85, 0.85, 0.85)
    digits = "".join(c for c in gid if c.isdigit())
    return GROUP_PALETTE[(int(digits) if digits else 0) % len(GROUP_PALETTE)]


def _apply_floor_friction(root_path: str) -> None:
    """Bind a friction=1.0 physics material to the floor (subtree of ``root_path``).

    The Go2 flat policy was TRAINED on a floor with static/dynamic friction 1.0
    (the shipped isaacsim Go2 example sets exactly this). PhysX's default ~0.5 lets
    the feet slip, so the policy can't stabilise and the dogs collapse into a
    belly-slide ("crawl"). Binding with the *physics* material purpose inherits
    down the namespace, so one bind on the env root covers every collision mesh.

    Strength MUST be ``strongerThanDescendants``: the Simple_Warehouse asset ships
    its own material bindings on the floor mesh, and ``weakerThanDescendants`` would
    let those win — so our friction=1.0 silently reverts to PhysX's slippery ~0.5
    on the warehouse floor and the dogs crawl. ``strongerThanDescendants`` forces
    1.0 everywhere; on the bare ground plane (no competing bindings) it is a no-op."""
    import omni.usd
    from pxr import Sdf, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, Sdf.Path("/World/PhysicsMaterials/Floor"))
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(1.0)
    pm.CreateDynamicFrictionAttr().Set(1.0)
    pm.CreateRestitutionAttr().Set(0.0)
    prim = stage.GetPrimAtPath(root_path)
    if prim.IsValid():
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                     materialPurpose="physics")


def _add_lighting() -> None:
    """Light the interior — the native World adds no light, and the warehouse
    asset is dark on its own, so without this the whole stream is black. A dome
    for ambient fill plus an angled distant key so the dogs cast shadows."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(1200.0)
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key.CreateIntensityAttr(1500.0)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 0.0))


def _set_stream_camera() -> None:
    """Point the streamed viewport at the warehouse floor so the squad is framed.

    Headless + --no-window means there is no GUI to orbit the view, so without an
    explicit camera the stream shows Kit's default pose. The Simple_Warehouse roof
    sits at z~9.3 m, so the eye MUST stay below it (an overhead view just renders
    the dark roof). This is an inside, eye-level look east down the open floor over
    the squad's 0->5 m working area (verified via scripts/_go2_warehouse_probe.py)."""
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=[-9.0, 0.0, 4.0], target=[2.0, 0.0, 0.4])
    except Exception as exc:  # noqa: BLE001 -- camera is cosmetic, never fatal
        carb.log_warn(f"could not set stream camera ({exc})")


class SquadController:
    """Operator/auto intents -> the ``Dispatcher.send`` ABI.

    Tracks each group's chosen formation and which group the operator is steering
    (``selected`` = ``None`` means the whole squad). Zone/formation commands act on
    the steered group only, so different groups can be sent to different areas with
    different formations. It is the single place that knows the warehouse geometry.
    """

    def __init__(self, squad: Squad, dispatcher: Dispatcher) -> None:
        self.squad = squad
        self.dispatcher = dispatcher
        self.formation: dict[str, Formation] = {}     # per-group formation
        self.area: dict[str, tuple[float, float]] = {}  # per-group last area (re-issue)
        self.selected: str | None = None              # steered group; None = whole squad
        self.size_i = 0
        self.patrol = False
        self.patrol_i = 0

    # -- which groups a command acts on ---------------------------------------

    def _steered(self) -> list[str]:
        """The group(s) the operator is currently steering."""
        if self.selected in self.squad.groups:
            return [self.selected]
        return list(self.squad.groups)

    def select_next(self) -> None:
        """Cycle the steered target: ALL -> G0 -> G1 -> ... -> ALL."""
        order: list[str | None] = [None, *self.squad.groups]
        i = order.index(self.selected) if self.selected in order else 0
        self.selected = order[(i + 1) % len(order)]
        print(f"[ctrl] steering {self.selected or 'ALL'}", flush=True)

    # -- sending --------------------------------------------------------------

    def _send(self, gid: str, area: tuple[float, float], formation: Formation | None = None) -> None:
        f = formation or self.formation.get(gid, Formation.WEDGE)
        self.formation[gid] = f
        self.area[gid] = area
        self.dispatcher.send(gid, area, f, spacing=1.4)

    def _fan(self, area: tuple[float, float], gids: list[str]) -> dict[str, tuple[float, float]]:
        """Spread ``gids`` across ``area`` (lateral to the approach) so several
        groups land side-by-side instead of piling on one point."""
        cx = sum(a.get_state().x for a in self.squad.agents.values()) / len(self.squad.agents)
        cy = sum(a.get_state().y for a in self.squad.agents.values()) / len(self.squad.agents)
        heading = math.atan2(area[1] - cy, area[0] - cx)
        lat = (math.cos(heading + math.pi / 2.0), math.sin(heading + math.pi / 2.0))
        gap = 3.0
        return {gid: (area[0] + lat[0] * (i - (len(gids) - 1) / 2.0) * gap,
                      area[1] + lat[1] * (i - (len(gids) - 1) / 2.0) * gap)
                for i, gid in enumerate(gids)}

    def dispatch(self, area: tuple[float, float]) -> None:
        """Send the steered target to ``area``. One group -> straight there; the whole
        squad -> fan the groups so they don't pile up."""
        gids = self._steered()
        if len(gids) == 1:
            self._send(gids[0], area)
        else:
            for gid, anchor in self._fan(area, gids).items():
                self._send(gid, anchor)

    def dispatch_all(self, area: tuple[float, float]) -> None:
        """Whole-squad send regardless of selection (used by recall/patrol)."""
        for gid, anchor in self._fan(area, list(self.squad.groups)).items():
            self._send(gid, anchor)

    # -- operator commands ----------------------------------------------------

    def cycle_formation(self) -> None:
        """Advance the steered group(s) to the next formation and re-issue."""
        for gid in self._steered():
            cur = self.formation.get(gid, Formation.WEDGE)
            nxt = SHAPES[(SHAPES.index(cur) + 1) % len(SHAPES)]
            if gid in self.area:
                self._send(gid, self.area[gid], nxt)
            else:
                self.formation[gid] = nxt
        who = self.selected or "ALL"
        print(f"[ctrl] {who} formation -> {self.formation.get(self._steered()[0]).value.upper()}", flush=True)

    def cycle_regroup(self) -> None:
        """Re-partition the squad into groups of the next size (1x6/2x3/3x2)."""
        self.size_i = (self.size_i + 1) % len(GROUP_SIZES)
        size = GROUP_SIZES[self.size_i]
        self.regroup_by_size(size)
        print(f"[ctrl] regroup -> {len(self.squad.groups)}x{size}  {self.squad.groups}", flush=True)

    def regroup_by_size(self, size: int) -> None:
        """Set an explicit group size (for web/ROS tasking)."""
        if size <= 0:
            return
        ids = sorted(self.squad.agent_ids, key=lambda a: self.squad.agents[a].get_state().y)
        self.dispatcher.regroup(split_evenly(ids, size))
        self.selected = None
        self.formation.clear()
        self.area.clear()

    def dispatch_target(
        self,
        area: tuple[float, float],
        *,
        group: str = "ALL",
        formation: Formation | None = None,
        spacing: float = 1.4,
    ) -> None:
        """Task-oriented dispatch API used by ROS/web ingress.

        group="ALL" fans assignments across groups. Specific group IDs send a
        single group mission.
        """
        self.patrol = False
        if group.upper() == "ALL":
            gids = list(self.squad.groups)
            if formation is None:
                for gid, anchor in self._fan(area, gids).items():
                    self._send(gid, anchor)
            else:
                for gid, anchor in self._fan(area, gids).items():
                    self.formation[gid] = formation
                    self.area[gid] = anchor
                    self.dispatcher.send(gid, anchor, formation, spacing=spacing)
            return

        # Match the group id case-insensitively so 'g1' resolves to 'G1'.
        gid = next((g for g in self.squad.groups if g.upper() == group.upper()), None)
        if gid is None:
            print(
                f"[ctrl][WARN] unknown group '{group}' "
                f"(known: {', '.join(self.squad.groups) or 'none'})",
                flush=True,
            )
            return
        if formation is None:
            self._send(gid, area)
        else:
            self.formation[gid] = formation
            self.area[gid] = area
            self.dispatcher.send(gid, area, formation, spacing=spacing)

    def toggle_patrol(self) -> None:
        self.patrol = not self.patrol
        print(f"[ctrl] patrol {'ON' if self.patrol else 'OFF'}", flush=True)
        if self.patrol:
            self.dispatch_all(ZONES[self.patrol_i][1])

    def recall(self) -> None:
        self.patrol = False
        print("[ctrl] recall -> HOME", flush=True)
        self.dispatch_all(HOME)

    def halt(self) -> None:
        self.patrol = False
        self.area.clear()
        for a in self.squad.agents.values():
            a.halt()
        self.dispatcher.missions.clear()
        print("[ctrl] HALT", flush=True)

    def apply(self, cmd: Command) -> bool:
        """Apply one operator Command. Returns False if it was 'quit'."""
        if cmd.verb == "zone":
            name, xy = ZONES[cmd.arg]
            self.patrol = False
            print(f"[ctrl] {self.selected or 'ALL'} -> {name} {xy}", flush=True)
            self.dispatch(xy)
        elif cmd.verb == "select":
            self.select_next()
        elif cmd.verb == "shape":
            self.cycle_formation()
        elif cmd.verb == "regroup":
            self.cycle_regroup()
        elif cmd.verb == "patrol":
            self.toggle_patrol()
        elif cmd.verb == "recall":
            self.recall()
        elif cmd.verb == "halt":
            self.halt()
        elif cmd.verb == "help":
            print(help_text([n for n, _ in ZONES]), flush=True)
        elif cmd.verb == "quit":
            print("[ctrl] quit requested.", flush=True)
            return False
        return True

    def apply_ros_command(self, payload: dict) -> bool:
        """Apply one normalized ROS command payload. Return False if quit."""
        verb = str(payload.get("verb", "")).lower()
        if verb == "dispatch":
            target = payload["target"]
            group = str(payload.get("group", "ALL"))
            spacing = float(payload.get("spacing", 1.4))
            formation = payload.get("formation")
            formation_value = Formation(formation) if formation else None
            xy = (float(target["x"]), float(target["y"]))
            print(f"[ros] dispatch group={group} target={xy} formation={formation or 'keep'}", flush=True)
            self.dispatch_target(xy, group=group, formation=formation_value, spacing=spacing)
            return True
        if verb == "regroup":
            size = int(payload["group_size"])
            self.regroup_by_size(size)
            print(f"[ros] regroup -> {len(self.squad.groups)}x{size} {self.squad.groups}", flush=True)
            return True
        if verb == "patrol":
            desired = bool(payload["enabled"])
            if desired != self.patrol:
                self.toggle_patrol()
            return True
        if verb == "recall":
            self.recall()
            return True
        if verb == "halt":
            self.halt()
            return True
        if verb == "help":
            print(help_text([n for n, _ in ZONES]), flush=True)
            return True
        if verb == "quit":
            print("[ros] quit requested.", flush=True)
            return False
        print(f"[ros][WARN] unsupported verb: {verb}", flush=True)
        return True

    def tick_patrol(self) -> None:
        """Advance the patrol to the next zone once every group has arrived."""
        if not self.patrol:
            return
        status = self.dispatcher.status()
        if status and all(status.values()):
            self.patrol_i = (self.patrol_i + 1) % len(ZONES)
            name, xy = ZONES[self.patrol_i]
            print(f"[patrol] next -> {name}", flush=True)
            self.dispatch_all(xy)


class DispatchConsoleHandle:
    """Adapter exposing the live squad to the dispatch extension (runtime.DispatchHandle).

    Translates the extension's verb/snapshot contract onto the existing
    SquadController — the extension is just another ingress, like the keyboard/ROS
    paths. Runs on the sim's main thread (the extension's callbacks fire inside
    world.step), so no locking is needed."""

    def __init__(self, squad: Squad, agents: dict, controller: SquadController) -> None:
        self._squad = squad
        self._agents = agents
        self._ctrl = controller

    def snapshot(self) -> dict:
        dogs = []
        for aid, a in self._agents.items():
            st = a.get_state()
            dogs.append({
                "id": aid, "group": a.group_id, "x": st.x, "y": st.y, "yaw": st.yaw,
                "tx": None if a.target is None else a.target[0],
                "ty": None if a.target is None else a.target[1],
                "arrived": a.arrived,
            })
        return {
            "zones": [{"name": n, "x": xy[0], "y": xy[1]} for n, xy in ZONES],
            "groups": {gid: list(ids) for gid, ids in self._squad.groups.items()},
            "selected": self._ctrl.selected,
            "formation": {g: f.value for g, f in self._ctrl.formation.items()},
            "patrol": self._ctrl.patrol,
            "dogs": dogs,
        }

    def dispatch_zone(self, index: int) -> None:
        if 0 <= index < len(ZONES):
            self._ctrl.apply(Command(verb="zone", arg=index))

    def select_next(self) -> None:
        self._ctrl.select_next()

    def cycle_formation(self) -> None:
        self._ctrl.cycle_formation()

    def cycle_regroup(self) -> None:
        self._ctrl.cycle_regroup()

    def toggle_patrol(self) -> None:
        self._ctrl.toggle_patrol()

    def recall(self) -> None:
        self._ctrl.recall()

    def halt(self) -> None:
        self._ctrl.halt()


class Go2SquadDemo:
    def __init__(self) -> None:
        self.world = None
        self.agents: dict[str, RobotAgent] = {}
        self.squad = None
        self.dispatcher = None
        self.controller = None
        self.console = KeyboardConsole()
        self.ros_bridge: SquadRosBridge | None = None
        # --auto scripted-run state (unchanged behaviour)
        self.phase = 1
        self._regrouped = False

    def setup(self) -> bool:
        self.world = World(
            physics_dt=PHYSICS_DT,
            rendering_dt=PHYSICS_DT * RENDER_EVERY,
            stage_units_in_meters=1.0,
        )

        assets_root = get_assets_root_path()
        if assets_root is None:
            carb.log_error("Isaac assets root not found")
            return False
        if args_cli.ground:
            # A/B test: the physics floor the proven smoke harness used.
            self.world.scene.add_default_ground_plane()
            _apply_floor_friction("/World/defaultGroundPlane")
        else:
            # Warehouse backdrop (same asset family as scripts/09): shelves + floor.
            stage_utils.add_reference_to_stage(
                usd_path=assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                path="/World/Warehouse",
            )
            _apply_floor_friction("/World/Warehouse")
        _add_lighting()

        # Spawn 6 Go2 + wrap each in the ABI agent.
        for i, (x, y) in enumerate(START_XY):
            backend = Go2Locomotion(prim_path=f"/World/Go2_{i}", position=[x, y, STAND_Z])
            agent = RobotAgent(f"d{i}", backend, params={"arrive_tol": args_cli.arrive_tol})
            self.agents[f"d{i}"] = agent

        self.world.reset()
        for agent in self.agents.values():
            agent.backend.initialize()
        _set_stream_camera()

        # Settle a few steps so the policy stabilises the stance before commands.
        for _ in range(30):
            for a in self.agents.values():
                a.backend.apply(PHYSICS_DT, (0.0, 0.0, 0.0))
            self.world.step(render=False)

        self.squad = Squad(self.agents)
        self.dispatcher = Dispatcher(self.squad)
        self.controller = SquadController(self.squad, self.dispatcher)

        # Hand the live squad to the dispatch console extension (if it loaded).
        try:
            from devicenexus.squad.dispatch import runtime as _dispatch_runtime

            _dispatch_runtime.set_handle(DispatchConsoleHandle(self.squad, self.agents, self.controller))
            print("[demo] dispatch console extension connected.", flush=True)
        except Exception as exc:  # noqa: BLE001 -- console is optional
            print(f"[demo] dispatch console not connected: {exc}", flush=True)

        if args_cli.ros_control:
            topics = RosTopics(
                command_topic=args_cli.ros_cmd_topic,
                status_topic=args_cli.ros_status_topic,
                action_topic=args_cli.ros_action_topic,
            )
            self.ros_bridge = SquadRosBridge(node_name=args_cli.ros_node_name, topics=topics)
            self.ros_bridge.start()
            print(
                "[ros] control interface enabled: "
                f"cmd={topics.command_topic}, status={topics.status_topic}, action={topics.action_topic}",
                flush=True,
            )

        if args_cli.auto:
            self._setup_auto()
        else:
            # Start as 2x3, idle, and let the operator drive.
            self.squad.set_groups(split_evenly(self.squad.agent_ids, 3))
            self.controller.size_i = 1  # GROUP_SIZES index for size=3
            if not args_cli.capture and not args_cli.ros_control:
                print(help_text([n for n, _ in ZONES]), flush=True)
                self.console.start()
            elif args_cli.ros_control:
                print("[ctrl] waiting for ROS dispatch commands (keyboard disabled).", flush=True)

        if args_cli.capture:
            self._capture(args_cli.capture)
            return False
        return True

    def _capture(self, path: str) -> None:
        """Offline smoke/diagnostic. With --capture-secs "0,6,15" run the demo and
        snapshot at each sim-time (suffix _t<sec>.png) — lets us watch whether the
        gait holds or the dogs fall and crawl. Without it, one frame after settle."""
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        def snap(tag: str) -> None:
            for _ in range(20):
                self.world.step(render=True)
            out = path if not tag else path.replace(".png", f"_t{tag}.png")
            capture_viewport_to_file(get_active_viewport(), out)
            for _ in range(40):
                simulation_app.update()
            standing = sum(self.agents[a].get_state() is not None for a in self.agents)
            print(f"[capture] saved {out} (t={tag or '0'}s, agents={standing})", flush=True)

        times = [float(t) for t in args_cli.capture_secs.split(",")] if args_cli.capture_secs else [0.0]
        t_done = 0.0
        for t in times:
            steps = max(0, int((t - t_done) / PHYSICS_DT))
            for i in range(steps):
                if args_cli.auto:
                    self._tick_auto()
                for agent in self.agents.values():
                    agent.step(PHYSICS_DT)
                self.world.step(render=(i % RENDER_EVERY == 0))
            t_done = t
            snap("" if t == 0.0 else f"{int(t)}")

    def _setup_auto(self) -> None:
        """Original scripted 2x3 -> staging -> regroup -> 3x2 run."""
        self.squad.set_groups(split_evenly(self.squad.agent_ids, 3))
        for gid, area in {"G0": (6.0, 2.0), "G1": (6.0, -2.0)}.items():
            self.dispatcher.send(gid, area, "wedge", spacing=1.4)
        print("\n" + "=" * 70)
        print("  Go2 Squad Dispatch (AUTO) — 2x3 -> regroup -> 3x2")
        print("=" * 70 + "\n", flush=True)

    def _tick_auto(self) -> None:
        targets = {"G0": (12.0, -3.0), "G1": (12.0, 0.0), "G2": (12.0, 3.0)}
        status = self.dispatcher.update(PHYSICS_DT)
        if self.phase == 1 and status and all(status.values()) and not self._regrouped:
            self._regrouped = True
            self.phase = 2
            ids_by_y = sorted(self.squad.agent_ids, key=lambda a: self.agents[a].get_state().y)
            groups = {f"G{i}": ids_by_y[2 * i:2 * i + 2] for i in range(len(targets))}
            self.dispatcher.regroup(groups)
            targets_by_y = sorted(targets.values(), key=lambda t: t[1])
            for gid, area in zip(groups.keys(), targets_by_y):
                self.dispatcher.send(gid, area, "row", spacing=1.4)
            print(f"[auto] PHASE 1 complete -> REGROUP 3x2: {self.squad.groups}", flush=True)
        elif self.phase == 2 and status and all(status.values()):
            self.phase = 3
            print("[auto] PHASE 2 complete — squad at targets.", flush=True)

    def _build_markers(self) -> list[dict]:
        """rviz markers in the /map frame: each dog (arrow=pose, colored by group +
        id label), the named zones (disk + label), and each dog's current target."""
        items: list[dict] = []
        # Zones: a flat translucent disk + a floating name label.
        for zi, (name, (zx, zy)) in enumerate(ZONES):
            items.append({"ns": "zones", "id": zi, "shape": "cylinder", "x": zx, "y": zy,
                          "z": 0.02, "sx": 1.4, "sy": 1.4, "sz": 0.04,
                          "r": 0.2, "g": 0.8, "b": 0.9, "a": 0.45})
            items.append({"ns": "zone_labels", "id": zi, "shape": "text", "x": zx, "y": zy,
                          "z": 0.6, "sz": 0.5, "r": 0.9, "g": 0.95, "b": 1.0, "a": 0.9,
                          "text": name})
        # Dogs: an arrow (position + heading) colored by group, with an id label.
        for di, (aid, a) in enumerate(self.agents.items()):
            st = a.get_state()
            cr, cg, cb = _group_color(a.group_id)
            items.append({"ns": "dogs", "id": di, "shape": "arrow", "x": st.x, "y": st.y,
                          "z": 0.25, "yaw": st.yaw, "sx": 0.7, "sy": 0.14, "sz": 0.14,
                          "r": cr, "g": cg, "b": cb, "a": 1.0})
            items.append({"ns": "dog_labels", "id": di, "shape": "text", "x": st.x, "y": st.y,
                          "z": 0.6, "sz": 0.28, "r": 1.0, "g": 1.0, "b": 1.0, "a": 0.9,
                          "text": f"{aid}/{a.group_id or '-'}"})
            if a.target is not None:
                items.append({"ns": "targets", "id": di, "shape": "sphere",
                              "x": a.target[0], "y": a.target[1], "z": 0.1,
                              "sx": 0.35, "sy": 0.35, "sz": 0.35,
                              "r": cr, "g": cg, "b": cb, "a": 0.5})
        return items

    def run(self) -> None:
        substeps = int(args_cli.max_seconds / PHYSICS_DT)
        for i in range(substeps):
            if not simulation_app.is_running():
                return
            if args_cli.auto:
                self._tick_auto()
            else:
                if self.ros_bridge is not None:
                    for cmd in self.ros_bridge.poll_commands():
                        if not self.controller.apply_ros_command(cmd):
                            return
                else:
                    for cmd in self.console.poll():
                        if not self.controller.apply(cmd):
                            return
                self.controller.tick_patrol()

            for agent in self.agents.values():
                cmd = agent.step(PHYSICS_DT)    # ABI -> velocity cmd -> policy.forward
                if self.ros_bridge is not None:
                    self.ros_bridge.publish_action(
                        {
                            "t": round(i * PHYSICS_DT, 3),
                            "agent_id": agent.agent_id,
                            "group": agent.group_id,
                            "command": {"vx": cmd[0], "vy": cmd[1], "wz": cmd[2]},
                            "target": None if agent.target is None else {"x": agent.target[0], "y": agent.target[1]},
                            "arrived": agent.arrived,
                        }
                    )
            self.world.step(render=(i % RENDER_EVERY == 0))

            if self.ros_bridge is not None and i % 20 == 0:
                zones = [{"name": n, "x": xy[0], "y": xy[1]} for n, xy in ZONES]
                self.ros_bridge.publish_status(
                    {
                        "t": round(i * PHYSICS_DT, 3),
                        "phase": self.phase,
                        "groups": self.squad.groups,
                        "patrol": self.controller.patrol if self.controller is not None else False,
                        "selected": self.controller.selected if self.controller is not None else None,
                        "zones": zones,
                        "agents": {
                            aid: {
                                "group": a.group_id,
                                "pos": {"x": (s := a.get_state()).x, "y": s.y, "yaw": s.yaw},
                                "target": None if a.target is None else {"x": a.target[0], "y": a.target[1]},
                                "arrived": a.arrived,
                            }
                            for aid, a in self.agents.items()
                        },
                    }
                )
                self.ros_bridge.publish_markers(self._build_markers())

            if i % 200 == 0:
                arrived = sum(a.arrived for a in self.agents.values())
                mode = f"phase={self.phase}" if args_cli.auto else f"groups={len(self.squad.groups)}"
                print(f"[t={i*PHYSICS_DT:5.1f}s] {mode} arrived={arrived}/6", flush=True)
        print("[demo] budget reached; exiting.", flush=True)

    def close(self) -> None:
        try:
            from devicenexus.squad.dispatch import runtime as _dispatch_runtime

            _dispatch_runtime.set_handle(None)
        except Exception:  # noqa: BLE001
            pass
        self.console.stop()
        if self.ros_bridge is not None:
            self.ros_bridge.close()
        if self.world is not None:
            self.world.stop()


def main() -> None:
    demo = Go2SquadDemo()
    try:
        if demo.setup():
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
