# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001 -- Isaac modules imported only after SimulationApp starts.

"""Unitree Go2 squad dispatch (Isaac Sim 6.0).

Six Go2 quadrupeds start as TWO groups of THREE, advance in formation to two
staging areas, then the dispatcher REGROUPS them into THREE groups of TWO that
move to three target areas.

Architecture (see tools/controll_scripts/squad/):
    formation/scheduler/mission/robot_agent  — pure-logic, Isaac-free, tested in
        tests/test_squad_dispatch.py
    locomotion.Go2Locomotion                 — the only Go2-bound piece (RL policy)

Locomotion is the shipped Go2FlatTerrainPolicy (pretrained policy auto-detected for
the active engine — physx_policy.pt under PhysX). The RobotAgent ABI turns a move_to
target into a [v_x, v_y, w_z] command per physics step; the scheduler/missions never
see Isaac or the Go2.

Boots the NATIVE isaacsim SimulationApp (not Isaac Lab's AppLauncher, whose
PhysxManager breaks the native Go2 policy). Headless by default; pass --livestream 2
for the WebRTC stream.

Run:  python scripts/13_go2_squad_dispatch.py [--livestream 2] [--max-seconds N]
"""

import argparse
import os
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Go2 squad 2x3 -> 3x2 dispatch demo")
parser.add_argument("--max-seconds", type=float, default=60.0, help="Wall budget for the run.")
parser.add_argument("--arrive-tol", type=float, default=0.6, help="Slot arrival tolerance (m).")
parser.add_argument("--livestream", type=int, default=0, help="1/2 = enable WebRTC livestream.")
args_cli = parser.parse_args()

# The Go2 RL policy is a NATIVE-isaacsim component and does NOT run under Isaac
# Lab's AppLauncher (it installs its own PhysxManager as SimulationManager, which
# lacks set_backend and breaks the native experimental Articulation). So boot the
# native SimulationApp directly — same harness as scripts/_go2_smoke.py.
simulation_app = SimulationApp({"headless": True})

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

# ── Enable the native extensions the base app does not auto-load ──
# `isaacsim.core.api` World lives in extsDeprecated; the policy needs
# `isaacsim.robot.policy.examples`. (See scripts/_go2_smoke.py — same bootstrap.)
import omni.kit.app  # noqa: E402
from omni.ext import ExtensionPathType  # noqa: E402

import isaacsim  # noqa: E402

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_ext_mgr.add_path(
    os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"),
    ExtensionPathType.COLLECTION,
)
for _ext in ("isaacsim.core.api", "isaacsim.robot.policy.examples"):
    _ext_mgr.set_extension_enabled_immediate(_ext, True)
simulation_app.update()

if args_cli.livestream >= 1:
    from livestream.livestream_support import enable_native_livestream  # noqa: E402

    enable_native_livestream()

# ── Isaac imports after the app is up ───────────────────────────────────────
import carb  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from controll_scripts.squad import (  # noqa: E402
    Dispatcher, Formation, FormationMoveToArea, RobotAgent, Squad, split_evenly,
)
from controll_scripts.squad.locomotion import Go2Locomotion  # noqa: E402

PHYSICS_DT = 0.005          # 200 Hz, matches policy training
RENDER_EVERY = 4            # render at ~50 Hz
STAND_Z = 0.5

# 6 dogs: two columns of three at the start line.
START_XY = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, -1.0), (0.0, -2.0), (0.0, -3.0)]
STAGING = {"G0": (6.0, 2.0), "G1": (6.0, -2.0)}                       # phase 1 (2x3)
TARGETS = {"G0": (12.0, -3.0), "G1": (12.0, 0.0), "G2": (12.0, 3.0)}  # phase 2 (3x2)


class Go2SquadDemo:
    def __init__(self) -> None:
        self.world = None
        self.agents: dict[str, RobotAgent] = {}
        self.squad = None
        self.dispatcher = None
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
        stage_utils.add_reference_to_stage(
            usd_path=assets_root + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        # Spawn 6 Go2 + wrap each in the ABI agent.
        for i, (x, y) in enumerate(START_XY):
            backend = Go2Locomotion(prim_path=f"/World/Go2_{i}", position=[x, y, STAND_Z])
            agent = RobotAgent(f"d{i}", backend, params={"arrive_tol": args_cli.arrive_tol})
            self.agents[f"d{i}"] = agent

        self.world.reset()
        for agent in self.agents.values():
            agent.backend.initialize()

        # Settle a few steps so the policy stabilises the stance before commands.
        for _ in range(30):
            for a in self.agents.values():
                a.backend.apply(PHYSICS_DT, (0.0, 0.0, 0.0))
            self.world.step(render=False)

        self.squad = Squad(self.agents)
        self.dispatcher = Dispatcher(self.squad)

        # ── Phase 1: two groups of three to the staging areas ──
        self.squad.set_groups(split_evenly(self.squad.agent_ids, 3))
        for gid, area in STAGING.items():
            self.dispatcher.assign(gid, FormationMoveToArea(area, shape=Formation.WEDGE, spacing=1.4))

        print("\n" + "=" * 70)
        print("  Go2 Squad Dispatch — 2x3 -> regroup -> 3x2")
        print(f"  Phase 1: {self.squad.groups}  ->  staging {STAGING}")
        print("=" * 70 + "\n", flush=True)
        return True

    def _tick_dispatch(self) -> None:
        status = self.dispatcher.update(PHYSICS_DT)
        if self.phase == 1 and status and all(status.values()) and not self._regrouped:
            self._regrouped = True
            self.phase = 2
            # Regroup SPATIALLY: pair dogs by current lateral (y) position and send
            # each pair to its nearest objective. The dogs have no mutual collision
            # avoidance, so an id-order regroup (which sends top dogs to the bottom
            # target and vice-versa) makes groups cross the field and pile up. A
            # monotone position->target mapping keeps the paths crossing-free.
            ids_by_y = sorted(self.squad.agent_ids, key=lambda a: self.agents[a].get_state().y)
            groups = {f"G{i}": ids_by_y[2 * i:2 * i + 2] for i in range(len(TARGETS))}
            self.dispatcher.regroup(groups)
            targets_by_y = sorted(TARGETS.values(), key=lambda t: t[1])
            for gid, area in zip(groups.keys(), targets_by_y):
                self.dispatcher.assign(gid, FormationMoveToArea(area, shape=Formation.ROW, spacing=1.4))
            print(f"[dispatch] PHASE 1 complete -> REGROUP 3x2: {self.squad.groups}", flush=True)
        elif self.phase == 2 and status and all(status.values()):
            self.phase = 3
            print("[dispatch] PHASE 2 complete — squad at targets.", flush=True)

    def run(self) -> None:
        substeps = int(args_cli.max_seconds / PHYSICS_DT)
        for i in range(substeps):
            if not simulation_app.is_running():
                return
            self._tick_dispatch()
            for agent in self.agents.values():
                agent.step(PHYSICS_DT)          # ABI -> velocity cmd -> policy.forward
            self.world.step(render=(i % RENDER_EVERY == 0))
            if i % 200 == 0:
                arrived = sum(a.arrived for a in self.agents.values())
                print(f"[t={i*PHYSICS_DT:5.1f}s] phase={self.phase} arrived={arrived}/6", flush=True)
                # When stragglers remain, show each unarrived dog's pos/target/dist.
                if arrived < 6 and self.phase in (1, 2):
                    import math as _m
                    for aid, a in self.agents.items():
                        if a.arrived or a.target is None:
                            continue
                        st = a.get_state()
                        d = _m.hypot(a.target[0] - st.x, a.target[1] - st.y)
                        print(f"    {aid}: pos=({st.x:5.2f},{st.y:5.2f}) tgt=({a.target[0]:4.1f},"
                              f"{a.target[1]:4.1f}) dist={d:4.2f} yaw={st.yaw:5.2f}", flush=True)
        # Hold open only when livestreaming (so the client keeps a picture);
        # a headless run exits at the budget instead of spinning forever.
        if args_cli.livestream >= 1:
            print("[demo] budget reached; holding for livestream client.", flush=True)
            while simulation_app.is_running():
                self.world.step(render=True)
        else:
            print("[demo] budget reached; exiting.", flush=True)

    def close(self) -> None:
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
