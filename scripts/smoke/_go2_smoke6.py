# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001, E402
"""6-dog gait test: does driving via POST_PHYSICS_STEP callback keep them up?

Mirrors scripts/smoke/_go2_smoke.py but spawns 6 Go2 and drives every one through the
SimulationManager POST_PHYSICS_STEP callback (the harness that walks). If they
walk and stay standing, script 13's collapse is the inline-drive, not the policy.
"""
import os
import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))

import omni.kit.app
from omni.ext import ExtensionPathType
import isaacsim

_m = omni.kit.app.get_app().get_extension_manager()
_m.add_path(os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"), ExtensionPathType.COLLECTION)
for _e in ("isaacsim.core.api", "isaacsim.robot.policy.examples"):
    _m.set_extension_enabled_immediate(_e, True)
simulation_app.update()

from isaacsim.core.api import World
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from controll_scripts.squad.locomotion import Go2Locomotion

DT = 1.0 / 200.0
START = [(-1.5, 1.2), (-1.5, 0.0), (-1.5, -1.2), (-3.0, 1.2), (-3.0, 0.0), (-3.0, -1.2)]


def main() -> None:
    world = World(physics_dt=DT, rendering_dt=1.0 / 50.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    dogs = [Go2Locomotion(prim_path=f"/World/Go2_{i}", position=[x, y, 0.5]) for i, (x, y) in enumerate(START)]
    world.reset()
    for d in dogs:
        d.initialize()

    cmd = [0.0, 0.0, 0.0]

    def on_phys(step_size, _ctx):
        for d in dogs:
            d.apply(step_size, tuple(cmd))

    SimulationManager.register_callback(on_phys, IsaacEvents.POST_PHYSICS_STEP)

    for _ in range(200):  # settle, zero command (callback drives)
        world.step(render=False)
    z0 = [d.policy.robot.get_world_poses()[0] for d in dogs]
    import warp as wp
    h0 = [float(wp.to_torch(p).reshape(-1)[2]) for p in z0]
    x0 = [d.get_pose()[0] for d in dogs]
    print(f"[smoke6] settled heights={[round(h,2) for h in h0]}", flush=True)

    cmd[:] = [1.0, 0.0, 0.0]
    for _ in range(600):  # 3 s walking
        world.step(render=False)
    h1 = [float(wp.to_torch(d.policy.robot.get_world_poses()[0]).reshape(-1)[2]) for d in dogs]
    x1 = [d.get_pose()[0] for d in dogs]
    dx = [round(x1[i] - x0[i], 2) for i in range(len(dogs))]
    print(f"[smoke6] after 3s heights={[round(h,2) for h in h1]}", flush=True)
    print(f"[smoke6] forward dx={dx}", flush=True)
    standing = sum(h > 0.25 for h in h1)
    print(f"[smoke6] {'PASS' if standing == len(dogs) else 'FAIL'} — {standing}/{len(dogs)} standing", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
