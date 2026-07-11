# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001

"""Minimal Go2 locomotion smoke test on NATIVE isaacsim (headless).

The shipped Go2FlatTerrainPolicy is a native-isaacsim component (uses the native
SimulationManager / experimental Articulation). It does NOT run under Isaac Lab's
SimulationContext (which swaps in its own PhysxManager). So this uses the native
`isaacsim.SimulationApp` + `World`.

Launch -> World -> one Go2 (pretrained policy) -> forward command -> check it
walks forward.  Run:  python scripts/smoke/_go2_smoke.py
"""

import os
import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))

import omni.kit.app  # noqa: E402


def _enable(ext_id: str) -> None:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(ext_id, True)


def _register_deprecated() -> None:
    """isaacsim.core.api.World lives in extsDeprecated — register that collection."""
    import isaacsim
    from omni.ext import ExtensionPathType

    dep = os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated")
    omni.kit.app.get_app().get_extension_manager().add_path(dep, ExtensionPathType.COLLECTION)


_register_deprecated()
_enable("isaacsim.core.api")
_enable("isaacsim.robot.policy.examples")
simulation_app.update()

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402

from controll_scripts.squad.locomotion import Go2Locomotion  # noqa: E402

DT = 1.0 / 200.0  # 200 Hz, matches the policy training (shipped test uses this)

# Command driven into the per-physics-step callback.
_cmd = [0.0, 0.0, 0.0]


def main() -> None:
    world = World(physics_dt=DT, rendering_dt=1.0 / 50.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    print("[smoke] creating Go2 (loads pretrained policy)...", flush=True)
    dog = Go2Locomotion(prim_path="/World/Go2", position=[0.0, 0.0, 0.5])
    try:
        print("[smoke] active physics engine:", SimulationManager.get_active_physics_engine(), flush=True)
    except Exception as e:
        print("[smoke] engine query failed:", e, flush=True)

    world.reset()
    dog.initialize()

    # Run the policy EVERY physics step via the callback (the key fix).
    def on_physics_step(step_size, _ctx):
        dog.apply(step_size, tuple(_cmd))

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    print("[smoke] initialized; settling (zero command)...", flush=True)
    for _ in range(200):
        world.step(render=False)

    x0, y0, yaw0 = dog.get_pose()
    print(f"[smoke] settled pose=({x0:.3f},{y0:.3f},{yaw0:.2f}); commanding v_x=1.0", flush=True)
    _cmd[:] = [1.0, 0.0, 0.0]
    for i in range(400):  # 2 s
        world.step(render=False)
        if i % 100 == 0:
            x, y, z = dog.get_pose()[0], dog.get_pose()[1], 0.0
            print(f"  step {i}: x={x:.3f} y={y:.3f}", flush=True)

    x1, y1, _ = dog.get_pose()
    dx = x1 - x0
    print(f"[smoke] RESULT dx={dx:.3f} m (lateral drift dy={y1-y0:.3f})", flush=True)
    print("[smoke] PASS — Go2 walks" if dx > 0.5 else "[smoke] DID NOT WALK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
