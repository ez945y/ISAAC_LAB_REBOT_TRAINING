# ruff: noqa: I001
"""Decisive test: does the native Go2 policy walk when booted via Isaac Lab's
AppLauncher (09's proven WebRTC path) instead of a bare isaacsim.SimulationApp?

If dx > 0.5 here, 13 can use AppLauncher + apply_livestream_defaults exactly like
09 and we drop the custom native-livestream helper.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))

import omni.kit.app  # noqa: E402
from omni.ext import ExtensionPathType  # noqa: E402

import isaacsim  # noqa: E402

_mgr = omni.kit.app.get_app().get_extension_manager()
_mgr.add_path(os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"), ExtensionPathType.COLLECTION)
for _ext in ("isaacsim.core.api", "isaacsim.robot.policy.examples"):
    _mgr.set_extension_enabled_immediate(_ext, True)
simulation_app.update()

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from controll_scripts.squad.locomotion import Go2Locomotion  # noqa: E402

DT = 1.0 / 200.0


def main() -> None:
    world = World(physics_dt=DT, rendering_dt=1.0 / 50.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    print("[al-smoke] SimulationManager class:", type(SimulationManager).__name__, flush=True)
    print("[al-smoke] creating Go2...", flush=True)
    dog = Go2Locomotion(prim_path="/World/Go2", position=[0.0, 0.0, 0.5])

    world.reset()
    dog.initialize()

    print("[al-smoke] settling...", flush=True)
    for _ in range(200):
        dog.apply(DT, (0.0, 0.0, 0.0))
        world.step(render=False)

    x0, y0, _ = dog.get_pose()
    print(f"[al-smoke] settled=({x0:.3f},{y0:.3f}); commanding v_x=1.0", flush=True)
    for i in range(400):
        dog.apply(DT, (1.0, 0.0, 0.0))
        world.step(render=False)
        if i % 100 == 0:
            x, y, _ = dog.get_pose()
            print(f"  step {i}: x={x:.3f} y={y:.3f}", flush=True)

    x1, y1, _ = dog.get_pose()
    dx = x1 - x0
    print(f"[al-smoke] RESULT dx={dx:.3f} m (dy={y1-y0:.3f})", flush=True)
    print("[al-smoke] PASS — AppLauncher works for Go2" if dx > 0.5 else "[al-smoke] FAIL", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
