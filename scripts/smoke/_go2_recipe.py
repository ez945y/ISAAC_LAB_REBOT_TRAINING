# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001, E402
"""Replicate the shipped isaacsim Go2 example recipe EXACTLY and verify the dog
stays UPRIGHT (height), not just moves forward. Isolates what keeps it standing:
grid ground + friction=1.0 bound to the collision plane + post_reset + callback.
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

import omni.usd
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.storage.native import get_assets_root_path
from controll_scripts.squad.locomotion import Go2Locomotion
from pxr import Sdf, UsdPhysics, UsdShade
import warp as wp

DT = 1.0 / 200.0


def height(d):
    return float(wp.to_torch(d.policy.robot.get_world_poses()[0]).reshape(-1)[2])


def main() -> None:
    # The shipped example sets these BEFORE creating the robot. GPU dynamics +
    # torch backend is what the policy was trained on; CPU physics collapses it.
    SimulationManager.set_backend("torch")
    SimulationManager.set_physics_sim_device("cuda")
    world = World(physics_dt=DT, rendering_dt=0.02, stage_units_in_meters=1.0, device="cuda", backend="torch")
    assets = get_assets_root_path()
    stage_utils.add_reference_to_stage(usd_path=assets + "/Isaac/Environments/Grid/default_environment.usd",
                                       path="/World/ground")
    # friction=1.0 bound DIRECTLY to the collision plane (exactly like the example)
    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, Sdf.Path("/World/ground/Looks/PhysicsMaterial"))
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(1.0)
    pm.CreateDynamicFrictionAttr().Set(1.0)
    pm.CreateRestitutionAttr().Set(0.0)
    geom = stage.GetPrimAtPath("/World/ground/GroundPlane/CollisionPlane")
    print(f"[recipe] CollisionPlane valid={geom.IsValid()}", flush=True)
    if geom.IsValid():
        UsdShade.MaterialBindingAPI.Apply(geom).Bind(mat, materialPurpose="physics")

    dog = Go2Locomotion(prim_path="/World/Go2", position=[0.0, 0.0, 0.5])
    world.reset()
    dog.initialize()
    dog.policy.post_reset()              # <-- reset to default joint pose (example does this)

    cmd = [0.0, 0.0, 0.0]

    def on_phys(step_size, _ctx):
        dog.apply(step_size, tuple(cmd))

    SimulationManager.register_callback(on_phys, IsaacEvents.POST_PHYSICS_STEP)

    for _ in range(200):
        world.step(render=False)
    print(f"[recipe] settled height={height(dog):.3f}", flush=True)
    cmd[:] = [1.0, 0.0, 0.0]
    for k in range(600):
        world.step(render=False)
        if k % 150 == 0:
            print(f"  t={k*DT:.2f}s height={height(dog):.3f} x={dog.get_pose()[0]:.2f}", flush=True)
    h = height(dog)
    print(f"[recipe] {'PASS upright' if h > 0.28 else 'FAIL collapsed'} final height={h:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
