# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001, E402
"""Fast framing/lighting probe for the Go2 warehouse demo (no RL policies).

Loads the same warehouse, adds a dome light, drops marker cubes where the dogs
and zones sit, points a candidate camera, and captures a PNG so we can SEE the
frame and tune camera/light without booting 6 Go2 policies. Prints the warehouse
bounding box so the camera can be placed inside the building.

Run: python scripts/_go2_warehouse_probe.py --eye -9,0,4 --target 2,0,0.4 --out /tmp/wh.png
"""
import argparse
import os
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--eye", default="-9,0,4")
parser.add_argument("--target", default="2,0,0.4")
parser.add_argument("--out", default="/tmp/wh_probe.png")
parser.add_argument("--intensity", type=float, default=1500.0)
args = parser.parse_args()


def _xyz(s):
    return [float(v) for v in s.split(",")]


simulation_app = SimulationApp({"headless": True})

import omni.kit.app
from omni.ext import ExtensionPathType
import isaacsim

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_ext_mgr.add_path(os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"),
                  ExtensionPathType.COLLECTION)
for _ext in ("isaacsim.core.api",):
    _ext_mgr.set_extension_enabled_immediate(_ext, True)
simulation_app.update()

import carb
import omni.usd
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdLux, UsdGeom, Gf, Sdf, UsdShade

assets_root = get_assets_root_path()
stage_utils.add_reference_to_stage(
    usd_path=assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    path="/World/Warehouse",
)
for _ in range(10):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()

# --- bounding box of the warehouse so we know the interior extents ---
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
rng = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Warehouse")).ComputeAlignedRange()
print(f"[probe] warehouse bbox min={tuple(round(v,2) for v in rng.GetMin())} "
      f"max={tuple(round(v,2) for v in rng.GetMax())}", flush=True)

# --- dome light so the interior is not pitch black ---
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(args.intensity)
distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
distant.CreateIntensityAttr(args.intensity)
distant.CreateAngleAttr(1.0)
UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 0))

# --- marker cubes: red = dog starts, green = zones ---
starts = [(-1.5, 1.2), (-1.5, 0.0), (-1.5, -1.2), (-3.0, 1.2), (-3.0, 0.0), (-3.0, -1.2)]
zones = [(5.0, 2.5), (5.0, -2.5), (-5.0, 0.0), (0.0, 3.5), (0.0, -3.5)]


def _marker(path, xy, color, size=0.4, z=0.3):
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(xy[0], xy[1], z))


for i, xy in enumerate(starts):
    _marker(f"/World/Markers/dog_{i}", xy, (1.0, 0.1, 0.1))
for i, xy in enumerate(zones):
    _marker(f"/World/Markers/zone_{i}", xy, (0.1, 1.0, 0.2), size=0.6)

for _ in range(10):
    simulation_app.update()

# --- camera ---
from isaacsim.core.utils.viewports import set_camera_view
set_camera_view(eye=_xyz(args.eye), target=_xyz(args.target))
for _ in range(20):
    simulation_app.update()

# --- capture ---
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
vp = get_active_viewport()
capture_viewport_to_file(vp, args.out)
for _ in range(60):
    simulation_app.update()
print(f"[probe] saved {args.out}", flush=True)

simulation_app.close()
