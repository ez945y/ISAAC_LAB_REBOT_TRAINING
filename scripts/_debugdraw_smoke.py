# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001

"""Smoke test: does isaacsim.util.debug_draw render into the (streamed) viewport?

If DebugDraw points/lines show up in a headless viewport capture, then the same
draw calls show up in the WebRTC stream — i.e. we can paint zones/targets/paths
right in the stream and skip rviz entirely. Saves /tmp/debugdraw_smoke.png."""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402
from omni.ext import ExtensionPathType  # noqa: E402

import isaacsim  # noqa: E402

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_ext_mgr.add_path(os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"),
                  ExtensionPathType.COLLECTION)
for _ext in ("isaacsim.core.api", "isaacsim.util.debug_draw"):
    _ext_mgr.set_extension_enabled_immediate(_ext, True)
simulation_app.update()

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.util.debug_draw import _debug_draw  # noqa: E402
from pxr import Sdf, UsdLux  # noqa: E402
import omni.usd  # noqa: E402

world = World(physics_dt=0.005, rendering_dt=0.02, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Dome")).CreateIntensityAttr(1500.0)
world.reset()
set_camera_view(eye=[6.0, 6.0, 6.0], target=[0.0, 0.0, 0.0])

draw = _debug_draw.acquire_debug_draw_interface()
# Zone points (big colored dots) + target dots + a path line + a closed ring spline.
draw.draw_points(
    [(5, 2.5, 0.1), (5, -2.5, 0.1), (-5, 0, 0.1)],
    [(0.2, 0.8, 0.9, 1), (0.2, 0.8, 0.9, 1), (0.2, 0.8, 0.9, 1)],
    [25, 25, 25],
)
draw.draw_lines([(0, 0, 0.1)], [(5, 2.5, 0.1)], [(1, 0.6, 0.1, 1)], [4])
ring = [(2 + 0.6 * __import__("math").cos(t * 0.628),
         0.6 * __import__("math").sin(t * 0.628), 0.1) for t in range(10)]
draw.draw_lines_spline(ring, (0.3, 0.9, 0.4, 1), 4, True)

from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

for _ in range(30):
    world.step(render=True)
out = "/tmp/debugdraw_smoke.png"
capture_viewport_to_file(get_active_viewport(), out)
for _ in range(40):
    simulation_app.update()
print(f"[smoke] lines={draw.get_num_lines()} points={draw.get_num_points()} -> {out}", flush=True)
simulation_app.close()
