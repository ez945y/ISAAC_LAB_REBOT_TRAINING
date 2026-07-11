# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001 -- Isaac modules imported only after SimulationApp starts.

"""Offline occupancy-map generator for the Simple_Warehouse (Isaac Sim 6.0).

Pre-bakes ONE static ROS occupancy grid of the warehouse that every Go2 shares
(the map is the environment, not per-robot). Output is the standard ROS map_server
pair under tools/ros/maps/:
    warehouse.pgm   P5 grayscale: 0=occupied (black), 254=free (white), 205=unknown
    warehouse.yaml  resolution / origin / thresholds

It boots the SAME native-SimulationApp bootstrap as scripts/11 (the omap Generator
is a native isaacsim component), loads the warehouse, then raycasts a 2D slice at
dog-torso height with isaacsim.asset.gen.omap and rasterizes the returned world
points into the grid. No SLAM, no sensors — the warehouse is static and known, so
we read its collision geometry directly.

Run (needs the venv + libgomp preload, see memory/env-activation):
    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
        python scripts/_warehouse_occupancy.py [--cell 0.05] [--z 0.30]
"""

import argparse
import os
import struct
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Bake the warehouse ROS occupancy map")
parser.add_argument("--cell", type=float, default=0.05, help="Cell size / resolution (m).")
parser.add_argument("--z-nav", type=float, default=0.30, help="Reachability slice for free space (m, dog torso).")
parser.add_argument("--z-min", type=float, default=0.10, help="Lowest obstacle slice (m).")
parser.add_argument("--z-max", type=float, default=2.20, help="Highest obstacle slice (m).")
parser.add_argument("--z-step", type=float, default=0.30, help="Vertical spacing between obstacle slices (m).")
parser.add_argument("--min-x", type=float, default=-13.0)
parser.add_argument("--min-y", type=float, default=-19.0)
parser.add_argument("--max-x", type=float, default=13.0)
parser.add_argument("--max-y", type=float, default=22.0)
parser.add_argument("--out", default="tools/ros/maps", help="Output dir for warehouse.pgm/.yaml.")
parser.add_argument("--name", default="warehouse", help="Map basename.")
args_cli = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

simulation_app = SimulationApp({"headless": True})

# Enable the native extensions the base app does not auto-load: the core.api World
# (extsDeprecated) and the occupancy-map generator. (Mirrors scripts/11 bootstrap.)
import omni.kit.app  # noqa: E402
from omni.ext import ExtensionPathType  # noqa: E402

import isaacsim  # noqa: E402

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_ext_mgr.add_path(
    os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"),
    ExtensionPathType.COLLECTION,
)
for _ext in ("isaacsim.core.api", "isaacsim.asset.gen.omap"):
    _ext_mgr.set_extension_enabled_immediate(_ext, True)
simulation_app.update()

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.asset.gen.omap.bindings import _omap  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402


def _write_pgm(path: str, grid: np.ndarray) -> None:
    """Write an 8-bit binary (P5) PGM. ``grid`` is uint8 (h, w), row 0 = top."""
    h, w = grid.shape
    with open(path, "wb") as f:
        f.write(b"P5\n")
        f.write(f"{w} {h}\n255\n".encode("ascii"))
        f.write(grid.astype(np.uint8).tobytes())


def _write_yaml(path: str, pgm_name: str, cell: float, origin_xy: tuple[float, float]) -> None:
    """Standard ROS map_server YAML. origin = world coords of the bottom-left pixel."""
    with open(path, "w") as f:
        f.write(f"image: {pgm_name}\n")
        f.write("mode: trinary\n")
        f.write(f"resolution: {cell}\n")
        f.write(f"origin: [{origin_xy[0]:.4f}, {origin_xy[1]:.4f}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")


def main() -> int:
    world = World(physics_dt=0.005, rendering_dt=0.02, stage_units_in_meters=1.0)

    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Isaac assets root not found")
        return 1
    stage_utils.add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
        path="/World/Warehouse",
    )
    world.reset()
    # Let collision cooking settle so PhysX has the warehouse meshes before raycasting.
    for _ in range(10):
        world.step(render=False)

    physx = omni.physx.get_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    gen = _omap.Generator(physx, stage_id)
    # values: occupied=1.0, unoccupied=0.0, unknown=0.5 (we rasterize from world
    # positions below, so the exact values only matter for the buffer path).
    gen.update_settings(args_cli.cell, 1.0, 0.0, 0.5)

    cell = args_cli.cell
    min_x, min_y, max_x, max_y = args_cli.min_x, args_cli.min_y, args_cli.max_x, args_cli.max_y
    lower = (min_x, min_y, 0.0)
    upper = (max_x, max_y, 0.0)

    def _scan(z: float) -> tuple[list, list]:
        # 2D slice at z = origin.z; min/max are relative to origin (NVIDIA standalone
        # example uses (-x,-y,0)/(x,y,0) → a horizontal plane at the origin height).
        gen.set_transform((0.0, 0.0, float(z)), lower, upper)
        gen.generate2d()
        return gen.get_occupied_positions(), gen.get_free_positions()

    w = int(round((max_x - min_x) / cell))
    h = int(round((max_y - min_y) / cell))
    grid = np.full((h, w), 205, dtype=np.uint8)  # unknown

    def _stamp(points, value: int) -> None:
        for p in points:
            c = int((p[0] - min_x) / cell)
            r = int((max_y - p[1]) / cell)  # row 0 = top = max_y
            if 0 <= r < h and 0 <= c < w:
                grid[r, c] = value

    # Free space comes from the nav-height reachable set (where a dog actually walks).
    _, free = _scan(args_cli.z_nav)
    _stamp(free, 254)

    # Obstacles = union of occupied cells across a stack of heights, so a whole shelf
    # (deck, boxes, uprights) projects to a solid footprint — not just the legs the
    # single nav slice would catch. Occupied overrides free.
    n = max(1, int(round((args_cli.z_max - args_cli.z_min) / args_cli.z_step)) + 1)
    total_occ = 0
    for i in range(n):
        z = args_cli.z_min + i * args_cli.z_step
        occ, _ = _scan(z)
        total_occ += len(occ)
        _stamp(occ, 0)
    print(f"[omap] scanned {n} obstacle slices z={args_cli.z_min}..{args_cli.z_max}; "
          f"raw occupied hits={total_occ}", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args_cli.out)
    os.makedirs(out_dir, exist_ok=True)
    pgm_name = f"{args_cli.name}.pgm"
    pgm_path = os.path.join(out_dir, pgm_name)
    yaml_path = os.path.join(out_dir, f"{args_cli.name}.yaml")
    _write_pgm(pgm_path, grid)
    _write_yaml(yaml_path, pgm_name, cell, (min_x, min_y))

    occ = int((grid == 0).sum())
    fre = int((grid == 254).sum())
    unk = int((grid == 205).sum())
    print(
        f"[omap] wrote {pgm_path} ({w}x{h} @ {cell}m) "
        f"occupied={occ} free={fre} unknown={unk}",
        flush=True,
    )
    print(f"[omap] wrote {yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(rc)
