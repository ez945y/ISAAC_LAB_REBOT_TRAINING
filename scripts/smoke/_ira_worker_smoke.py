# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I001

"""Isolation smoke-test: one walking warehouse worker via Replicator Agent (IRA).

`omni.anim.people` is gone in Isaac 6.0 / Kit 110; the people system is now
`isaacsim.replicator.agent` (IRA), which IS installed. This proves a single
construction worker walks a path on THIS machine, before we fold the character
setup into demo 9. Run it on the WebRTC livestream and watch.

    python scripts/smoke/_ira_worker_smoke.py --livestream 2

Once the worker walks here, the same `character` config block ports into demo 9
via CharacterLoader (against demo 9's existing stage instead of a loaded env).

⚠️ First-run unknowns to watch / report back: (1) does setup_simulation finish
(it bakes a nav mesh + applies the anim-graph rig), (2) do the legs animate,
(3) does the worker actually follow the patrol points. Each is isolated so we
fix one thing at a time.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IRA walking-worker smoke test")
parser.add_argument("--steps", type=int, default=1200, help="App update steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from livestream.livestream_support import apply_livestream_defaults  # noqa: E402

apply_livestream_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac/Omniverse imports only after the app is up ────────────────────────
import asyncio  # noqa: E402

import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402


def enable_extension(ext_id: str) -> None:
    """Namespace-independent enable (isaacsim.core.utils is deprecated here)."""
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate(ext_id, True)


def _register_deprecated_exts() -> None:
    """IRA -> sensors.rtx.placement -> sensors.camera, but sensors.camera lives
    in `extsDeprecated` (off the default search path). Register that collection
    so the dependency resolves."""
    import isaacsim
    from omni.ext import ExtensionPathType

    dep = os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated")
    omni.kit.app.get_app().get_extension_manager().add_path(dep, ExtensionPathType.COLLECTION)


# IRA (Replicator Agent) is the Kit-110 people system; enable it (+ its deps).
_register_deprecated_exts()
simulation_app.update()

# Nav-mesh schema MUST be registered before IRA's setup_simulation() creates the
# NavMeshVolume prims — otherwise NavSchemaNavMeshAreaAPI is unregistered, the
# `nav:area` attribute has an empty typeName, and navmesh baking fails.
for _ext in (
    "omni.anim.navigation.schema",
    "omni.anim.navigation.core",
    "omni.anim.navigation.bundle",
):
    enable_extension(_ext)
simulation_app.update()


def _register_nav_usd_schema() -> None:
    """Enabling the Kit ext does NOT register the codeless USD schema into the
    UsdSchemaRegistry on this build, so `NavSchemaNavMeshAreaAPI` stays unknown
    and navmesh baking fails. Register the schema's plugInfo.json explicitly."""
    import glob

    import isaacsim
    from pxr import Plug

    base = os.path.dirname(isaacsim.__file__)
    hits = glob.glob(
        os.path.join(base, "extscache", "omni.anim.navigation.schema-*", "**", "plugInfo.json"),
        recursive=True,
    )
    for p in hits:
        Plug.Registry().RegisterPlugins(os.path.dirname(p))
    print(f"[ira] registered {len(hits)} nav schema plugInfo(s)", flush=True)


_register_nav_usd_schema()
simulation_app.update()

enable_extension("isaacsim.replicator.agent.core")
simulation_app.update()

from isaacsim.replicator.agent.core.api import set_config, setup_simulation  # noqa: E402
from isaacsim.replicator.agent.core.configuration.models.root import RootConfig  # noqa: E402

WAREHOUSE = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
WORKER = f"{ISAAC_NUCLEUS_DIR}/People/Characters/original_male_adult_construction_05/male_adult_construction_05.usd"


def build_config() -> RootConfig:
    # Mirror data/test_configs/test_data_gen.yaml, trimmed to one walking worker.
    return RootConfig.model_validate(
        {
            "isaacsim.replicator.agent": {
                "version": "1.6.0",
                "simulation_duration": 60.0,
                "environment": {"base_stage_asset_path": WAREHOUSE},
                "character": {
                    "root_prim_path": "/World/Characters",
                    "groups": {
                        "warehouse_workers": {
                            "asset_path": WORKER,
                            "num": 1,
                            "routines": [
                                {
                                    "patrol": {
                                        "speed_range": [1.0, 1.4],
                                        # Walk back and forth along a short line.
                                        "path_points": [
                                            [2.0, -3.0, 0.0],
                                            [2.0, 3.0, 0.0],
                                        ],
                                    }
                                }
                            ],
                        }
                    },
                },
            }
        }
    )


def _run_async(coro) -> None:
    """Drive a Kit coroutine to completion while pumping app updates."""
    task = asyncio.ensure_future(coro)
    while not task.done():
        simulation_app.update()
    task.result()  # surface exceptions


def main() -> None:
    print("[ira] building config + running setup_simulation() ...", flush=True)
    set_config(build_config())
    _run_async(setup_simulation())
    print("[ira] setup complete; starting timeline and stepping.", flush=True)

    omni.timeline.get_timeline_interface().play()
    for i in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        simulation_app.update()
        if i % 120 == 0:
            print(f"[ira] step {i}", flush=True)
    print("[ira] done; keeping window alive (Ctrl+C to exit).", flush=True)
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
