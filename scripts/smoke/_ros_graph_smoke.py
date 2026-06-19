# ruff: noqa: I001
"""Minimal test: does OmniGraph ROS2-bridge graph creation work under the NATIVE
isaacsim.SimulationApp? (It fails under Isaac Lab's AppLauncher.)

Boots native -> World -> builds a tiny graph (OnPlaybackTick -> ROS2PublishClock)
-> steps. If '[smoke] graph built' prints and no OmniGraphError, the native route
is validated and we can port demo 10 to it. Check '/clock' from another shell:
    source /opt/ros/jazzy/setup.bash && ros2 topic echo /clock --once
"""

import os
import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))

import omni.kit.app  # noqa: E402
from omni.ext import ExtensionPathType  # noqa: E402
import isaacsim  # noqa: E402

_mgr = omni.kit.app.get_app().get_extension_manager()
_mgr.add_path(os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated"), ExtensionPathType.COLLECTION)
for _ext in (
    "isaacsim.core.api",
    "omni.graph.action",
    "omni.graph.nodes",
    "isaacsim.core.nodes",
    "isaacsim.ros2.bridge",
):
    _mgr.set_extension_enabled_immediate(_ext, True)
simulation_app.update()

import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import World  # noqa: E402


def main() -> None:
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    keys = og.Controller.Keys
    try:
        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PubClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick", "PubClock.inputs:execIn"),
                    ("ReadTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
                    ("Context.outputs:context", "PubClock.inputs:context"),
                ],
                keys.SET_VALUES: [("PubClock.inputs:topicName", "/clock")],
            },
        )
        print("[smoke] graph built OK", flush=True)
    except Exception as exc:
        print(f"[smoke] graph build FAILED: {exc}", flush=True)
        raise

    world.reset()
    for i in range(180):
        world.step(render=False)
        if i % 60 == 0:
            print(f"[smoke] step {i} (publishing /clock)", flush=True)
    print("[smoke] DONE — native OmniGraph ROS2 works", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
