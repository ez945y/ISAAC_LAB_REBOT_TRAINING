# Minimal WebRTC livestream keep-alive for debugging.
#
# Strips out all business logic: it only launches the Isaac Lab app (so
# --livestream works), builds a trivial visible scene, and steps forever.
# It NEVER calls close() on its own, so you have unlimited time to connect
# the WebRTC client and confirm whether streaming works at all.
#
# Usage (server):
#   source ~/IsaacLab/env_isaaclab/bin/activate
#   python tools/livestream/livestream_keepalive.py --livestream 2
#
# Then on the Mac client: Server 192.168.90.162, Signal 49100, Stream 47998 -> Connect.
# Stop with Ctrl+C in this terminal (or by closing the client).

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

parser = argparse.ArgumentParser(description="Minimal livestream keep-alive")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Inject publicIp + allowDynamicResize so a remote WebRTC client can connect.
from tools.livestream.livestream_support import apply_livestream_defaults

apply_livestream_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below must be imported AFTER the app is launched ----
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # A camera angle so the client shows something framed, not an edge-on void.
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.5])

    # Ground + light so the viewport is clearly NOT black if streaming works.
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    )
    cube_cfg = sim_utils.CuboidCfg(
        size=(0.6, 0.6, 0.6),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 1.0)),
    )
    cube_cfg.func("/World/Cube", cube_cfg, translation=(0.0, 0.0, 0.3))

    sim.reset()
    print("=" * 70, flush=True)
    print("[keepalive] scene ready. Streaming should be LIVE now.", flush=True)
    print("[keepalive] connect the WebRTC client to 192.168.90.162 (49100/47998).", flush=True)
    print("[keepalive] this will run forever -- Ctrl+C to stop.", flush=True)
    print("=" * 70, flush=True)

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
