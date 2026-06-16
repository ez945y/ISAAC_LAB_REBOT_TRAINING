# ruff: noqa: I001
"""Fast probe: which Go2 policy assets exist + their control rates. No World/stepping."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
print("[probe] SimulationApp up", flush=True)

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.robot.policy.examples", True
)
simulation_app.update()

import omni.client  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

root = get_assets_root_path()
print(f"[probe] assets_root = {root}", flush=True)

base = f"{root}/Isaac/Samples/Policies/go2"
for name in ("newton_policy.pt", "newton_env.yaml", "physx_policy.pt", "physx_env.yaml"):
    url = f"{base}/{name}"
    res, entry = omni.client.stat(url)
    ok = str(res) == "Result.OK"
    size = getattr(entry, "size", "?") if ok else "-"
    print(f"[probe] {'EXISTS' if ok else 'MISSING':7} {name:18} size={size} ({res})", flush=True)

# Dump control properties from any env yaml that exists.
try:
    from isaacsim.robot.policy.examples.controllers.config_loader import (  # noqa: E402
        get_physics_properties,
        parse_env_config,
    )

    for name in ("newton_env.yaml", "physx_env.yaml"):
        url = f"{base}/{name}"
        if str(omni.client.stat(url)[0]) != "Result.OK":
            continue
        params = parse_env_config(url)
        dec, dt, render = get_physics_properties(params)
        print(f"[probe] {name}: decimation={dec} dt={dt} render_interval={render} "
              f"action_scale={params.get('action_scale')}", flush=True)
except Exception as e:
    print(f"[probe] config dump failed: {e}", flush=True)

simulation_app.close()
