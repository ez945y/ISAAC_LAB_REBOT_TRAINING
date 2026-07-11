# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 locomotion backend — the only module bound to the Go2 RL policy.

Wraps Isaac Sim's shipped :class:`Go2FlatTerrainPolicy` (which auto-loads the
pretrained ``Samples/Policies/go2/newton_policy.pt``) behind the
:class:`LocomotionBackend` ABI. Swapping to Spot/ANYmal = a sibling of this file
pointing at ``SpotFlatTerrainPolicy`` etc.; nothing above the ABI changes.

Import this only after the Isaac simulation app is running.
"""

from __future__ import annotations

import math


class Go2Locomotion:
    """LocomotionBackend backed by the Go2 flat-terrain RL policy."""

    def __init__(
        self,
        prim_path: str,
        position: list[float],
        orientation: list[float] | None = None,
        *,
        policy_path: str | None = None,
        env_config_path: str | None = None,
    ) -> None:
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy

        # Isaac Lab's SimulationManager (PhysxManager) lacks get_active_physics_engine,
        # which the policy controller calls to pick the policy + USD physics variant.
        # Isaac Lab runs PhysX, so shim it to report "physx". (Native isaacsim already
        # has this method and also reports "physx" here, so the shim is a no-op there.)
        if not hasattr(SimulationManager, "get_active_physics_engine"):
            SimulationManager.get_active_physics_engine = staticmethod(lambda: "physx")

        # We run PhysX, and the matching physx_policy.pt / physx_env.yaml DO ship
        # (verified via omni.client.stat). Let Go2FlatTerrainPolicy auto-detect them
        # from the active engine — forcing the Newton policy under PhysX produced a
        # broken gait (~0.18 m of drift instead of walking). Explicit paths are still
        # honoured if a caller passes them (e.g. a custom-trained policy).
        self.policy = Go2FlatTerrainPolicy(
            prim_path=prim_path,
            position=position,
            orientation=orientation,
            policy_path=policy_path,
            env_config_path=env_config_path,
        )
        self._device = "cuda"

    def initialize(self) -> None:
        """Call after the world has been reset (sets up the articulation)."""
        self.policy.initialize()
        try:
            self._device = str(self.policy.robot._device)
        except Exception:
            self._device = "cuda"

    # -- LocomotionBackend ABI ------------------------------------------------

    def get_pose(self) -> tuple[float, float, float]:
        import warp as wp

        pos, quat = self.policy.robot.get_world_poses()
        p = wp.to_torch(pos).reshape(-1)
        q = wp.to_torch(quat).reshape(-1)  # (w, x, y, z)
        w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return (float(p[0]), float(p[1]), yaw)

    def apply(self, dt: float, command: tuple[float, float, float]) -> None:
        import torch

        cmd = torch.tensor(command, dtype=torch.float32, device=self._device)
        self.policy.forward(dt, cmd)
