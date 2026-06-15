# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Slide an already-spawned worker prop back and forth along a path (DAM demos).

Skeletal animation of People characters is impossible on this Isaac Sim build
(the anim/skeleton stack is ABI-broken and forces a T-pose). We accept the
T-pose and just *move* the character: the demo spawns the worker USD via the
scene config (so it reliably appears), and this class drives that prim's
translate + orient ops along a timed straight path — typically ALONG a red
forbidden band (constant y, sweeping x) so the un-guarded RAW car drives into it
during its opening surge while the DAM car is clamped short.

The worker walks back and forth (ping-pong) so it keeps moving, and it is
oriented to face its travel direction (use ``yaw_offset_deg`` to correct the
model's front). No skeleton binding, no xform-order clearing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

WORKER_USD = (
    f"{ISAAC_NUCLEUS_DIR}/People/Characters/"
    "original_male_adult_construction_05/male_adult_construction_05.usd"
)


@dataclass
class WorkerPath:
    """A timed back-and-forth slide in the world XY plane (meters / sim steps).

    Holds at ``start_xy`` until ``enter_step``; one one-way traverse takes
    ``walk_steps``; with ``loop`` it ping-pongs start<->end forever. Tune
    ``enter_step`` / ``walk_steps`` so the worker is at the crash point when the
    RAW car surges (demo 9 breaches ~step 54-200).
    """

    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    enter_step: int = 40
    walk_steps: int = 320
    z: float = 0.0
    loop: bool = True
    yaw_offset_deg: float = 0.0   # correct the model's "front" axis
    bob_amplitude: float = 0.0    # vertical bounce (0 = pure slide, no "hop")
    bob_period: float = 18.0


class SlidingWorker:
    """Drives an already-spawned prim at ``prim_path`` along a ``WorkerPath``."""

    def __init__(self, prim_path: str, path: WorkerPath) -> None:
        self.prim_path = prim_path
        self.path = path
        self._t_op = None
        self._r_op = None

    def build(self) -> bool:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim or not prim.IsValid():
            print(f"[SlidingWorker] prim not found at {self.prim_path}")
            return False

        # Reuse the translate/orient ops created by the asset's init_state (do
        # NOT clear the op order — that wipes the model's up-axis transform).
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        prec = UsdGeom.XformOp.PrecisionDouble
        self._t_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        self._r_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
        if self._t_op is None:
            self._t_op = xf.AddTranslateOp(prec)
        if self._r_op is None:
            self._r_op = xf.AddOrientOp(prec)
        self._apply(0)
        return True

    def update(self, step: int) -> None:
        if self._t_op is not None:
            self._apply(step)

    def _apply(self, step: int) -> None:
        from pxr import Gf

        p = self.path
        if p.loop:
            # Always moving from step 0; enter_step is a phase offset (shifts
            # where in the back-and-forth cycle the worker is, to align the crash).
            raw = (step + p.enter_step) / float(p.walk_steps)
            tri = raw % 2.0
            u, forward = (tri, True) if tri <= 1.0 else (2.0 - tri, False)
            moving = True
        elif step <= p.enter_step:
            u, forward, moving = 0.0, True, False
        else:
            raw = (step - p.enter_step) / float(p.walk_steps)
            u, forward, moving = min(raw, 1.0), True, raw < 1.0

        sx, sy = p.start_xy
        ex, ey = p.end_xy
        x = sx + (ex - sx) * u
        y = sy + (ey - sy) * u
        z = p.z
        if moving and p.bob_amplitude:
            z += abs(math.sin(2.0 * math.pi * step / p.bob_period)) * p.bob_amplitude
        self._t_op.Set(Gf.Vec3d(float(x), float(y), float(z)))

        # Face travel direction (yaw about world Z), flipping on the return leg.
        dx, dy = (ex - sx, ey - sy) if forward else (sx - ex, sy - ey)
        yaw = math.atan2(dy, dx) + math.radians(p.yaw_offset_deg)
        half = yaw / 2.0
        self._r_op.Set(Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half)))


# Back-compat alias.
WalkingWorker = SlidingWorker
