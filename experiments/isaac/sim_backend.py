"""Isaac Sim execution backend for the thesis experiments (E2/E3.x/E4.x).

Drop-in replacement for the kinematic integrator: the SAME control pipeline as
``common.kinesim.KineSim`` (nominal P-controller -> safety filter at 50 Hz,
identical observation noise/delay knobs, identical metrics recorder), but the
safe command is EXECUTED by a real Go2 quadruped in Isaac Sim — the shipped
``Go2FlatTerrainPolicy`` RL policy tracks ``[vx, vy, omega]`` at 200 Hz physics
— and poses are measured back from the simulator.

Throughput comes from ARENA TILING: one physics world hosts K spatially
separated arenas (ARENA_SPACING apart along +x), each running an independent
episode; ``world.step`` cost is amortised across all of them. Episodes in an
arena are expressed in scenario coordinates — the arena offset is added at
teleport time and subtracted at measurement time, so the filter, metrics and
scenario code never see it.

What stays identical to kinesim (inherited, not copied):
    nominal():                  waypoint advance + body-frame P-control
    _snapshot_observations():   obs delay buffer + per-step gaussian noise
    _neighbors_of():            neighbour tuples incl. static wall culling

What changes (all deliberate, part of the embodiment result):
    execution        first-order integration -> Go2 RL policy + PhysX
    walls            metric point-chains stay for the filter/metrics, but ALSO
                     become real static colliders
    exec_noise_std   injected on the COMMANDED world velocity (kinesim: the
                     executed one; the policy tracker sits in between)
    falls            fallen dogs (tilt-only detection — the go2 asset's root z
                     reads ~0.22 standing / ~0.10 mid-gait, so height
                     thresholds false-positive) hold a zero command; rows gain
                     n_falls / diverged columns

Import this module only AFTER isaacsim.SimulationApp is running.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))
_TOOLS_DIR = _EXP_DIR.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from common.kinesim import KineSim, SimConfig, StepRecord, wrap_angle  # noqa: E402

PHYSICS_DT = 0.005          # 200 Hz, matches Go2 policy training (demo11/smoke)
STAND_Z = 0.5               # spawn height; policy settles to stand from here
SETTLE_S = 1.0              # zero-command settle after each teleport reset
ARENA_SPACING = 40.0        # arena a is offset (a * ARENA_SPACING, 0)
PARK_ORIGIN = (0.0, 60.0)   # idle dogs wait far north of the arena band
PARK_SPACING = 4.0
FALL_TILT = 0.6             # body-z . world-z below this => fallen (~53 deg)
WALL_H = 0.5                # physical wall box height (m)
WALL_T = 0.12               # physical wall box thickness (m)


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]  # (w, x, y, z)


class BatchedGo2Runtime:
    """Batched replacement for N per-dog ``Go2FlatTerrainPolicy.forward`` calls.

    The per-dog path costs ~1 ms/dog/physics-step (measured): every dog does
    its own warp->torch GPU sync reads for the observation plus its own policy
    inference. All dogs share the SAME policy weights, so one multi-prim
    ``Articulation`` view + one ``(N, 48)`` jit call replaces all of it — the
    per-step cost becomes ~constant in N. Observation layout mirrors
    ``go2.py::_compute_observation`` exactly (verified against the shipped
    source); dt/decimation semantics are identical (single global counter, all
    dogs infer on the same tick).
    """

    def __init__(self, dogs: list) -> None:
        import torch
        import warp as wp
        from isaacsim.core.experimental.prims import Articulation

        self._torch = torch
        self._wp = wp
        proto = dogs[0].policy                    # weights/params shared by all
        self.view = Articulation(paths=[d.policy._prim_path for d in dogs])
        self.n = len(dogs)
        self.policy = proto.policy                # torch.jit module (batch-safe MLP)
        self.decimation = int(proto._decimation)
        self.action_scale = float(proto._action_scale)
        self.device = torch.device(str(proto.robot._device))
        self.default_pos = proto.default_pos.to(self.device).reshape(1, -1)
        self.prev_action = torch.zeros(self.n, 12, device=self.device)
        self.counter = 0
        self._gz = torch.tensor([0.0, 0.0, -1.0], device=self.device)

    def reset_slots(self, slots: list[int]) -> None:
        """Clear policy memory for teleported dogs (obs[36:48] of a fresh
        episode must start at zero, same as a fresh per-dog wrapper)."""
        for s in slots:
            self.prev_action[s].zero_()

    def _quat_rot(self, q):
        """(N, 4) wxyz -> (N, 3, 3) body-to-world rotation matrices."""
        t = self._torch
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return t.stack([
            t.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=1),
            t.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=1),
            t.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=1),
        ], dim=1)

    def step(self, cmds) -> None:
        """Call once per physics step BEFORE world.step (same contract as the
        per-dog ``policy.forward``); ``cmds`` is the flat [(vx, vy, om)] list."""
        if self.counter % self.decimation == 0:
            t, wp = self._torch, self._wp
            lin_w, ang_w = self.view.get_velocities()
            pos_w, quat_w = self.view.get_world_poses()
            dofp = wp.to_torch(self.view.get_dof_positions())
            dofv = wp.to_torch(self.view.get_dof_velocities())
            R = self._quat_rot(wp.to_torch(quat_w))            # (N, 3, 3) body->world
            lin_b = t.einsum("nij,ni->nj", R, wp.to_torch(lin_w))
            ang_b = t.einsum("nij,ni->nj", R, wp.to_torch(ang_w))
            grav_b = -R[:, 2, :]                               # R^T @ [0,0,-1]
            cmd = t.tensor(cmds, dtype=t.float32, device=self.device)
            obs = t.cat([lin_b, ang_b, grav_b, cmd,
                         dofp - self.default_pos, dofv,
                         self.prev_action], dim=1)             # (N, 48)
            with t.no_grad():
                act = self.policy(obs).detach().reshape(self.n, -1)
            self.prev_action = act.clone()
            targets = self.default_pos + act * self.action_scale
            self.view.set_dof_position_targets(wp.from_torch(targets.contiguous()))
        self.counter += 1


class Go2Pool:
    """K arenas x S dog slots in ONE world, plus per-scenario physical walls.

    Spawning an articulation needs a world.reset() (seconds), so the pool only
    grows. Wall layout is GLOBAL (replicated per arena) and changing it also
    needs a world.reset() — the runner must only call ``configure`` while no
    episode is active (barrier), then teleport-reset each arena it fills.
    """

    def __init__(self) -> None:
        from isaacsim.core.api import World

        self.world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT * 4,
                           stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self._apply_floor_friction("/World/defaultGroundPlane")
        self.dogs: list = []            # flat pool, slot = arena * S + i
        self.K = 1                      # arenas
        self.S = 0                      # slots per arena
        self.runtime: BatchedGo2Runtime | None = None
        self._pose_cache: list[tuple] | None = None
        self._wall_sig: tuple | None = ()
        self._wall_paths: list[str] = []
        self._needs_reset = False

    # -- scene management ---------------------------------------------------

    @staticmethod
    def _apply_floor_friction(root_path: str) -> None:
        """friction=1.0 physics material bind (the Go2 policy trained on it;
        PhysX's default ~0.5 makes the dogs belly-slide — see demo11)."""
        import omni.usd
        from pxr import Sdf, UsdPhysics, UsdShade

        stage = omni.usd.get_context().get_stage()
        mat = UsdShade.Material.Define(stage, Sdf.Path("/World/PhysicsMaterials/Floor"))
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr().Set(1.0)
        pm.CreateDynamicFrictionAttr().Set(1.0)
        pm.CreateRestitutionAttr().Set(0.0)
        prim = stage.GetPrimAtPath(root_path)
        if prim.IsValid():
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                         materialPurpose="physics")

    def offset(self, arena: int) -> tuple[float, float]:
        return (arena * ARENA_SPACING, 0.0)

    def slot(self, arena: int, i: int) -> int:
        return arena * self.S + i

    def _park_xy(self, slot: int) -> tuple[float, float]:
        return (PARK_ORIGIN[0] + PARK_SPACING * (slot % 8),
                PARK_ORIGIN[1] + PARK_SPACING * (slot // 8))

    @staticmethod
    def _wall_chains(wall_specs: list) -> list[tuple]:
        """Group the static point-agents back into straight chains by aid prefix
        (scenarios build walls as ``{prefix}{i}`` runs of collinear points)."""
        chains: dict[str, list] = {}
        for s in wall_specs:
            prefix = s.aid.rstrip("0123456789")
            chains.setdefault(prefix, []).append(s)
        out = []
        for prefix, pts in chains.items():
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
            length = math.hypot(x1 - x0, y1 - y0)
            out.append((prefix, (x0 + x1) / 2, (y0 + y1) / 2,
                        length, math.atan2(y1 - y0, x1 - x0)))
        return out

    def _set_walls(self, wall_specs: list) -> None:
        """(Re)build wall boxes, replicated in every arena, on layout change."""
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        sig = (self.K, tuple(sorted((round(s.x, 3), round(s.y, 3)) for s in wall_specs)))
        if sig == self._wall_sig:
            return
        stage = omni.usd.get_context().get_stage()
        for path in self._wall_paths:
            stage.RemovePrim(path)
        self._wall_paths = []
        for arena in range(self.K):
            ox, oy = self.offset(arena)
            for prefix, cx, cy, length, yaw in self._wall_chains(wall_specs):
                path = f"/World/Walls/A{arena}_{prefix}"
                cube = UsdGeom.Cube.Define(stage, path)
                cube.GetSizeAttr().Set(1.0)
                xf = UsdGeom.Xformable(cube.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(cx + ox, cy + oy, WALL_H / 2))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, math.degrees(yaw)))
                # span the chain plus the metric point spacing margin
                xf.AddScaleOp().Set(Gf.Vec3f(length + 0.25, WALL_T, WALL_H))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                self._wall_paths.append(path)
        self._wall_sig = sig
        self._needs_reset = True

    def configure(self, arenas: int, slots: int, wall_specs: list) -> None:
        """Barrier-only: make the world host ``arenas`` x ``slots`` dogs with the
        given wall layout. Must not be called while an episode is active —
        world.reset() would snap every dog back to its default state."""
        from controll_scripts.squad.locomotion import Go2Locomotion

        self.K, self.S = arenas, slots
        while len(self.dogs) < arenas * slots:
            i = len(self.dogs)
            px, py = self._park_xy(i)
            self.dogs.append(Go2Locomotion(
                prim_path=f"/World/Go2_{i}", position=[px, py, STAND_Z]))
            self._needs_reset = True
        self._set_walls(wall_specs)
        if self._needs_reset:
            self.world.reset()
            for dog in self.dogs:
                dog.initialize()
            self.runtime = BatchedGo2Runtime(self.dogs)
            self._needs_reset = False

    def teleport_arena(self, arena: int, starts: list[tuple]) -> None:
        """Place this arena's dogs at ``starts`` [(x, y, yaw)] (scenario frame),
        park its unused slots, and clear each policy's internal memory. The
        post-teleport settle is driven by the runner (zero-command ticks)."""
        ox, oy = self.offset(arena)
        for i in range(self.S):
            dog = self.dogs[self.slot(arena, i)]
            if i < len(starts):
                x, y, yaw = starts[i]
                x, y = x + ox, y + oy
            else:
                (x, y), yaw = self._park_xy(self.slot(arena, i)), 0.0
            robot = dog.policy.robot
            robot.reset_to_default_state()          # default joints + zero velocities
            robot.set_world_poses([[x, y, STAND_Z]], [_yaw_quat(yaw)])
        self.runtime.reset_slots([self.slot(arena, i) for i in range(self.S)])

    _prof = {"apply_s": 0.0, "world_s": 0.0, "steps": 0}

    def step_physics(self, cmds: list[tuple]) -> None:
        """One 200 Hz physics step; ``cmds`` is the FLAT per-slot command list."""
        t0 = time.perf_counter()
        self.runtime.step(cmds)
        t1 = time.perf_counter()
        self.world.step(render=False)
        self._pose_cache = None
        p = self._prof
        p["apply_s"] += t1 - t0
        p["world_s"] += time.perf_counter() - t1
        p["steps"] += 1
        if p["steps"] % 2000 == 0:
            n = max(1, len(self.dogs))
            print(f"[prof] {p['steps']} phys steps: policy {p['apply_s']:.1f}s "
                  f"({1e3 * p['apply_s'] / p['steps']:.2f} ms/step, "
                  f"{1e3 * p['apply_s'] / p['steps'] / n:.3f} ms/dog) | "
                  f"world.step {p['world_s']:.1f}s "
                  f"({1e3 * p['world_s'] / p['steps']:.2f} ms/step)", flush=True)

    def zero_cmds(self) -> list[tuple]:
        return [(0.0, 0.0, 0.0)] * len(self.dogs)

    def _refresh_poses(self) -> None:
        """ONE batched pose read for the whole pool (per physics-settled tick),
        instead of a per-dog GPU sync per control step."""
        import warp as wp

        pos, quat = self.runtime.view.get_world_poses()
        p = wp.to_torch(pos).cpu()
        q = wp.to_torch(quat).cpu()  # (w, x, y, z)
        cache = []
        for s in range(len(self.dogs)):
            w, x, y, z = (float(q[s, 0]), float(q[s, 1]), float(q[s, 2]), float(q[s, 3]))
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            upright = 1.0 - 2.0 * (x * x + y * y)   # R[2][2]
            cache.append((float(p[s, 0]), float(p[s, 1]), yaw, upright))
        self._pose_cache = cache

    def get_state(self, arena: int, i: int) -> tuple[float, float, float, float]:
        """(x, y, yaw, upright) for arena-local dog ``i`` in scenario frame."""
        if self._pose_cache is None:
            self._refresh_poses()
        x, y, yaw, upright = self._pose_cache[self.slot(arena, i)]
        ox, oy = self.offset(arena)
        return (x - ox, y - oy, yaw, upright)


class IsaacArenaSim(KineSim):
    """One episode in one arena, driven tick-by-tick by the suite runner.

    Same inheritance trick as before (nominal/observations/neighbours straight
    from KineSim), but the loop is externalised so K arenas can interleave on
    one physics world:

        sim = IsaacArenaSim(...)        # teleports its dogs
        <runner settles with zero cmds> # SETTLE_S of shared physics time
        sim.begin()                     # first measured sync
        each control tick:
            sim.control_tick(cmds)      # filter -> writes its slots in cmds
            <runner steps physics x4>
            sim.post_step(recorder)     # measure back; True when episode over
    """

    def __init__(self, specs: list, safety_filter, cfg: SimConfig | None,
                 rng_seed: int, pool: Go2Pool, arena: int):
        super().__init__(specs, safety_filter, cfg, rng_seed=rng_seed)
        self.pool = pool
        self.arena = arena
        self.fallen: set[str] = set()
        self.diverged = False
        self.finished = False
        self._k = 0
        self._max_steps = int(round(self.cfg.max_time / self.cfg.dt))
        self._rec: StepRecord | None = None
        self._dyn_ids = [aid for aid, ag in self.agents.items() if not ag.spec.static]
        # E1.3 actuation delay (mirrors kinesim): FIFO of filtered commands per
        # dog; the executed command is cmd_delay_steps control ticks old.
        from collections import deque as _deque
        self._cmd_q = {aid: _deque([(0.0, 0.0, 0.0)] * self.cfg.cmd_delay_steps,
                                   maxlen=self.cfg.cmd_delay_steps + 1)
                       for aid in self._dyn_ids} if self.cfg.cmd_delay_steps > 0 else None
        if len(self._dyn_ids) > pool.S:
            raise ValueError(f"scenario needs {len(self._dyn_ids)} dogs but arena "
                             f"slots={pool.S}; configure() the pool first")
        pool.teleport_arena(arena, [(ag.x, ag.y, ag.yaw) for ag in
                                    (self.agents[a] for a in self._dyn_ids)])

    @property
    def t(self) -> float:
        return self._k * self.cfg.dt

    def begin(self) -> None:
        self._sync_measured(first=True)

    def _sync_measured(self, first: bool = False) -> None:
        """Pull measured poses into the AgentState fields, so the inherited
        observation/nominal/metrics code sees Isaac ground truth."""
        c = self.cfg
        for i, aid in enumerate(self._dyn_ids):
            ag = self.agents[aid]
            x, y, yaw, upright = self.pool.get_state(self.arena, i)
            if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 1e3 or abs(y) > 1e3:
                self.diverged = True
                continue
            if first:
                ag.wvx = ag.wvy = 0.0
            else:
                ag.wvx = (x - ag.x) / c.dt
                ag.wvy = (y - ag.y) / c.dt
                ag.path_len += math.hypot(x - ag.x, y - ag.y)
            ag.x, ag.y, ag.yaw = x, y, wrap_angle(yaw)
            if upright < FALL_TILT:
                self.fallen.add(aid)

    def control_tick(self, cmds: list[tuple]) -> None:
        """Nominal -> filter for every dog; write held commands into the flat
        per-slot buffer. Mirrors KineSim.run()'s decision half exactly."""
        c = self.cfg
        self._snapshot_observations()
        rec = StepRecord(t=self.t, poses={}, raw_cmds={}, safe_cmds={}, filter_s={})
        for i, aid in enumerate(self._dyn_ids):
            ag = self.agents[aid]
            if aid in self.fallen:
                rec.raw_cmds[aid] = rec.safe_cmds[aid] = (0.0, 0.0, 0.0)
                rec.filter_s[aid] = 0.0
                continue
            raw = self.nominal(ag)
            t0 = time.perf_counter()
            safe, info = self.filter.filter(
                aid, raw, (ag.x, ag.y, ag.yaw),
                self._neighbors_of(aid), self_priority=ag.spec.priority,
            )
            rec.filter_s[aid] = time.perf_counter() - t0
            rec.raw_cmds[aid] = raw
            rec.safe_cmds[aid] = safe
            rec.decisions[aid] = info
            if self._cmd_q is not None:
                q = self._cmd_q[aid]
                q.append(safe)
                safe = q.popleft()
            vx = max(-c.vmax, min(c.vmax, safe[0]))
            vy = max(-c.vmax, min(c.vmax, safe[1]))
            om = max(-c.wmax, min(c.wmax, safe[2]))
            if c.exec_noise_std > 0.0:
                # kinesim injects on the EXECUTED world velocity; with a real
                # tracker in the loop the closest equivalent is the COMMANDED one.
                cs, sn = math.cos(ag.yaw), math.sin(ag.yaw)
                wvx = vx * cs - vy * sn + self._rng.gauss(0.0, c.exec_noise_std)
                wvy = vx * sn + vy * cs + self._rng.gauss(0.0, c.exec_noise_std)
                vx = wvx * cs + wvy * sn
                vy = -wvx * sn + wvy * cs
            cmds[self.pool.slot(self.arena, i)] = (vx, vy, om)
        self._rec = rec

    def post_step(self, recorder=None) -> bool:
        """Measure back after the physics substeps; True when the episode ends."""
        c = self.cfg
        rec = self._rec
        self._sync_measured()
        for aid in self._dyn_ids:
            ag = self.agents[aid]
            if not ag.done and math.hypot(ag.spec.gx - ag.x, ag.spec.gy - ag.y) < c.goal_tol:
                ag.done, ag.done_time = True, self.t
        for aid, ag in self.agents.items():
            rec.poses[aid] = (ag.x, ag.y, ag.yaw)
        if recorder is not None:
            recorder.step(self, rec)
        self._k += 1
        self.finished = (self.diverged
                         or self._k >= self._max_steps
                         or all(ag.done for ag in self.agents.values()
                                if not ag.spec.static))
        return self.finished
