"""Multi-arena scheduler: runs one experiment's sweep on the Isaac backend.

Up to K episodes execute simultaneously in spatially separated arenas of ONE
physics world (see sim_backend.ARENA_SPACING), amortising the per-step
simulation cost. An arena that finishes its episode immediately teleports to
the next queued one (with its own settle window) while the others keep
running — a barrier only happens when the WALL layout changes (different
scenario), because rebuilding walls needs a world.reset().

Episode results checkpoint through common.sweep_io (resume-safe).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import replace
from pathlib import Path

from common.kinesim import SimConfig
from common.metrics import EpisodeRecorder, MetricsConfig
from common.registry import Experiment
from common.scenarios import SCENARIOS
from common.sweep_io import Checkpointer

from sim_backend import PHYSICS_DT, SETTLE_S, Go2Pool, IsaacArenaSim

K_MAX = 4  # arenas; beyond this the per-dog policy cost dominates anyway


def _wall_sig(scen: str, cond) -> tuple:
    specs = SCENARIOS[scen](0, **cond.scenario_kwargs)
    return tuple(sorted((round(s.x, 3), round(s.y, 3)) for s in specs if s.static))


def run_experiment_isaac(exp: Experiment, pool: Go2Pool, out_dir: Path, *,
                         seeds: int | None = None, seed_base: int = 0,
                         max_time: float = 60.0, arenas: int | None = None,
                         slot_cap: int = 24) -> list[dict]:
    ck = Checkpointer(out_dir)
    n_seeds = seeds if seeds is not None else exp.seeds
    S = exp.max_dogs
    K = arenas if arenas is not None else max(1, min(K_MAX, slot_cap // S))

    base_sim = SimConfig(max_time=max_time)
    base_met = MetricsConfig()
    # Ordered jobs, grouped into runs of identical wall layout (barrier between).
    groups: list[tuple[tuple, list, list]] = []   # (sig, wall_specs, jobs)
    total = 0
    for scen in exp.scenarios:
        for cond in exp.conditions:
            sig = _wall_sig(scen, cond)
            s_cfg = (replace(base_sim, **cond.sim_overrides)
                     if cond.sim_overrides else base_sim)
            m_cfg = (replace(base_met, **cond.metrics_overrides)
                     if cond.metrics_overrides else base_met)
            jobs = []
            for i in range(n_seeds):
                seed = seed_base + i
                total += 1
                if not ck.skip(scen, cond.label, seed):
                    jobs.append((scen, cond, seed, s_cfg, m_cfg))
            if not jobs:
                continue
            if groups and groups[-1][0] == sig:
                groups[-1][2].extend(jobs)
            else:
                wall_specs = [s for s in SCENARIOS[scen](0, **cond.scenario_kwargs)
                              if s.static]
                groups.append((sig, wall_specs, jobs))

    settle_ticks = int(round(SETTLE_S / base_sim.dt))
    substeps = int(round(base_sim.dt / PHYSICS_DT))
    k = len(ck.rows)

    for _sig, wall_specs, jobs in groups:
        pool.configure(K, S, wall_specs)          # barrier: no active episodes
        queue = deque(jobs)
        active: dict[int, dict] = {}
        while queue or active:
            for a in range(K):
                if a in active or not queue:
                    continue
                scen, cond, seed, s_cfg, m_cfg = queue.popleft()
                specs = SCENARIOS[scen](seed, **cond.scenario_kwargs)
                filt = cond.make_filter()
                sim = IsaacArenaSim(specs, filt, s_cfg, rng_seed=seed,
                                    pool=pool, arena=a)
                active[a] = {"sim": sim, "filt": filt,
                             "rec": EpisodeRecorder(m_cfg),
                             "settle": settle_ticks,
                             "meta": (scen, cond, seed),
                             "t0": time.perf_counter()}
            cmds = pool.zero_cmds()
            for st in active.values():
                if st["settle"] == 0:
                    st["sim"].control_tick(cmds)
            for _ in range(substeps):
                pool.step_physics(cmds)
            for a in list(active):
                st = active[a]
                if st["settle"] > 0:
                    st["settle"] -= 1
                    if st["settle"] == 0:
                        st["sim"].begin()
                    continue
                if not st["sim"].post_step(st["rec"]):
                    continue
                scen, cond, seed = st["meta"]
                sim = st["sim"]
                row = st["rec"].finish(sim, scen, seed, method=cond.label)
                row["condition"] = cond.label
                row["n_falls"] = len(sim.fallen)
                row["diverged"] = sim.diverged
                row["arena"] = a
                row["wall_s"] = round(time.perf_counter() - st["t0"], 1)
                if hasattr(st["filt"], "close"):
                    st["filt"].close()
                ck.add(row)
                k += 1
                print(f"[{exp.name} {k}/{total}] {scen} {cond.label} seed={seed}  "
                      f"done={row['all_done']} makespan={row['makespan_s']:.1f}s "
                      f"minDD={row['min_dogdog_m']:.2f} viol={row['viol_steps_dog']} "
                      f"dlk={row['deadlock']} falls={row['n_falls']} "
                      f"wall={row['wall_s']}s", flush=True)
                del active[a]

    md = ck.finalize(exp.md_fields)
    ck.close()
    print(f"\n== {exp.name} complete ==\n" + md + "\n", flush=True)
    return ck.rows
