#!/usr/bin/env python3
"""Registry-driven kinesim runner: the dam-reference twin of the Isaac suite.

Runs the SAME manifest (``common/registry.py``) as
``experiments/isaac/run_isaac_suite.py``, on the lightweight kinematic sim with
the real dam guard — one command instead of twelve ``run_e*.py`` invocations,
same checkpoint format (resume-safe), directly comparable outputs.

The historical per-experiment scripts (run_e2_baselines.py, run_e3x/e4x)
remain the reference entry points for pydam/raw/orca variants and E3.7's
census; THIS is the fast path for regenerating the dam reference pool.

    source ~/IsaacLab/env_isaaclab/bin/activate
    python experiments/run_kinesim_suite.py --exp all
    python experiments/run_kinesim_suite.py --exp e32,e35 --seeds 20
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.kinesim import KineSim, SimConfig  # noqa: E402
from common.metrics import EpisodeRecorder, MetricsConfig  # noqa: E402
from common.registry import KINESIM_DIRS, Experiment, build_experiments  # noqa: E402
from common.scenarios import SCENARIOS  # noqa: E402
from common.sweep_io import Checkpointer  # noqa: E402


def run_experiment_kinesim(exp: Experiment, out_dir: Path, *,
                           seeds: int | None = None, seed_base: int = 0,
                           max_time: float = 60.0) -> None:
    ck = Checkpointer(out_dir)
    n_seeds = seeds if seeds is not None else exp.seeds
    base_sim = SimConfig(max_time=max_time)
    base_met = MetricsConfig()
    total = len(exp.scenarios) * len(exp.conditions) * n_seeds
    k = len(ck.rows)
    for scen in exp.scenarios:
        gen = SCENARIOS[scen]
        for cond in exp.conditions:
            s_cfg = (replace(base_sim, **cond.sim_overrides)
                     if cond.sim_overrides else base_sim)
            m_cfg = (replace(base_met, **cond.metrics_overrides)
                     if cond.metrics_overrides else base_met)
            for i in range(n_seeds):
                seed = seed_base + i
                if ck.skip(scen, cond.label, seed):
                    continue
                filt = cond.make_filter()
                sim = KineSim(gen(seed, **cond.scenario_kwargs), filt, s_cfg,
                              rng_seed=seed)
                rec = EpisodeRecorder(m_cfg)
                t0 = time.perf_counter()
                sim.run(recorder=rec)
                row = rec.finish(sim, scen, seed, method=cond.label)
                row["condition"] = cond.label
                row["wall_s"] = round(time.perf_counter() - t0, 1)
                if hasattr(filt, "close"):
                    filt.close()
                ck.add(row)
                k += 1
                print(f"[{exp.name} {k}/{total}] {scen} {cond.label} seed={seed}  "
                      f"done={row['all_done']} makespan={row['makespan_s']:.1f}s "
                      f"minDD={row['min_dogdog_m']:.2f} viol={row['viol_steps_dog']} "
                      f"dlk={row['deadlock']}", flush=True)
    md = ck.finalize(exp.md_fields)
    ck.close()
    print(f"\n== {exp.name} complete ==\n" + md + "\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="kinesim dam-reference suite")
    ap.add_argument("--exp", default="all")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override seed count (default: registry per-experiment)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--out-root",
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    exps = build_experiments()
    keys = list(exps) if args.exp == "all" else \
        [s.strip() for s in args.exp.split(",") if s.strip()]
    for key in keys:
        exp = exps[key]
        out_dir = Path(args.out_root) / KINESIM_DIRS[key]
        print(f"\n== {exp.name} -> {out_dir} ==", flush=True)
        run_experiment_kinesim(exp, out_dir, seeds=args.seeds,
                               seed_base=args.seed_base, max_time=args.max_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
