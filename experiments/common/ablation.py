"""Shared sweep runner for the ablation/robustness experiments (E3.x, E4.x).

An experiment = scenarios x conditions x seeds. A condition bundles:
    label            goes into the episode row's "condition" column
    make_filter      callable() -> fresh filter instance (no state bleed)
    sim_overrides    dict of SimConfig field overrides (noise, delay, ...)
    scenario_kwargs  dict passed to the scenario generator (n dogs, priorities, ...)

Outputs (under ``out_dir``): episodes.csv, aggregate.csv, summary.md.
Aggregation keys: (scenario, condition).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .kinesim import KineSim, SimConfig
from .metrics import EpisodeRecorder, MetricsConfig, aggregate, to_markdown, write_csv
from .scenarios import SCENARIOS


@dataclass
class Condition:
    label: str
    make_filter: callable
    sim_overrides: dict = field(default_factory=dict)
    scenario_kwargs: dict = field(default_factory=dict)
    metrics_overrides: dict = field(default_factory=dict)


def run_sweep(
    exp_name: str,
    scenarios: list[str],
    conditions: list[Condition],
    seeds: int,
    out_dir: str,
    *,
    seed_base: int = 0,
    sim_cfg: SimConfig | None = None,
    met_cfg: MetricsConfig | None = None,
    markdown_fields: list[str] | None = None,
    verbose: bool = True,
) -> list[dict]:
    base_sim = sim_cfg or SimConfig()
    base_met = met_cfg or MetricsConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(scenarios) * len(conditions) * seeds
    k = 0
    for scen in scenarios:
        gen = SCENARIOS[scen]
        for cond in conditions:
            s_cfg = replace(base_sim, **cond.sim_overrides) if cond.sim_overrides else base_sim
            m_cfg = replace(base_met, **cond.metrics_overrides) if cond.metrics_overrides else base_met
            for i in range(seeds):
                seed = seed_base + i
                specs = gen(seed, **cond.scenario_kwargs)
                filt = cond.make_filter()
                sim = KineSim(specs, filt, s_cfg, rng_seed=seed)
                rec = EpisodeRecorder(m_cfg)
                sim.run(recorder=rec)
                row = rec.finish(sim, scen, seed, method=cond.label)
                row["condition"] = cond.label
                rows.append(row)
                if hasattr(filt, "close"):
                    filt.close()
                k += 1
                if verbose:
                    print(f"[{exp_name} {k}/{total}] {scen} {cond.label} seed={seed}  "
                          f"done={row['all_done']} makespan={row['makespan_s']:.1f}s "
                          f"minDD={row['min_dogdog_m']:.2f} viol={row['viol_steps_dog']} "
                          f"dlk={row['deadlock']}")

    write_csv(out / "episodes.csv", rows)
    agg = aggregate(rows, keys=("scenario", "condition"))
    write_csv(out / "aggregate.csv", agg)
    md = to_markdown(agg, fields=markdown_fields)
    (out / "summary.md").write_text(md + "\n")
    if verbose:
        print("\n" + md)
        print(f"\nwrote {out}/episodes.csv, aggregate.csv, summary.md")
    return rows
