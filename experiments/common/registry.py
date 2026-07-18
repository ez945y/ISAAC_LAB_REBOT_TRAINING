"""SINGLE SOURCE OF TRUTH for the thesis experiment sweeps (E2, E3.x, E4.x).

Every runner — the Isaac-in-the-loop suite (``experiments/isaac/``) and the
kinesim dam-reference suite (``experiments/run_kinesim_suite.py``) — consumes
THIS manifest, so conditions/labels/scenarios/seeds are defined exactly once.
The definitions are 1:1 ports of the per-experiment ``run_e*.py`` scripts
(kept as the historical kinesim entry points; do not edit conditions there).

The guard is always the REAL dam Guardrail (``--method dam`` equivalent).
ORCA is absent (no pyrvo2 on this machine); E3.7's census is a kinesim-side
extra (sim-independent, see run_e37_gradient.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .ablation import Condition
from .filters import FilterRouter, RawFilter, StopFilter, make_guard


@dataclass
class Experiment:
    name: str            # e.g. "E3.2"
    key: str             # results dir name, e.g. "e32_swirl"
    scenarios: list
    conditions: list
    md_fields: list
    seeds: int           # default seed count for the Isaac run
    max_dogs: int        # pool size this experiment needs


def _dam(**knobs):
    return make_guard("dam", **knobs)


def _rogue_router(rogues: tuple = ()) -> FilterRouter:
    return FilterRouter(_dam(), {aid: RawFilter() for aid in rogues})


# Rogue ids merged across scenarios exactly as in run_e44_rogue.py
_ROGUES_1 = ("G1_0", "D0")
_ROGUES_2 = ("G1_0", "G0_1", "D0", "D1")


def build_experiments() -> dict[str, Experiment]:
    exps: dict[str, Experiment] = {}

    # -- E2 main table (run_e2_baselines.py; orca skipped) -------------------
    exps["e2"] = Experiment(
        "E2", "e2_baselines", ["S1", "S2", "S3", "S4", "S5"],
        [
            Condition(label="dam", make_filter=_dam),
            Condition(label="raw", make_filter=RawFilter),
            Condition(label="stop", make_filter=StopFilter),
        ],
        ["makespan_s", "min_dogdog_m", "viol_steps_dog", "deadlock_rate",
         "all_done_rate", "intervention_rate", "filter_p99_ms"],
        seeds=10, max_dogs=6,
    )

    # -- E3.1 priority (run_e31_priority.py, n_per_group=2) ------------------
    exps["e31"] = Experiment(
        "E3.1", "e31_priority", ["S1"],
        [
            Condition(label="priority_on", make_filter=_dam,
                      scenario_kwargs={"n_per_group": 2, "pri_g0": 3.0, "pri_g1": 1.0}),
            Condition(label="symmetric", make_filter=_dam,
                      scenario_kwargs={"n_per_group": 2, "pri_g0": 1.0, "pri_g1": 1.0}),
        ],
        ["completion_G0_done_s", "completion_G1_done_s", "path_ratio_G0",
         "path_ratio_G1", "deadlock_rate", "all_done_rate", "min_dogdog_m",
         "viol_steps_dog"],
        seeds=10, max_dogs=4,
    )

    # -- E3.2 swirl x jitter (run_e32_swirl.py) ------------------------------
    exps["e32"] = Experiment(
        "E3.2", "e32_swirl", ["S2"],
        [
            Condition(label=f"swirl_{s}_j{j}",
                      make_filter=lambda s=s: _dam(swirl=s),
                      scenario_kwargs={"jitter": j})
            for s in (0.0, 0.6) for j in (0.0, 0.15)
        ],
        ["deadlock_rate", "all_done_rate", "makespan_s", "min_dogdog_m",
         "viol_steps_dog", "path_ratio", "intervention_rate"],
        seeds=10, max_dogs=2,
    )

    # -- E3.3 soft/hard layering (run_e33_softhard.py, n=4) ------------------
    sk33 = {"n": 4}
    exps["e33"] = Experiment(
        "E3.3", "e33_softhard", ["S3"],
        [
            Condition(label="layered", scenario_kwargs=sk33, make_filter=_dam),
            Condition(label="hard_only", scenario_kwargs=sk33,
                      make_filter=lambda: _dam(comfort_dist=0.7)),
            Condition(label="comfort_hard", scenario_kwargs=sk33,
                      make_filter=lambda: _dam(w_soft=1000.0)),
        ],
        ["makespan_s", "mean_completion_s", "all_done_rate", "deadlock_rate",
         "min_dogdog_m", "viol_steps_dog", "mean_dcmd", "max_hard_slack_m"],
        seeds=10, max_dogs=4,
    )

    # -- E3.4 capsule vs disc (run_e34_capsule.py) ---------------------------
    exps["e34"] = Experiment(
        "E3.4", "e34_capsule", ["S4"],
        [
            Condition(label="capsule", make_filter=_dam),
            Condition(label="disc_in", make_filter=lambda: _dam(capsule_half=0.0)),
            Condition(label="disc_circ",
                      make_filter=lambda: _dam(capsule_half=0.0, min_dist=1.2,
                                               wall_min_dist=0.95)),
        ],
        ["all_done_rate", "deadlock_rate", "makespan_s", "min_dogdog_m",
         "viol_steps_dog", "viol_steps_wall", "mean_dcmd",
         "intervention_rate", "max_hard_slack_m"],
        seeds=10, max_dogs=2,
    )

    # -- E3.5 velocity-aware x vmax (run_e35_velocity.py) --------------------
    exps["e35"] = Experiment(
        "E3.5", "e35_velocity", ["S2"],
        [
            Condition(label=f"va_{'on' if va else 'off'}_v{v}",
                      make_filter=lambda va=va, v=v: _dam(velocity_aware=va, max_v=v),
                      sim_overrides={"vmax": v},
                      scenario_kwargs={"jitter": 0.15})
            for va in (True, False) for v in (1.0, 1.5, 2.0)
        ],
        ["min_dogdog_m", "viol_steps_dog", "deadlock_rate", "all_done_rate",
         "makespan_s", "mean_dcmd", "intervention_rate", "max_hard_slack_m"],
        seeds=10, max_dogs=2,
    )

    # -- E3.6 gamma / dt sweep (run_e36_gamma.py) ----------------------------
    exps["e36"] = Experiment(
        "E3.6", "e36_gamma", ["S1", "S3"],
        [
            Condition(label=f"g{g}_dt0.2", make_filter=lambda g=g: _dam(gamma=g))
            for g in (0.1, 0.2, 0.4, 0.7, 1.0)
        ] + [
            Condition(label=f"g0.4_dt{dt}",
                      make_filter=lambda dt=dt: _dam(gamma=0.4, dt=dt))
            for dt in (0.1, 0.4)
        ] + [
            Condition(label="g0.1_dt0.1",
                      make_filter=lambda: _dam(gamma=0.1, dt=0.1)),
        ],
        ["makespan_s", "mean_completion_s", "all_done_rate", "deadlock_rate",
         "min_dogdog_m", "viol_steps_dog", "mean_dcmd", "intervention_rate",
         "max_hard_slack_m"],
        seeds=5, max_dogs=4,
    )

    # -- E3.7 analytic vs autograd (run_e37_gradient.py; census stays kinesim)
    exps["e37"] = Experiment(
        "E3.7", "e37_gradient", ["S2", "S3"],
        [
            Condition(label=f"{mode}_v{v}",
                      make_filter=lambda mode=mode, v=v: _dam(grad_mode=mode, max_v=v),
                      sim_overrides={"vmax": v})
            for mode in ("analytic", "autograd") for v in (1.5, 2.0)
        ],
        ["stop_steps", "reject_steps", "min_dogdog_m", "viol_steps_dog",
         "all_done_rate", "deadlock_rate", "makespan_s",
         "intervention_rate", "filter_p99_ms"],
        seeds=5, max_dogs=4,
    )

    # -- E4.1 tracking-error injection (run_e41_tracking.py) -----------------
    exps["e41"] = Experiment(
        "E4.1", "e41_tracking", ["S2", "S3"],
        [
            Condition(label=f"sig{s}", make_filter=_dam,
                      sim_overrides={"exec_noise_std": s})
            for s in (0.0, 0.1, 0.2, 0.4)
        ],
        ["min_dogdog_m", "viol_steps_dog", "viol_steps_wall",
         "all_done_rate", "deadlock_rate", "makespan_s", "mean_dcmd",
         "max_hard_slack_m"],
        seeds=5, max_dogs=4,
    )

    # -- E4.2 state noise + latency (run_e42_obsnoise.py) --------------------
    exps["e42"] = Experiment(
        "E4.2", "e42_obsnoise", ["S2", "S3"],
        [
            Condition(label=f"noise{s}", make_filter=_dam,
                      sim_overrides={"obs_noise_std": s})
            for s in (0.0, 0.05, 0.1, 0.2)
        ] + [
            Condition(label=f"delay{k * 20}ms", make_filter=_dam,
                      sim_overrides={"obs_delay_steps": k})
            for k in (5, 10, 25)
        ] + [
            Condition(label="real_0.1m_100ms", make_filter=_dam,
                      sim_overrides={"obs_noise_std": 0.1, "obs_delay_steps": 5}),
        ],
        ["min_dogdog_m", "viol_steps_dog", "viol_steps_wall",
         "all_done_rate", "deadlock_rate", "makespan_s", "mean_dcmd",
         "intervention_rate", "max_hard_slack_m"],
        seeds=5, max_dogs=4,
    )

    # -- E4.4 non-cooperative agents (run_e44_rogue.py) ----------------------
    exps["e44"] = Experiment(
        "E4.4", "e44_rogue", ["S1", "S5"],
        [
            Condition(label="coop", make_filter=_rogue_router),
            Condition(label="rogue1", make_filter=lambda: _rogue_router(_ROGUES_1)),
            Condition(label="rogue2", make_filter=lambda: _rogue_router(_ROGUES_2)),
        ],
        ["min_dogdog_m", "viol_steps_dog", "max_viol_depth_m",
         "all_done_rate", "deadlock_rate", "makespan_s", "max_hard_slack_m"],
        seeds=10, max_dogs=6,
    )

    # -- E4.5 scale sweep (run_e45_scale.py) ---------------------------------
    exps["e45"] = Experiment(
        "E4.5", "e45_scale", ["S5"],
        [
            Condition(label=f"n{n}", make_filter=_dam, scenario_kwargs={"n": n})
            for n in (2, 4, 8, 12, 16)
        ],
        ["min_dogdog_m", "viol_steps_dog", "all_done_rate", "deadlock_rate",
         "makespan_s", "mean_completion_s", "intervention_rate",
         "filter_p50_ms", "filter_p99_ms", "max_hard_slack_m"],
        seeds=5, max_dogs=16,
    )

    return exps


# experiment key -> kinesim reference results dir under experiments/results/
# (produced by the original run_e*.py scripts / run_kinesim_suite.py)
KINESIM_DIRS = {
    "e2": "e2_full",
    "e31": "e31_priority_dam",
    "e32": "e32_swirl_dam",
    "e33": "e33_softhard_dam",
    "e34": "e34_capsule_dam",
    "e35": "e35_velocity_dam",
    "e36": "e36_gamma_dam",
    "e37": "e37_gradient_dam",
    "e41": "e41_tracking_dam",
    "e42": "e42_obsnoise_dam",
    "e44": "e44_rogue_dam",
    "e45": "e45_scale_dam",
}
