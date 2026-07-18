#!/usr/bin/env python3
# ruff: noqa: I001, E402 -- Isaac modules imported only after SimulationApp starts.
"""Run the thesis experiment sweeps Isaac-in-the-loop (headless).

Same conditions as experiments/run_e*.py, executed by real Go2 quadrupeds
(shipped flat-terrain RL policy) with the REAL dam 0.7 Guardrail, in ONE
headless SimulationApp session. Results are checkpointed per episode
(episodes.jsonl) so a killed run resumes.

    source ~/IsaacLab/env_isaaclab/bin/activate
    python experiments/isaac/run_isaac_suite.py --exp e2               # one experiment
    python experiments/isaac/run_isaac_suite.py --exp all              # everything
    python experiments/isaac/run_isaac_suite.py --exp e32 --seeds 3    # quick pass

Outputs: experiments/isaac/results/<exp_key>/{episodes.jsonl,episodes.csv,
aggregate.csv,summary.md}. Compare against the kinesim reference with
experiments/isaac/compare.py (no Isaac needed).
"""

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description="Isaac-in-the-loop experiment suite")
ap.add_argument("--exp", default="all",
                help="comma list of experiment keys (e2,e31,e32,e33,e34,e35,"
                     "e36,e37,e41,e42,e44,e45) or 'all'")
ap.add_argument("--seeds", type=int, default=None,
                help="override seed count for EVERY selected experiment")
ap.add_argument("--seed-base", type=int, default=0)
ap.add_argument("--max-time", type=float, default=60.0)
ap.add_argument("--out-root", default=str(_HERE / "results"))
ap.add_argument("--arenas", type=int, default=None,
                help="parallel arenas (default: auto, up to 4 within --slot-cap dogs)")
ap.add_argument("--slot-cap", type=int, default=24,
                help="max total Go2 robots in the pool (arenas x squad size)")
args = ap.parse_args()

# ── boot Isaac headless (same proven recipe as scripts/smoke/_go2_smoke.py) ──
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os

import omni.kit.app


def _enable(ext_id: str) -> None:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(ext_id, True)


def _register_deprecated() -> None:
    """isaacsim.core.api.World lives in extsDeprecated — register that collection."""
    import isaacsim
    from omni.ext import ExtensionPathType

    dep = os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated")
    omni.kit.app.get_app().get_extension_manager().add_path(dep, ExtensionPathType.COLLECTION)


_register_deprecated()
_enable("isaacsim.core.api")
_enable("isaacsim.robot.policy.examples")
simulation_app.update()

sys.path.insert(0, str(_HERE))          # sim_backend / runner as top-level modules
sys.path.insert(0, str(_HERE.parent))   # experiments/common

from common.registry import build_experiments
from runner import run_experiment_isaac
from sim_backend import Go2Pool


def main() -> int:
    exps = build_experiments()
    keys = list(exps) if args.exp == "all" else \
        [k.strip() for k in args.exp.split(",") if k.strip()]
    unknown = [k for k in keys if k not in exps]
    if unknown:
        print(f"unknown experiment keys: {unknown} (have {list(exps)})")
        return 1

    print(f"[suite] running {keys} (seeds override={args.seeds}); "
          f"building world + Go2 pool...", flush=True)
    pool = Go2Pool()
    t0 = time.perf_counter()
    for k in keys:
        exp = exps[k]
        out_dir = Path(args.out_root) / exp.key
        print(f"\n[suite] == {exp.name} -> {out_dir} ==", flush=True)
        run_experiment_isaac(exp, pool, out_dir, seeds=args.seeds,
                             seed_base=args.seed_base, max_time=args.max_time,
                             arenas=args.arenas, slot_cap=args.slot_cap)
    print(f"\n[suite] ALL DONE in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(code)
