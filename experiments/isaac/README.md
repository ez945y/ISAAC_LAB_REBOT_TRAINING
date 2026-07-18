# Isaac-in-the-loop experiment rerun (E2 / E3.x / E4.x)

Re-runs the thesis experiments with **Isaac Sim executing the motion** (real Go2
quadrupeds, shipped `Go2FlatTerrainPolicy` RL policy, PhysX @ 200 Hz, headless)
and the **real dam 0.7 `Guardrail`** as the safety filter — to check the
kinesim/pydam numbers hold on a real embodiment. The kinesim scripts and their
results are untouched; everything Isaac-side lives in this folder.

## Run

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"   # aarch64 isaacsim quirk

python experiments/isaac/run_isaac_suite.py --exp all        # everything (hours)
python experiments/isaac/run_isaac_suite.py --exp e2,e32     # subset
python experiments/isaac/run_isaac_suite.py --exp e45 --seeds 3
```

One long-lived headless `SimulationApp` per invocation. Episodes checkpoint to
`results/<exp>/episodes.jsonl` — re-running the same command **resumes** (done
`(scenario, condition, seed)` triples are skipped). Final `episodes.csv`,
`aggregate.csv`, `summary.md` are rewritten when the sweep finishes.

Compare with the kinesim reference (no Isaac needed):

```bash
python experiments/isaac/compare.py          # -> results/COMPARISON.md
```

## Design

- `common/registry.py` — the sweep manifest (conditions 1:1 from `run_e*.py`:
  same knobs, labels, scenarios, seeds semantics), consumed by BOTH the Isaac
  suite and `experiments/run_kinesim_suite.py`. Guard is always `--method dam`.
  Add new conditions THERE, nowhere else.
- `sim_backend.py` — `IsaacArenaSim(KineSim)`: inherits the nominal waypoint
  P-controller, observation delay/noise pipeline and neighbour assembly
  **verbatim** from `common/kinesim.py`. The 50 Hz safe command is held for
  4 × 200 Hz physics steps (policy applied every physics step — the proven
  demo11/smoke pattern), then poses are measured back. The loop is externalised
  (`control_tick` / `post_step`) so episodes can interleave.
- `runner.py` — **arena tiling**: `Go2Pool` hosts K arenas (40 m apart, ≤4,
  capped by `--slot-cap` total dogs) in ONE world; each runs an independent
  episode and `world.step` is amortised across them. An arena that finishes
  teleports straight to the next queued episode (own 1 s settle window) while
  the others keep running. Barrier + `world.reset()` only when the WALL layout
  changes (scenario switch) or the pool grows. Episode reset = teleport +
  default joint state + policy-memory clear; idle dogs park at (0, 60).
- `common/sweep_io.py` — shared per-episode jsonl checkpointing (resume) +
  csv/aggregate/summary finalization, used by both backends.
- Walls (S3/S4) are fed to the filter as the same metric point-chains AND
  spawned as real static colliders (a raw dog can physically hit one),
  replicated per arena.

## Semantics that differ from kinesim (expected, part of the result)

- **Tracking dynamics**: the RL policy tracks `[vx, vy, ω]` imperfectly
  (acceleration limits, gait); kinesim executed commands exactly. vmax 2.0
  conditions (E3.5/E3.7) may saturate the policy's trained command range.
- **exec_noise_std** (E4.1) is injected on the *commanded* world velocity
  (kinesim: on the *executed* one) — the tracker sits in between.
- **Falls/divergence**: rows gain `n_falls` / `diverged` / `wall_s` columns; a
  fallen dog gets a zero command for the rest of the episode (→ timeout).
- **Gait wobble** inflates `path_len` slightly and adds noise to the
  finite-difference neighbour velocities the guard consumes.
- **filter_p50/p99_ms** are this machine's CPU numbers, not the Mac's.
- ORCA (B2) is skipped (no pyrvo2 here); E3.7's census lives kinesim-side.

## Status

See `experiments/EXPERIMENTS.md` § "Isaac machine environment" for the running
status board and the kinesim-reference regen layout.
