# Thesis Experiments — Progress & Handoff

Branch: `exp/thesis-experiments` (forked from `feat/go2-squad-dispatch` after its PR).
Maps to the RQ design (RQ1–RQ4; RQ5 dropped). One experiment script at a time:
finish + fake-data-validate the current one before starting the next.

## Local environment (IMPORTANT)

Use the DAM dev venv for anything needing torch/osqp/dam — **on this Mac**:

```bash
VENV="/Users/chenyizhong/Documents/Claude/Projects/Security Guard.nosync/.venv/bin/python3"
"$VENV" experiments/run_e2_baselines.py --methods raw,stop,pydam,dam ...
```

- venv has: numpy, scipy, torch 2.10, dam **0.7.0** (editable install of the
  local DAM repo at `.../Security Guard.nosync`), osqp 1.1.3 (installed via
  `uv pip install osqp --python "$VENV"` on 2026-07-11).
- Plain `python3` (no venv) runs raw/stop/pydam experiments (stdlib+numpy/scipy).

## Ground rules

- **Fake-data first**: every script must run and produce sane numbers on the
  lightweight kinematic sim (`common/kinesim.py`) with `raw`/`stop` filters on
  any machine (stdlib + nothing else). DAM/ORCA runs and Isaac-in-the-loop
  validation happen later on the Isaac machine.
- Results go to `experiments/results/` (gitignored, regenerable).
- Every episode is `(scenario, seed)`-reproducible; report mean ± 95% CI.

## Status board

| Item | What | Status |
|---|---|---|
| Capsule upgrade | 3-sphere sampling → exact segment-segment capsule in `go2_dam_wrapper.py` (+ new `capsule_geometry.py`) | ✅ done; geometry unit-tested (500 random cases vs brute force); wrapper live-tested against dam 0.7 in the venv |
| **DAM 0.7 migration** | `Go2DAMWrapper` migrated `SafetyGuard`→`Guardrail` (dict-in `{"base_pose", "action"}`, command-space 3-vector, indices 3:6→0:3, `safe_action=[0,0,0]`); stackfile hardware block rewritten (no preset, inline action_layout) | ✅ done + live-verified (PASS passthrough / head-on brake+sidestep / cohesion pull). ⚠️ `jetbot_dam_wrapper.py` + its yaml NOT migrated (demo10 breaks under 0.7 until done). Isaac machine must upgrade to dam 0.7 |
| pydam reference filter | `common/pydam.py`: numpy/scipy (SLSQP) mirror of the guard, every ablation knob exposed | ✅ done. **Cross-check vs real dam: 300 random cases, 205 clamped, worst component err 0.000** (`tests/test_pydam_vs_dam.py`); kinesim S1/S2 trajectories match real dam to 3 decimals |
| Shared infra | `common/`: kinesim (exec/obs noise, obs delay), scenarios S1–S5, filters (raw/stop/orca-TODO/dam/pydam + FilterRouter), metrics (auto-agg, per-group completion, hard-slack telemetry), `ablation.py` sweep runner | ✅ done, smoke-tested |
| **E2 baselines (RQ2)** | `run_e2_baselines.py`: scenario × method × seed → episodes.csv + aggregate.csv + summary.md | ✅ script done, fake-data validated (raw/stop, S1–S5). ⏳ pending: `dam` run on Isaac machine; ORCA baseline |
| E3.1 priority ablation | `run_e31_priority.py`: S1, priority(3:1) vs symmetric, 20 seeds pydam | ✅ **ACCEPTED**: high-pri G0 completes 7.29s vs low-pri G1 7.61s; yield carried by G1 (path_ratio 1.077 vs 1.048); violations halved (79.9→34.8 steps), min capsule dist 0.46→0.62 m. Residual 5% deadlock in BOTH conditions (likely intra-group symmetric conflict — see F2b). ⏳ re-run `--method dam` for the thesis table |
| E3.2 swirl ablation | S2, swirl on/off, deadlock rate | ⬜ |
| E3.3 soft/hard ablation | S3, layered vs hard-only, throughput | ⬜ |
| E3.5 velocity-aware ablation | S2, TTC vs static neighbours | ⬜ |
| E3.6 γ / dt sweep | S1+S3, conservatism-performance curve | ⬜ |
| E3.4 capsule vs disc | S4, equal-radius disc comparison | ⬜ |
| E3.7 analytic vs autograd | S2, spurious-stop rate | ⬜ |
| E4.1 tracking-error injection | noise on executed vs commanded velocity | ⬜ |
| E4.2 state noise + latency | noisy/delayed neighbour observations | ⬜ |
| E4.3 slack usage | log max hard-slack across all runs (piggyback) | ⬜ |
| E4.4 non-cooperative agent | 1–2 dogs bypass the filter | ⬜ |
| E4.5 scale sweep | S5 with N ∈ {2,4,8,12,16} (+ E1.2 timing) | ⬜ |
| E1.x cross-embodiment / real latency | on Isaac machine | ⬜ |

## How to run

```bash
# fake-data smoke (any machine, stdlib only):
python3 experiments/tests/test_capsule_geometry.py
python3 experiments/run_e2_baselines.py --methods raw,stop --scenarios S1,S2,S3,S4,S5 --seeds 5

# full E2 on the Isaac-side machine (needs torch/osqp/dam in the venv; Isaac itself not needed):
python3 experiments/run_e2_baselines.py --methods raw,stop,dam --scenarios S1,S2,S3,S4,S5 --seeds 50 --out experiments/results/e2_full
```

## Early findings (fake-data phase — keep, these are thesis material)

- **F1 hard-floor penetration**: in fast head-on approaches (S2, closing ≈3 m/s)
  the capsule distance dips to ~0.49 m (0.21 m below min_dist=0.7) for ~30 steps
  around the closest pass — pydam and real dam identically. Cause hypothesis:
  the overshoot-safe projection on the CURRENT normal over a dt=0.2 s horizon
  under-reacts while both bodies orbit (normal rotates). This is exactly the
  E4.3/E3.6 story: quantify violation depth vs γ/dt/w_hard, and frame min_dist
  as `physical clearance + ε` (bodies still never touch: 0.49 > 2×body radius).
  DO NOT silently retune before the sweeps are run.
- **F2 symmetric-priority deadlock**: S1 crossing with all priorities equal
  deadlocks in 1/3 seeds (dam & pydam alike). E3.1's job is to show priority
  resolving it.
- **F2b priority does not remove all deadlock**: E3.1 shows priority halves
  violations and produces clean yielding, but a residual ~5% deadlock persists
  in both conditions — inter-group priority can't break INTRA-group symmetric
  conflicts (both dogs share a group priority). Worth a dedicated slice in the
  thesis (per-agent priority jitter, as demo11's `_priority()` does, is the fix
  to test).
- **F3 stop-baseline mutual freeze**: threshold-stop deadlocks ~100% in every
  symmetric scenario (mutual latch below resume distance) — the headline
  weakness of the B1 baseline.

## Handoff notes (read before continuing)

1. **DAM 0.7 breaking change**: dam 0.7 renamed `SafetyGuard`→`Guardrail` and
   switched to a dict API (see `docs/release-notes-0.7.0.md` in the DAM repo,
   "no compatibility shim"). `Go2DAMWrapper` is migrated; **`JetbotDAMWrapper`
   (demo09/10) and `dam_wrapper.py`/soarm (demo13/14) are NOT** — migrate them
   the same way before running those demos against 0.7. The Isaac machine's
   venv must install dam 0.7 (editable from the Security Guard repo) before
   demo11 runs there.
2. **Neighbour tuples grew an 8th element** (`static` flag, 1.0 = wall point),
   produced by `kinesim._neighbors_of`. `Go2DAMWrapper.filter` reads indices
   0–6 only, so it is unaffected; `StopFilter` uses it to apply smaller wall
   thresholds. Keep index compatibility if you extend the tuple.
3. **ORCA baseline (B2) is a stub** — wire up vetted RVO2 python bindings, do
   not hand-roll ORCA (baseline numbers must be unimpeachable).
4. **Walls are chains of static point-agents** (spacing 0.35 m). The guard has
   a single `min_dist` for all neighbours, so dog-wall clearance is also 0.7 m
   for DAM: the S3 gap (1.6 m) is only barely feasible. Per-type min-dist (or
   true segment obstacles) is a queued guard improvement; metrics already
   report dog-wall distance separately (`wall_dist` = 0.35 m floor).
5. **Stop baseline deadlocks ~100% in symmetric scenarios** (mutual latch below
   resume distance). That is the expected narrative (the weakness of threshold
   stopping), not a bug.
6. **kinesim uses the same first-order holonomic model as the guard** — by
   design there is zero model mismatch here. Model-mismatch robustness is
   exactly E4.1/E4.2's job (inject execution/observation error); Isaac-loop
   runs close the rest of the gap.
7. Fresh filter instance per episode (no state bleed); `DamFilter` lazily
   imports torch/dam so raw/stop runs work on machines without them.

## Decision log

- 2026-07-11 Branch created; true-capsule replaces 3-sphere sampling
  (inter-sample gaps under-estimated the body on oblique crossings).
- 2026-07-11 E2 fake-data findings that drove fixes: S3 starts could overlap
  (now min-sep sampled in a deeper spawn band, hard error if unplaceable);
  StopFilter froze 1.5 m from wall chains (now separate wall thresholds
  0.45/0.6 via the static flag).
