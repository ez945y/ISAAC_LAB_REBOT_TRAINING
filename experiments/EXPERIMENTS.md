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
| E3.2 swirl ablation | `run_e32_swirl.py`: S2 × {swirl 0, 0.6} × {jitter 0, 0.15}, 20 seeds | ✅ **ACCEPTED — found & fixed a real frame bug (F4)**. After fix: swirl 0.6 → **0 violations, min dist 0.785 m (hard floor held), fastest makespan 5.96 s** in both jitter variants; swirl 0 at perfect symmetry → near-collision 0.088 m + 82 violation steps. pydam↔dam consistency re-verified (err 0.000) |
| E3.3 soft/hard ablation | `run_e33_softhard.py`: S3 × {layered, hard_only, comfort_hard}, 20 seeds | ✅ **ACCEPTED — the layering result is the strongest so far**: layered 100% completion, makespan 12.2±0.7 s, 0 deadlock; hard_only 95% deadlock AND parks inside the floor (viol 2742 steps, slack 0.447 m — losing the soft layer costs BOTH liveness and safety); comfort_hard 0 violations but 100% deadlock (comfort-as-hard clogs the funnel). Debugging surfaced F5/F6/F7/F8. ⏳ re-run `--method dam` on Isaac machine |
| E3.5 velocity-aware ablation | `run_e35_velocity.py`: S2 × {va_on, va_off} × vmax {1.0, 1.5, 2.0}, 20 seeds | ✅ **ACCEPTED — clean dose-response**: va_off floor dip deepens with closing speed (minDD 0.824→0.688→0.619, viol 0→5→10 steps); va_on holds the floor at every speed (0.877/0.785/0.747, zero viol) for ≤ 3% makespan. Halved perceived closing rate = late reaction, exactly as theory predicts. Note: S2 is deterministic per condition (zero seed variance); jitter axis had no effect with swirl on. ⏳ re-run `--method dam` for thesis table |
| E3.6 γ / dt sweep | `run_e36_gamma.py`: S1+S3 × γ {0.1,0.2,0.4,0.7,1.0} (dt 0.2) + dt {0.1,0.4} at γ 0.4 + γ0.1×dt0.1, 20 seeds | ✅ **ACCEPTED — inverted-textbook curve + F9**: minDD rises MONOTONICALLY with γ (S1 0.30→0.78; S3 0.11→0.84) while liveness falls (S1 done 100%→75%, γ1.0 25% deadlock) — low γ = slack erosion + tangential blind spot (pass-through 2/20 S1, 5/20 S3 at γ0.1/dt0.2), high γ = brake-wall. dt is the second axis: dt0.1 removes S1 pass-through and DOMINATES dt0.2 on S3 (makespan 10.3 vs 12.2, same floor); dt0.4 catastrophic in congestion (80% dlk). Default γ0.4/dt0.2 sits at the knee (zero pass-through, 95–100% done); recommendation dt→0.1. S3 γ0.7 anomaly (60% dlk, worse than γ1.0): intermediate γ wedges the funnel — activates early enough to jam, too late to sort. ⏳ dam re-run |
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

# full E2 with the real guard (local DAM venv works; Isaac itself not needed):
"$VENV" experiments/run_e2_baselines.py --methods raw,stop,dam --scenarios S1,S2,S3,S4,S5 --seeds 50 --out experiments/results/e2_full

# ablations (pydam by default; add --method dam where supported):
python3 experiments/run_e31_priority.py --seeds 20      # E3.1 priority
python3 experiments/run_e32_swirl.py    --seeds 20      # E3.2 swirl x jitter
python3 experiments/run_e33_softhard.py --seeds 20      # E3.3 layering
python3 experiments/run_e35_velocity.py --seeds 20      # E3.5 velocity-aware x vmax
python3 experiments/run_e36_gamma.py    --seeds 20      # E3.6 gamma/dt sweep (slow: 320 eps)

# implementation-consistency gate (run after ANY guard/pydam change):
"$VENV" experiments/tests/test_pydam_vs_dam.py
python3 experiments/tests/test_capsule_geometry.py
```

## Early findings (fake-data phase — keep, these are thesis material)

- **F1 hard-floor penetration (REVISED by F4)**: the 0.45–0.49 m dips first seen
  in S2 were mostly caused by the swirl frame bug (F4). With the fix, S2 head-on
  holds ≥ 0.785 m with zero violations. Residual penetration still exists in
  multi-dog scenarios (S1 min ≈ 0.62 with priority) — quantify properly in
  E3.6/E4.3; keep the `min_dist = clearance + ε` framing.
- **F4 swirl reference-frame bug (found by E3.2, fixed 2026-07-11)**: the swirl
  tangent was a WORLD-frame direction applied directly to the BODY-frame
  [du_vx, du_vy] cost — two head-on dogs (yaw π apart) were biased to the SAME
  world side, staying collinear (probe: both drifted world-y −0.450). Fix:
  rotate the tangent into each body frame (`go2_dam_wrapper.py` + `pydam.py`,
  identical change). After the fix swirl provides BOTH liveness and safety:
  symmetric encounters go from 0.088 m near-miss to 0.785 m clean orbit.
  ⚠️ demo11's visual "swirl works" was asymmetry breaking the tie, not coherent
  circulation — thesis should report the before/after ablation as a finding.
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
- **F5 walls must be hard-floor-only (found by E3.3, fixed 2026-07-11)**:
  applying the comfort layer to static geometry makes any corridor narrower
  than 2×comfort_dist permanently impassable (dogs froze 1.4 m short of the
  1.6 m gap), and swirl must never "circulate" around a wall. Fix: static
  neighbours contribute ONLY the hard constraint, and carry share 1.0 (a wall
  cannot do its half of the avoiding). Same change in wrapper + pydam.
- **F6 wall points must be POINTS, not capsules (found by E3.3, fixed
  2026-07-11)**: the guard gave every neighbour the dog's own capsule
  half-length (0.25 m). For wall samples (yaw 0) this fattened each point by
  0.25 m on both sides — the S3 gap shrank 0.8→0.55 m per side, dist 0.667 <
  min_dist 0.7, so the gap CENTRE was already inside the hard floor and even a
  SINGLE dog froze 0.63 m short. Fix: `nb_half=0` for static neighbours
  (wrapper + pydam). Single dog now transits in ~8 s, 5/5 seeds.
- **F7 crowd crush — sustained congestion erodes the hard floor**: in the S3
  funnel the layered guard dips to minDD ≈ 0.45 (floor 0.7) with ~260
  violation steps despite tiny per-QP slack (~0.07): four raw commands pushing
  into conflicting constraints (walls + mutual) erode the floor gradually.
  hard_only is far worse (wedges at 0.49 and SITS inside the floor all
  episode). Quantify vs γ in E3.6; report via E4.3 slack telemetry.
- **F9 tangential linearization blind spot — γ is NOT a pure conservatism
  knob (found by E3.6 probe, 2026-07-11)**: at S1's 90° crossing with γ=0.1
  (dt 0.2) two capsules pass THROUGH each other (minDD 0.0) while both QPs
  report near-zero slack. Verified step-by-step: each dog satisfies its
  half-share constraint with EQUALITY (g·du = 0.5·req, slack 0), the shares
  sum to the full requirement — yet realized distance violates the CBF bound
  by ~0.17 m per horizon. Cause: d_pred projects the predicted positions of
  the CURRENT closest-point pair onto the CURRENT normal; at 90° with
  tangential relative motion the closest pair slides along the spines and the
  normal rotates within the 0.2 s horizon, so real closure is second-order
  invisible to the constraint, and equality-binding leaves zero margin to
  absorb it. Low γ makes it WORSE (constraints activate far out → dogs spend
  long periods manoeuvring tangentially at close range where the error
  accumulates); swirl (deliberate tangential motion) and va_off deepen it.
  γ→1 flips the failure to brake-wall deadlock instead. **dt is the binding
  fix: dt 0.2→0.1 removes S1 pass-through (2/20→0/20) and cuts S3's
  (5/20→1/20), while at the γ=0.4 default dt0.1 outright DOMINATES dt0.2 on
  S3** (makespan 10.3 vs 12.2 s, same floor). Guard improvement queued:
  shorten horizon (or multi-point spine constraints); stackfile default
  γ=0.4/dt=0.2 has zero pass-through (worst episode 0.12 m at dt0.4) but
  inherits the reduced-margin regime (S1 minDD ≈ 0.52±0.05).
- **F8 reactive filter is myopic by design — routes are the task's job**: a
  CBF filter will not path-plan around a wall (correct local-minimum freeze at
  the wall face). S3 therefore gives every method the same waypoint route
  through the gap centre (kinesim `AgentSpec.waypoints`); the filter is being
  tested on deconfliction INSIDE the funnel, not on global planning. Thesis
  framing: layered safety filter ≠ planner; cite as scope boundary.

## Handoff notes (read before continuing)

1. **DAM 0.7 breaking change**: dam 0.7 renamed `SafetyGuard`→`Guardrail` and
   switched to a dict API (see `docs/release-notes-0.7.0.md` in the DAM repo,
   "no compatibility shim"). `Go2DAMWrapper` is migrated; **`JetbotDAMWrapper`
   (demo09/10) and `dam_wrapper.py`/soarm (demo13/14) are NOT** — migrate them
   the same way before running those demos against 0.7. The Isaac machine's
   venv must install dam 0.7 (editable from the Security Guard repo) before
   demo11 runs there.
2. **Neighbour tuples grew an 8th element** (`static` flag, 1.0 = wall point),
   produced by `kinesim._neighbors_of`. `Go2DAMWrapper.filter` now forwards it
   into the holder and the guard uses it three ways (F5/F6): static ⇒ hard
   constraint only, share 1.0, `nb_half=0` (point, not capsule). `StopFilter`
   uses it for smaller wall thresholds. Keep index compatibility if you extend
   the tuple; the consistency test exercises the static branch (~30% of cases).
3. **ORCA baseline (B2) is a stub** — wire up vetted RVO2 python bindings, do
   not hand-roll ORCA (baseline numbers must be unimpeachable).
4. **Walls are chains of static point-agents** (spacing 0.5 m, culled to the
   nearest 6 within 3.5 m per dog). The guard keeps a single `min_dist` for
   all neighbours, so dog-wall clearance is 0.7 m from the dog's capsule AXIS
   to the wall point (F6: wall points are points, `nb_half=0`): the S3 gap
   (±0.8 m) leaves 0.1 m of slack — barely feasible BY DESIGN. Per-type
   min-dist (or true segment obstacles) is a queued guard improvement; metrics
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
- 2026-07-11 S3 greedy start sampling could wedge (seed 11): added
  deterministic retry rounds (`seed*1000+round`); goals still drawn from the
  base seed so goal layouts stay comparable across rounds.
- 2026-07-11 DAM 0.7 migration verified in the local venv; swirl frame bug
  (F4) found by E3.2 and fixed in both implementations the same day.
- 2026-07-11 E3.3 debugging found THREE wall-handling defects (F5, F6) and a
  task-design gap (F8): walls now hard-only/full-share/point-shaped in wrapper
  + pydam; S3 routes all dogs via gap-centre waypoints. Waypoint advance uses
  radius OR plane-crossing (congestion can shove a dog past a waypoint it
  never got within tol of; radius-only made it command BACK into the crowd —
  that was the residual 40% "deadlock", not the guard). kinesim also culls
  static neighbours to the nearest 6 within 3.5 m (QP size; nearer points
  geometrically dominate) and wall spacing is 0.5 m (still impassable at
  min_dist 0.7). pydam↔dam consistency re-verified after every change
  (err 0.000, static branch covered 30% of cases).
- 2026-07-11 S3 scenario semantics changed (waypoints + goal spacing + 0.5 m
  wall spacing) ⇒ any earlier S3 numbers (E2 probe runs) are stale;
  regenerate before quoting. E2 raw/stop S3+S4 re-smoked OK after the change
  (raw transits via gap with near-collisions 0.013 m; stop 100% freeze).
