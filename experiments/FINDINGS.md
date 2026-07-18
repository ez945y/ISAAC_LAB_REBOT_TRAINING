# Thesis Experiments — Result Record & Findings

Split out of EXPERIMENTS.md (2026-07-18) to keep that file a lean handoff.
This file is the THESIS-MATERIAL archive: per-experiment result blurbs,
findings F1–F11, and the decision log. Isaac-vs-kinesim consistency lives
in `experiments/isaac/results/COMPARISON.md` (regenerable).

## Result record (per experiment, kinesim phase)

| Item | What | Status |
|---|---|---|
| Capsule upgrade | 3-sphere sampling → exact segment-segment capsule in `go2_dam_wrapper.py` (+ new `capsule_geometry.py`) | ✅ done; geometry unit-tested (500 random cases vs brute force); wrapper live-tested against dam 0.7 in the venv |
| **DAM 0.7 migration** | `Go2DAMWrapper` migrated `SafetyGuard`→`Guardrail` (dict-in `{"base_pose", "action"}`, command-space 3-vector, indices 3:6→0:3, `safe_action=[0,0,0]`); stackfile hardware block rewritten (no preset, inline action_layout) | ✅ done + live-verified (PASS passthrough / head-on brake+sidestep / cohesion pull). ⚠️ `jetbot_dam_wrapper.py` + its yaml NOT migrated (demo10 breaks under 0.7 until done). Isaac machine must upgrade to dam 0.7 |
| pydam reference filter | `common/pydam.py`: numpy/scipy (SLSQP) mirror of the guard, every ablation knob exposed | ✅ done. **Cross-check vs real dam: 300 random cases, 205 clamped, worst component err 0.000** (`tests/test_pydam_vs_dam.py`); kinesim S1/S2 trajectories match real dam to 3 decimals |
| Shared infra | `common/`: kinesim (exec/obs noise, obs delay), scenarios S1–S5, filters (raw/stop/orca-RVO2/dam/pydam + FilterRouter), metrics (auto-agg, per-group completion, hard-slack telemetry), `ablation.py` sweep runner | ✅ done, smoke-tested |
| **E2 baselines (RQ2)** | `run_e2_baselines.py`: raw/stop/pydam/dam × S1–S5 × 50 seeds (1000 episodes, local venv) | ✅ **MAIN TABLE DONE — the Pareto story holds**: raw fastest but minDD ≈ 0 everywhere (collides); stop zero violations but 94–100% deadlock (never completes); **dam 98–100% completion, zero deadlock, floor held** (S2 0.786 / S5 0.914 / S4 0.612 / S1 0.527 / S3 0.456 crowd-crush) at 5–45% makespan cost, filter p99 ≤ 1.3 ms. **BONUS: pydam ≡ dam to 3 decimals in aggregate across all 5 scenarios × 50 seeds** — the ablations' pydam numbers ARE dam numbers; the per-experiment "⏳ dam re-run" items below are closed by this equivalence (Isaac-loop validation remains as E1.x). **ORCA (B2) DONE** — official RVO2 bindings, 5-method table complete: ORCA matches raw's speed and stays live in the open, but its disc model dips the TRUE capsule floor in every scenario (S1 0.339 / S2 0.440 / S3 0.224 / S4 0.324 / S5 0.543 vs dam 0.527/0.786/0.456/0.612/0.914) and deadlocks 18% at the bottleneck. dam is the only method simultaneously safe, live, and completing everywhere |
| E3.1 priority ablation | `run_e31_priority.py`: S1, priority(3:1) vs symmetric, 20 seeds pydam | ✅ **ACCEPTED**: high-pri G0 completes 7.29s vs low-pri G1 7.61s; yield carried by G1 (path_ratio 1.077 vs 1.048); violations halved (79.9→34.8 steps), min capsule dist 0.46→0.62 m. Residual 5% deadlock in BOTH conditions (likely intra-group symmetric conflict — see F2b). ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E3.2 swirl ablation | `run_e32_swirl.py`: S2 × {swirl 0, 0.6} × {jitter 0, 0.15}, 20 seeds | ✅ **ACCEPTED — found & fixed a real frame bug (F4)**. After fix: swirl 0.6 → **0 violations, min dist 0.785 m (hard floor held), fastest makespan 5.96 s** in both jitter variants; swirl 0 at perfect symmetry → near-collision 0.088 m + 82 violation steps. pydam↔dam consistency re-verified (err 0.000) |
| E3.3 soft/hard ablation | `run_e33_softhard.py`: S3 × {layered, hard_only, comfort_hard}, 20 seeds | ✅ **ACCEPTED — the layering result is the strongest so far**: layered 100% completion, makespan 12.2±0.7 s, 0 deadlock; hard_only 95% deadlock AND parks inside the floor (viol 2742 steps, slack 0.447 m — losing the soft layer costs BOTH liveness and safety); comfort_hard 0 violations but 100% deadlock (comfort-as-hard clogs the funnel). Debugging surfaced F5/F6/F7/F8. ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E3.5 velocity-aware ablation | `run_e35_velocity.py`: S2 × {va_on, va_off} × vmax {1.0, 1.5, 2.0}, 20 seeds | ✅ **ACCEPTED — clean dose-response**: va_off floor dip deepens with closing speed (minDD 0.824→0.688→0.619, viol 0→5→10 steps); va_on holds the floor at every speed (0.877/0.785/0.747, zero viol) for ≤ 3% makespan. Halved perceived closing rate = late reaction, exactly as theory predicts. Note: S2 is deterministic per condition (zero seed variance); jitter axis had no effect with swirl on. ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E3.6 γ / dt sweep | `run_e36_gamma.py`: S1+S3 × γ {0.1,0.2,0.4,0.7,1.0} (dt 0.2) + dt {0.1,0.4} at γ 0.4 + γ0.1×dt0.1, 20 seeds | ✅ **ACCEPTED — inverted-textbook curve + F9**: minDD rises MONOTONICALLY with γ (S1 0.30→0.78; S3 0.11→0.84) while liveness falls (S1 done 100%→75%, γ1.0 25% deadlock) — low γ = slack erosion + tangential blind spot (pass-through 2/20 S1, 5/20 S3 at γ0.1/dt0.2), high γ = brake-wall. dt is the second axis: dt0.1 removes S1 pass-through and DOMINATES dt0.2 on S3 (makespan 10.3 vs 12.2, same floor); dt0.4 catastrophic in congestion (80% dlk). Default γ0.4/dt0.2 sits at the knee (zero pass-through, 95–100% done); recommendation dt→0.1. S3 γ0.7 anomaly (60% dlk, worse than γ1.0): intermediate γ wedges the funnel — activates early enough to jam, too late to sort. ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E3.4 capsule vs disc | `run_e34_capsule.py`: S4 × {capsule, disc_in (h=0), disc_circ (h=0, dog-dog 1.2 / wall 0.95)}, 20 seeds | ✅ **ACCEPTED — capsule dominates both disc approximations**: capsule 100% done, true minDD 0.615; disc_in 100% done but true dips 0.476 + 32% more viol steps (under-approx is unsafe); disc_circ 100% deadlock AND true minDD 0.335 — the worst on BOTH axes: its constraints are infeasible in the 0.8 m corridor band, so the floor turns into slack erosion (F7) while liveness dies. Thesis line: over-conservative body models don't even buy safety. pydam gained ablation knob `wall_min_dist` (default unchanged; consistency gate re-passed). ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E3.7 analytic vs autograd | `run_e37_gradient.py` (needs venv): census 2000 configs + S2+S3 × {analytic, autograd} × vmax {1.5, 2.0}, 20 seeds | ✅ **ACCEPTED**: census — gradient agreement decays with proximity (cos 0.983 far → 0.861 near, min −1.0 = full inversion; 100% degenerate at predicted overlap = the theoretical spurious-stop mode). In-episode: spurious stops ≈ 0 in these scenarios; the OPERATIVE costs of naive autograd are the head-on floor (S2 0.697/0.605 vs analytic 0.785/0.747, viol 2–12 vs 0) and 3–5× filter latency (S3 p99 4.5 vs 0.84 ms); congestion floor is parity. Plus finding F10: the first autograd casualty was this experiment's OWN torch geometry (silent tail-pinning in degenerate branches). ⏳ n/a for dam (dam has no autograd path in 0.7) |
| E4.1 tracking-error injection | `run_e41_tracking.py`: S2+S3 × exec-noise σ {0, 0.1, 0.2, 0.4} m/s, 20 seeds | ✅ **ACCEPTED — graceful to a fault**: at σ=0.4 (27% of vmax) S2 floor 0.785→0.769-0.792 with ≤2 viol steps; S3 0.452→0.439 with violations slightly DOWN (dither greases the crush; makespan even improves) while hard slack rises 0.081→0.116 (QP visibly works harder). Mechanism: 50 Hz closed-loop re-linearisation corrects zero-mean noise within ~1 step (per-step displacement noise 8 mm ≪ γ margin). Thesis line: execution fidelity is NOT the binding assumption — the horizon model is (F9). ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E4.2 state noise + latency | `run_e42_obsnoise.py`: S2+S3 × {σ_obs 0–0.2 m, delay 100/200/500 ms, realistic 0.1 m+100 ms}, 20 seeds | ✅ **ACCEPTED — latency binds first, va-extrapolation carries to ~200 ms**: S2 floor 0.785→0.753 (100 ms) →0.722 (200 ms) →0.541+18 viol (500 ms, mis-placement 0.75 m ≥ the whole margin — yet velocity extrapolation still prevents anything worse); noise is milder (0.2 m → 0.718, ≤3 viol). Realistic point (0.1 m + 100 ms): S2 0.733, S3 unchanged — deployable. S3 stays crush-dominated (insensitive; worst 0.395 at 500 ms w/ 5% dlk; slack ↑ to 0.129 at 0.2 m noise). ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E4.3 slack usage | `run_e43_slack.py`: collates `max_hard_slack_m` across every finished results dir (no new sim; re-run after regenerating results; now guard-only — skips raw/stop/orca and the duplicate pydam equivalence rows) | ✅ **ACCEPTED — slack is only HALF a safety monitor**: 1836 guard episodes (re-collated 2026-07-12 over the final pool incl. e2_full + torch e37 rerun; was 1372/185 before those landed), Pearson(slack, viol) = +0.50 only. Two families separate cleanly: 128 QP-aware violation episodes (F7: hard_only wedge slack 0.447, obs-noise phantom constraints 0.129) vs **332 QP-blind episodes** (F9: swirl-off near-miss 0.088 m at slack 0.009 — the QP saw nothing). Thesis line: slack telemetry flags infeasibility-driven erosion; pair with a measured-distance monitor for linearization-driven penetration |
| E4.4 non-cooperative agent | `run_e44_rogue.py`: S1+S5 × {coop, rogue1, rogue2} (rogues run raw via FilterRouter), 20 seeds | ✅ **ACCEPTED — the 0.5-share assumption fails gracefully, not catastrophically**: floor dips dose-dependently with rogue count (S1 0.520→0.353→0.264, depth 0.18→0.35→0.44; S5 0.850→0.743→0.644) because compliant dogs cover only their half of the requirement against an agent that covers none — but NO collisions (min > 0.26 everywhere) and liveness intact (95–100% done). Guard improvement queued: carry share 1.0 against neighbours flagged non-cooperative (walls already do this, F5). Probe also found & fixed F11 (S5 cohesion-vs-goal freeze). ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E4.5 scale sweep | `run_e45_scale.py`: S5 × N {2,4,8,12,16} in the fixed ±6 m arena, 20 seeds | ✅ **ACCEPTED — liveness scales perfectly, safety degrades gracefully with density**: 100% completion and zero deadlock at EVERY N; makespan sublinear (5.9→9.0 s); floor clean through N=8 (0.729, 15 viol steps), density erosion beyond (N=12 0.601, N=16 0.478 / 100 viol — F7 pressure, arena is fixed so density doubles). Filter p99 grows 0.10→0.70 ms — 3.5% of the 20 ms @ 50 Hz budget even at N=16 (pydam/SLSQP trend only; absolute robot numbers = E1.2 with dam+OSQP). ✔ dam numbers = pydam numbers (equivalence shown in E2, 1000 eps) |
| E1.x cross-embodiment / real latency | on Isaac machine | ⬜ |

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
  by ~0.17 m per horizon. Root cause is the necessary consequence of
  finite-horizon linearization (NOT a mistuned knob): to write a linear QP
  constraint you must freeze the approach direction for one horizon. Cause:
  d_pred projects the predicted positions of
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
- **F10 the autograd path's real hazard is silent degenerate-geometry bugs
  (E3.7, 2026-07-12)**: the differentiable torch seg-seg distance written FOR
  this experiment had the classic failure — the parallel/point fallback pins
  the closest point at the TAIL, and the Gauss-Seidel re-projection only ran
  when t was clamped. Head-on d_pred was overestimated by up to 2h (0.5 m) and
  dogs clipped walls; the QP solved happily, nothing crashed, only behaviour
  degraded. Caught ONLY by brute-force comparison against the exact analytic
  geometry (3000 configs incl. forced parallel + point neighbours; now err
  ≤ 1e-6, re-projection unconditional). The production guard's analytic path
  was validated exactly this way on day one (500-case brute force). Thesis
  line: for safety filters, "differentiate through the geometry" trades a
  closed-form you can unit-test exhaustively for autodiff code whose failure
  mode is silent behavioural degradation. Census: gradient directions agree
  cos ≈ 0.98 far, degrade to 0.86 near (min −1.0), 100% degenerate at
  predicted overlap; latency 3–5×.
- **F11 cohesion fights independent goals (found by E4.4 probe, fixed
  2026-07-12)**: with S5's dogs alternating between two groups, the same-group
  cohesion pull (max_dist 4.0) fought their independent goals — dogs froze at
  the pull-vs-goal equilibrium (80% "deadlock", 5k stop-steps; MORE rogues
  meant MORE liveness because raw dogs ignore the pull). Fix: S5 gives every
  dog its own group (unstructured traffic has no squad). Thesis note: the
  cohesion boundary presumes a shared squad objective; enabling it across
  agents with independent tasks is a misconfiguration with a distinctive
  signature (commanded-but-stationary at ~max_dist from a group-mate).
- **F8 reactive filter is myopic by design — routes are the task's job**: a
  CBF filter will not path-plan around a wall (correct local-minimum freeze at
  the wall face). S3 therefore gives every method the same waypoint route
  through the gap centre (kinesim `AgentSpec.waypoints`); the filter is being
  tested on deconfliction INSIDE the funnel, not on global planning. Thesis
  framing: layered safety filter ≠ planner; cite as scope boundary.

## F12 hard_only is solver-version-sensitive (found by the 2026-07-18 regen)

Regenerating the kinesim dam pool on the Isaac machine (dam v0.7.0 tag,
osqp 1.0.5) reproduced the thesis numbers EXACTLY for E2 (all 15 cells),
E3.2, E3.4, E3.5 — but E3.3's **hard_only** flipped: thesis (Mac, dam 0.7-dev,
osqp 1.1.3) recorded 95% deadlock parked inside the floor (viol 2742, slack
0.447); here it is 100% completion with wall-scraping transit (makespan 18.2 s,
minDD 0.314, viol 503, viol_wall 201, slack 0.652). layered and comfort_hard
are unchanged. Interpretation: hard_only lives on the infeasibility knife edge
(F7) — the emergent outcome (wedge vs squeeze-through) is decided by which
slack split the QP solver returns under heavy infeasibility, so it is
sensitive to solver version/numerics. The LAYERED guard is robust across
solver versions; losing the soft layer costs not just safety+liveness but
also *determinism of the failure mode*. Thesis: report both outcomes as the
two faces of the same F7 regime; the Isaac rerun (osqp 1.0.5) shows a third:
0.6 completion + 6 falls from congestion contact.

## Isaac-in-the-loop rerun — final verdict (2026-07-18, 655 episodes)

Full suite (E2, E3.1–3.7, E4.1–4.5) re-executed with real Go2 RL locomotion +
PhysX + real dam 0.7 Guardrail, seed-matched against the kinesim dam pool
(tables+figures: `experiments/isaac/results/COMPARISON.md`).

**Every thesis-level qualitative claim survives the embodiment**: the E2
Pareto story (dam uniquely safe+live+completing; stop 100% freeze; raw now
PHYSICALLY crashes — 4 falls), capsule>disc, velocity-aware dose-response,
monotone γ curve incl. the γ0.7 wedge anomaly, latency-binds-first, rogue
dose-dependence, perfect liveness scaling to N=16 (0 falls, filter p99
1.35 ms ≈ 7 % of the 50 Hz budget).

**Systematic quantitative deltas (all explained by ~20 % command
under-tracking):** makespan +18–20 %; hard floors 0.05–0.17 m lower, gap
growing with commanded speed and crowd density (N≥12: 0.35 vs 0.47–0.51);
small viol-step counts where kinesim had zero.

**Embodiment-only findings:** (1) perfect-symmetry deadlocks don't materialise
— gait asymmetry breaks ties kinesim can't; (2) congestion produces real
contact falls (E3.3 hard_only: 6, E2 raw: 4) — "deadlock" understates what
congestion does to legged robots; (3) E4.3 slack telemetry is even blinder on
hardware-like execution (Pearson +0.17 vs +0.50; 292 QP-blind viol episodes)
— the dual-channel monitoring recommendation is mandatory, not optional;
(4) F12 solver-version sensitivity of hard_only.

## RQ1 experiments — final verdict (2026-07-19)

**E1.1 即時性（可預測）**: full-distribution benchmark over 105k replayed real
calls (quiet machine, canonical run in `results/e11_latency/`): p50 0.10–0.39 ms,
p99.9 ≤ 0.53 ms, max ≤ 0.67 ms, ZERO calls over 2 ms; cold start ≤ 0.9 ms.
Predictability: latency is monotone in the ACTIVE QP constraint count (p50
0.13 → 0.43 ms across 0→7 constraints, p99 ≤ 0.53 in every bucket), and the
constraint count is bounded by construction (≤6 statics + N−1 dynamics) →
bounded worst case, not an open tail. Rare ~40–50 ms OS-scheduling event at
~1e-5/call (non-deterministic, GC ruled out by A/B); consequence = one held
command = E1.3's delay-1-tick condition (no degradation). In-loop e45 numbers
corroborate (p99 1.35 ms at N=16 incl. contention).

**E1.2 跨具身（F13 rotational sweep + budget trade-off)**: same scenarios/
seeds/guard on a differential-drive Carter. (a) naive transfer (max_vy=0 only)
collapses — the sidestep liveness mechanism is embodiment-specific; (b) the
principled adapter (steering=cheap axis + swirl→yaw, `drive_mode=differential`)
restores kinesim behaviour, but the residual floor erosion is the ROTATIONAL
SWEEP: a turning capsule's nose swings toward the neighbour — F9's steering
variant, one-sided ≈ h, mutual ≈ 2h; (c) on real bodies the un-budgeted floor
means PHYSICAL CONTACT (Carter width 0.5 m: axis-floor 0.35 ⇒ overlap) —
kinesim cannot see this, Isaac shows falls/wedges; (d) floor re-budget maps the
trade-off: b095 (+h) perfect on S2, marginal S1/S4; b120 (+2h) safe everywhere
(0 falls, floors 0.72–0.86) but reproduces E3.4 over-conservatism in the S4
corridor (30 % completion). Isaac↔kinesim floors agree to ±0.03 in every dam
condition. Thesis line: the architecture transfers with a 4-line kinematic
adapter; the nonholonomic base faces a REAL safety-liveness trade-off the
holonomic base does not — and the guard exposes it as one interpretable knob.

**E1.3 延遲預算閉環**: actuation-side delay (guard OUTPUT arriving late —
distinct from E4.2's perception-side INPUT delay) injected at 0/20/40/100 ms on
BOTH backends: completion intact throughout; floors degrade mildly and
monotonically (S2 Isaac 0.710→0.639 at 100 ms). Against measured p99.9 =
0.53 ms this is an EMPIRICAL margin of ≥75× (flat to 40 ms) and ~190× to the
first mild degradation — closing RQ1's real-latency question without hardware.

## Handoff notes (historical, kinesim phase)

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
- 2026-07-12 Thesis chapter 4 drafted (`experiments/thesis/ch4_experiments.md`)
  + 7 publication figures (`make_figures.py`, regenerable; copies committed
  under experiments/thesis/figures/). Narrative: 3 baselines fail 3 ways ->
  7-axis attribution -> graceful degradation; F4-F11 woven in.
- 2026-07-12 ORCA baseline wired on official Python-RVO2 bindings (adapter:
  per-call micro-sim; discs r=0.35; walls = zero-max-speed agents; no
  priority). Final 5-method E2 table (1250 episodes) regenerated.
- 2026-07-12 E2 main table run locally (venv, 1000 episodes): pydam ≡ dam
  to 3 decimals in aggregate across S1–S5 × 50 seeds ⇒ ablation pydam
  numbers stand in for dam; remaining hardware questions move to E1.x.
- 2026-07-12 S5 groups changed to one-per-dog (F11) ⇒ earlier S5 numbers
  from filtered methods are stale (raw/stop unaffected — they ignore groups).
- 2026-07-12 metrics gained `stop_steps` (raw>0.5 m/s filtered to <0.05)
  and `reject_steps` (QP failure fallback) episode fields for E3.7/E4.x.
- 2026-07-11 S3 scenario semantics changed (waypoints + goal spacing + 0.5 m
  wall spacing) ⇒ any earlier S3 numbers (E2 probe runs) are stale;
  regenerate before quoting. E2 raw/stop S3+S4 re-smoked OK after the change
  (raw transits via gap with near-collisions 0.013 m; stop 100% freeze).
