# DAM + Isaac Demo Readiness Plan

This document is the handoff plan for the unpushed DAM/Isaac integration work
currently living in this sidecar branch. The goal is to demo DAM as a real
runtime safety layer in Isaac, not as a passive logging wrapper.

## Current Baseline

- Branch state at handoff time: `master` is ahead of `origin/master` by the DAM
  integration commits.
- Test entrypoint in this sidecar repo: `make test`.
- `make test` is not an upstream historical project target; it was added here
  so this branch has one repeatable local gate.
- Runtime demo still needs an Isaac runtime host. Local pytest coverage verifies
  packaging, joint-space filtering, EE-space resolver behavior, stale-target
  rejection, gripper handling, and shape contracts.

## Demo Story

The demo should prove three claims in order:

1. DAM receives every command before Isaac applies it.
2. DAM decisions are visible and actionable.
3. DAM failure modes stop safely at the wrapper boundary.

### Segment 1: Audience Comparison

Run:

```bash
python scripts/dam_scripted_comparison_demo.py --mode compare
```

This is the preferred recording demo. It replays the same scripted EE target
twice: first as `RAW COMMAND`, then as `DAM ON`. The trajectory starts with a
normal reach and then pushes into a deliberately unsafe/uncomfortable region.

Acceptance:

- The terminal clearly separates `RAW COMMAND` and `DAM ON` segments.
- The target trajectory is deterministic; no keyboard or leader arm is needed.
- The DAM segment displays `PASS`, `CLAMP`, `REJECT`, or `FAULT`.
- The recording shows the same input story with and without the safety layer.

Tuning:

```bash
python scripts/dam_scripted_comparison_demo.py --mode compare --unsafe-scale 1.4
python scripts/dam_scripted_comparison_demo.py --mode dam --steps 900 --log-every 45
python scripts/dam_scripted_comparison_demo.py --mode compare --summary-path outputs/dam_linkedin_summary.md
```

LinkedIn value message:

- **Hook**: "Same command, safer robot."
- **Problem**: policies and teleop streams can emit unsafe targets faster than
  humans can inspect.
- **Proof**: the replay prints risky command frames, DAM interventions, and
  decision counts.
- **Buyer value**: safer robot iteration without rewriting the controller,
  policy, or Isaac scene.

### Segment 2: Joint-Space Safety

Run:

```bash
python scripts/dam_teleoperate_demo.py --controller ik
```

Use a leader-arm joint-space input path after the scripted comparison is stable.
The operator should see:

- `[DAM PASS]` during normal movement.
- `last_safe_gripper` applied to the gripper joint target.
- No direct simulated joint target bypassing `DAMSafetyWrapper.filter()`.

Acceptance:

- Arm targets come from `dam.filter(...)`.
- Gripper target comes from `dam.last_safe_gripper`.
- Status output includes decision and clamp rate.

### Segment 3: Safety Intervention

Use an aggressive or intentionally unsafe input profile that exceeds the DAM
stackfile limits.

Acceptance:

- Status can show `CLAMP`, `REJECT`, or `FAULT`, not only `PASS`.
- If multiple guard results are present, the displayed decision follows
  `FAULT > REJECT > CLAMP > PASS`.
- The robot receives the validated target, not the raw unsafe proposal.

### Segment 4: EE-Space Resolver Path

Run:

```bash
python scripts/dam_safety_demo.py --controller ik
```

This path should show Isaac EE pose targets flowing through:

```text
Keyboard target pose
-> DAMSafetyWrapper.filter_ee()
-> IsaacControllerKinematicsResolver.inverse_kinematics()
-> DAM SafetyGuard(input_space="ee")
-> Pinocchio FK confirmation
-> validated arm + gripper target back to Isaac
```

Acceptance:

- `filter_ee()` is called only after `attach_isaac_controller(...)`.
- Multi-env input is rejected. The demo must run with `--num_envs 1`.
- Resolver output must be complete `arm + gripper`; incomplete output fails
  before Isaac receives a command.
- Stale resolver targets are cleared before each EE guard call.

## What To Record

For a credible demo package, capture:

- Screen recording of the robot moving under normal DAM `PASS` status.
- A second clip showing a visible `CLAMP`, `REJECT`, or `FAULT`.
- Terminal log lines for `step`, `clamp_rate`, `ee_err`, and gripper target.
- The `LINKEDIN DEMO SUMMARY` block, or a summary file produced with
  `--summary-path`.
- The exact stackfile path used.
- Isaac Sim, Isaac Lab, GPU, driver, and Python environment versions.
- `make test` output from this sidecar branch.

For post structure, captions, and the 30-second shot list, use
[`dam_linkedin_launch_kit.md`](./dam_linkedin_launch_kit.md).

## Isaac Sim 6.0 / Isaac Lab 3.0 Watchlist

The repository README currently targets Isaac Sim 5.1+ and Isaac Lab 2.3.0.
Treat Isaac Sim 6.0 as a migration, not a version-string edit.

Official references:

- Isaac Sim 6.0 release notes:
  https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html
- Isaac Sim 6.0 requirements:
  https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html
- ROS 2 OmniGraph migration:
  https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/ros2_omnigraph_migration.html
- Isaac Lab releases:
  https://github.com/isaac-sim/IsaacLab/releases

### Compatibility Risks

- **Hardware and driver floor**: Isaac Sim 6.0 lists high-end RTX GPUs and
  recent drivers in its requirements. Do not schedule the demo on an older
  workstation without first running a smoke test.
- **Isaac Lab 3.0 is a major architecture change**: Isaac Lab 3.0 Beta is built
  on Isaac Sim 6.0 and introduces multi-backend physics, a pluggable renderer
  system, and kit-less workflows. Validate this repo's imports and controller
  assumptions before claiming 6.0 support.
- **Controller validation is stricter**: Re-check command tensor shapes,
  `joint_ids`, NaN handling, and gripper target shape under Isaac Sim 6.0.
  The DAM wrapper's recent shape checks are necessary but not a full runtime
  guarantee.
- **Sensor scheduling changed**: If future demos add cameras or RTX sensors,
  re-test sensor timing against physics stepping.
- **ROS 2 OmniGraph nodes changed**: ROS 2 joint state and transform publishing
  should use the migrated sensor/source-node path rather than deprecated
  direct prim inputs.
- **Newton backend is not the demo default**: Keep the initial DAM demo on the
  PhysX path. Only evaluate Newton after the PhysX demo is stable.

## Go / No-Go Checklist

- [ ] `make test` passes in `/tmp/isaac_lab_study`.
- [ ] Isaac runtime host confirms Isaac Sim and Isaac Lab versions.
- [ ] `scripts/dam_scripted_comparison_demo.py --mode compare` launches and
      records the RAW vs DAM comparison.
- [ ] `scripts/dam_safety_demo.py --controller ik --num_envs 1` launches.
- [ ] `scripts/dam_teleoperate_demo.py --controller ik` launches with the
      leader-arm input path or a documented substitute.
- [ ] Demo records at least one non-`PASS` DAM decision.
- [ ] No command reaches Isaac without passing through DAM wrapper output.
- [ ] Failure cases stop with clear wrapper errors, not low-level tensor or
      controller exceptions.
