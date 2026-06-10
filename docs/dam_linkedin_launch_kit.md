# DAM LinkedIn Launch Kit

Use this after recording `scripts/dam_scripted_comparison_demo.py --mode compare`.
The goal is to make the value obvious to robotics, simulation, and AI policy
teams in the first 3 seconds.

## 30-Second Video Structure

### 0-3s: Hook

On-screen text:

```text
Same command. Safer robot.
```

Visual:

- Split or sequential title card.
- Left label: `RAW COMMAND`
- Right label: `DAM ON`

### 3-12s: Show The Problem

Visual:

- RAW segment from the scripted replay.
- Keep the target marker visible.
- Show terminal line or overlay with `RAW`, target offset, and tracking error.

Narration:

```text
Robot policies and teleop streams can produce unsafe targets faster than a human can inspect them.
```

### 12-24s: Show The Safety Boundary

Visual:

- DAM segment from the same replay.
- Show `PASS`, `CLAMP`, `REJECT`, or `FAULT` status.
- Show the validated robot motion, not just a terminal log.

Narration:

```text
DAM sits between the command and Isaac, validates every target, and only passes through safe joint commands.
```

### 24-30s: Proof + CTA

Use the `LINKEDIN DEMO SUMMARY` block from the script.

On-screen text:

```text
Risky frames: <N>
DAM interventions: <N>
Controller rewrite: 0
```

CTA:

```text
If you are testing robot policies in simulation, safety should be a runtime boundary, not a post-mortem chart.
```

## Post Draft

```text
Same command. Safer robot.

I built a scripted Isaac Sim demo that replays the same risky end-effector command twice:

1. RAW COMMAND: the controller receives the target directly.
2. DAM ON: the same target is filtered by a runtime safety layer before Isaac receives joint commands.

Why this matters:
Robot policies and teleoperation streams can generate unsafe targets faster than humans can inspect them. DAM turns safety rules into a live control boundary: every target is validated before it reaches the simulated robot.

In this run:
- Risky command frames: <from LINKEDIN DEMO SUMMARY>
- DAM interventions: <from LINKEDIN DEMO SUMMARY>
- Controller rewrite required: 0
- Isaac scene rewrite required: 0

The point is not just avoiding one bad motion. It is making safety observable, testable, and reusable while iterating on robot behavior.

Next step: run this against richer policy outputs and record how often safety constraints intervene before deployment.
```

## Caption Variants

- Same command, safer robot.
- Safety should sit in the control path, not only in the post-run chart.
- A policy can be creative. The robot still needs boundaries.
- Runtime safety for robot learning loops: visible, testable, reusable.

## What To Avoid

- Do not open with architecture diagrams.
- Do not spend the first 10 seconds explaining Pinocchio, IK, or tensor shapes.
- Do not show only terminal output. The robot motion must carry the story.
- Do not claim Isaac Sim 6.0 support unless that exact runtime was tested.

## Recording Checklist

- [ ] Run `make test` first.
- [ ] Record `python scripts/dam_scripted_comparison_demo.py --mode compare --summary-path outputs/dam_linkedin_summary.md`.
- [ ] Capture the terminal around `RAW COMMAND`, `DAM ON`, and `LINKEDIN DEMO SUMMARY`.
- [ ] Keep the target marker visible.
- [ ] Show at least one non-`PASS` DAM decision, or tune `--unsafe-scale` / stackfile before publishing.
- [ ] Save Isaac Sim, Isaac Lab, GPU, driver, and Python versions.

## Product Message

For robotics teams, the value is speed with guardrails:

- Keep iterating on policies and teleop behavior.
- Do not rewrite the controller for each safety experiment.
- Make interventions measurable.
- Keep the safety story understandable to non-authors of the code.
