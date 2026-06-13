# DAM LinkedIn Launch Kit

Use this after recording either comparison demo:

```bash
python scripts/09_dam_car_scripted_comparison_demo.py --summary-path outputs/car_linkedin_summary.md
python scripts/10_dam_scripted_comparison_demo.py --mode compare --summary-path outputs/dam_linkedin_summary.md
```

For a general LinkedIn audience, lead with the car/Jetbot version. The side-by-side
lanes make the RAW vs DAM intervention legible even to viewers who do not know
robot arms.

The goal is to make the value obvious to robotics, simulation, and AI policy
teams in the first 3 seconds.

## 30-Second Video Structure

### 0-3s: Hook

On-screen text:

```text
Same command. Safer robot.
```

For the car version, prefer:

```text
Same command. Safer vehicle.
```

Visual:

- Split/twin-lane title card.
- Left label: `RAW COMMAND`
- Right label: `DAM GUARDED`

### 3-12s: Show The Problem

Visual:

- RAW lane from the scripted replay.
- For the car version, keep both lanes and the red gate visible.
- For the arm version, keep the target marker visible.
- Show terminal line or overlay with `RAW`, target offset, and tracking error.

Narration:

```text
Robot policies and teleop streams can produce unsafe targets faster than a human can inspect them.
```

### 12-24s: Show The Safety Boundary

Visual:

- DAM lane from the same synchronized replay.
- Show `PASS`, `CLAMP`, `REJECT`, or `FAULT` status.
- Show the validated robot motion, not just a terminal log.
- For the car version, show `CLAMP`, `REJECT`, or `FAULT` near the gate.

Narration:

```text
DAM sits between the command and Isaac, validates every wheel target, and only passes through SafetyGuard output.
```

### 24-30s: Proof + CTA

Use the `LINKEDIN DEMO SUMMARY` block from the script.

On-screen text:

```text
Guard interventions: <N>
Controller rewrite: 0
```

CTA:

```text
If you are testing robot policies in simulation, safety should be a runtime boundary, not a post-mortem chart.
```

## Post Draft

```text
Same command. Safer vehicle.

I built a scripted Isaac Sim demo that runs the same risky vehicle command in two lanes at once:

1. RAW COMMAND: Isaac receives the wheel target directly.
2. DAM GUARDED: the same target is filtered by `dam.SafetyGuard` before Isaac receives wheel commands.

Why this matters:
Robot policies and teleoperation streams can generate unsafe speed commands faster than humans can inspect them. A runtime action monitor can clamp or reject the command before it reaches the simulated robot.

In this run:
- DAM interventions: <from TWIN-LANE DAM DEMO SUMMARY>
- Max DAM correction: <from TWIN-LANE DAM DEMO SUMMARY>
- Controller rewrite required: 0
- Isaac scene rewrite required: 0

The point is not just avoiding one bad motion. It is making safety observable, testable, and reusable while iterating on robot behavior.

Next step: run this against richer policy outputs and record how often safety constraints intervene before deployment.
```

## Caption Variants

- Same command, safer robot.
- Same command, safer vehicle.
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
- [ ] Record `python scripts/09_dam_car_scripted_comparison_demo.py --summary-path outputs/car_linkedin_summary.md`.
- [ ] Optionally record the arm version with `python scripts/10_dam_scripted_comparison_demo.py --mode compare --summary-path outputs/dam_linkedin_summary.md`.
- [ ] Capture the terminal around DAM `PASS` / `CLAMP` / `REJECT` decisions and `TWIN-LANE DAM DEMO SUMMARY`.
- [ ] For the car version, capture both lanes and the red gate in the same frame.
- [ ] Show at least one `CLAMP`, `REJECT`, or `FAULT`, or tune `--unsafe-speed` / `--turn-command` before publishing.
- [ ] Save Isaac Sim, Isaac Lab, GPU, driver, and Python versions.

## Product Message

For robotics teams, the value is speed with guardrails:

- Keep iterating on policies and teleop behavior.
- Do not rewrite the controller for each safety experiment.
- Make interventions measurable.
- Keep the safety story understandable to non-authors of the code.
