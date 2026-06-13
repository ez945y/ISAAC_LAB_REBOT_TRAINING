# ISAAC_LAB_ROBOT_TRAINING

A workspace for robot simulation, reinforcement learning, and imitation learning using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

## Repository Structure

```
ISAAC_LAB_ROBOT_TRAINING/
├── scripts/                    # Standalone experiment scripts
├── projects/
│   ├── rl_direct/              # Reinforcement Learning (Direct env)
│   ├── rl_manager/             # Reinforcement Learning (Manager-based env)
│   └── so_arm_mimic/           # Imitation Learning (Isaac Mimic)
└── tools/
    ├── controll_scripts/       # Shared robot control library & USD assets
    └── contoller_client/       # Leader arm teleoperation client (Mac side)
```

## Requirements

- **OS**: Ubuntu 24.04
- **Python**: 3.11
- **NVIDIA Isaac Sim**: 5.1+
- **Isaac Lab**: 2.3.0+

## Installation

```bash
git clone https://github.com/ez945y/ISAAC_LAB_REBOT_TRAINING.git
cd ISAAC_LAB_REBOT_TRAINING
pip install -e .
```

### Isaac Lab Runtime Dependencies

Install this repository inside an existing Isaac Sim / Isaac Lab environment.
The project intentionally does not install `torch` or `numpy` for you, because
those packages must match the simulator runtime.

For Isaac Sim 5.1 + Isaac Lab 2.3, keep the runtime on the guarded versions in:

```bash
constraints/isaaclab-2.3-isaacsim-5.1.txt
```

If a previous install upgraded PyTorch, restore the simulator-compatible stack
before launching demos:

```bash
python -m pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

python -m pip install --force-reinstall \
  -c constraints/isaaclab-2.3-isaacsim-5.1.txt \
  "numpy<2" packaging==23.0
```

Then install this repo without changing simulator dependencies:

```bash
python -m pip install -e . --no-deps
```

## Testing

```bash
make test
```

The test target runs `pytest` with `PYTHONPATH=tools`. Override the interpreter
when needed, for example:

```bash
make test PYTHON=/path/to/venv/bin/python
```

## Projects

| Project | Approach | Robot | Task |
|---------|----------|-------|------|
| [rl_direct](./projects/rl_direct) | Direct RL (PPO) | Robotic Hand (12 DOF) | Locomotion & balance |
| [rl_manager](./projects/rl_manager) | Manager-based RL (PPO) | Unitree A1 | Waypoint navigation |
| [so_arm_mimic](./projects/so_arm_mimic) | Imitation Learning | SO-ARM-101 | Pick & place |

## Standalone Scripts

Quick prototyping scripts in [`scripts/`](./scripts/) covering Isaac Lab basics: scene setup, keyboard control, physics demos, trajectory recording, robot teleoperation, and dataset replay.

For the current DAM/Isaac safety integration branch, see the
[DAM + Isaac demo readiness plan](./docs/dam_isaac_demo_plan.md).

## Tools

- **[controll_scripts](./tools/controll_scripts/)** — SO-ARM-101 control modules: IK/OSC controllers, input devices, and USD assets
- **[contoller_client](./tools/contoller_client/)** — Leader arm teleoperation client (streams joint positions via socket)

## License

BSD-3-Clause
