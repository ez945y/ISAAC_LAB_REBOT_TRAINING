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

## Tools

- **[controll_scripts](./tools/controll_scripts/)** — SO-ARM-101 control modules: IK/OSC controllers, input devices, and USD assets
- **[contoller_client](./tools/contoller_client/)** — Leader arm teleoperation client (streams joint positions via socket)

## License

BSD-3-Clause
