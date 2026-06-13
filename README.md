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

Isaac Sim / Isaac Lab are installed first via NVIDIA's official `uv` flow, which
provisions the simulator runtime (PyTorch, NumPy, etc.). This repository is then
installed **into that same environment** — it only registers its own local
packages and intentionally does **not** install or upgrade `torch`/`numpy`,
because those must match the simulator runtime.

```bash
git clone https://github.com/ez945y/ISAAC_LAB_REBOT_TRAINING.git
cd ISAAC_LAB_REBOT_TRAINING

# Activate the venv that NVIDIA's uv flow created for Isaac Sim / Isaac Lab
# e.g. source /path/to/isaac/.venv/bin/activate
```

### Isaac Lab Runtime Dependencies

For Isaac Sim 5.1 + Isaac Lab 2.3, the runtime is guarded at the versions in
[`constraints/isaaclab-2.3-isaacsim-5.1.txt`](constraints/isaaclab-2.3-isaacsim-5.1.txt):

```text
torch==2.7.0  torchvision==0.22.0  torchaudio==2.7.0  (cu128)
numpy<2  packaging==23.0
```

Install PyTorch **only if it is not already present** — the import guard leaves
the simulator's existing stack untouched and never upgrades it:

```bash
python -c "import torch" 2>/dev/null || \
  uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

> Do not pass `-U`/`--upgrade` for torch: that forces a re-resolve and can
> overwrite the simulator-matched build. If a previous install already broke the
> stack, restore it explicitly:
>
> ```bash
> uv pip install --reinstall \
>   --index-url https://download.pytorch.org/whl/cu128 \
>   torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
> uv pip install --reinstall \
>   -c constraints/isaaclab-2.3-isaacsim-5.1.txt \
>   "numpy<2" packaging==23.0
> ```

### DAM Safety Wrapper

The demos depend on the `robot-dam` package. Install it **only if it is not
already installed**, so re-running setup never reinstalls it:

```bash
uv pip show robot-dam >/dev/null 2>&1 || uv pip install robot-dam
```

### Install this repository

Finally, install this repo without touching any simulator dependencies:

```bash
uv pip install -e . --no-deps
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
