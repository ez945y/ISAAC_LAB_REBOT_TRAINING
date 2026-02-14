# Manager-based RL — A1 Waypoint Navigation

Unitree A1 quadruped waypoint navigation using **Manager-based RL** environment.

## Features

- Waypoint navigation through a 4-point sequence
- 360° spinning LiDAR for obstacle detection
- Foot contact sensors for gait analysis
- Custom rewards: position progress, heading alignment, velocity tracking, waypoint bonus
- Domain randomization (mass) for sim-to-real transfer

## Task

| Task ID | Robot | Description |
|---------|-------|-------------|
| `Isaac-Waypoint-Navigation-v0` | Unitree A1 | Navigate through 4 waypoints with LiDAR |

## Usage

```bash
cd projects/rl_manager

# Train
python scripts/rsl_rl/train.py --task=Isaac-Waypoint-Navigation-v0

# Play trained policy
python scripts/rsl_rl/play.py --task=Isaac-Waypoint-Navigation-v0

# Test with zero / random actions
python scripts/zero_agent.py --task=Isaac-Waypoint-Navigation-v0
python scripts/random_agent.py --task=Isaac-Waypoint-Navigation-v0

# List all registered environments
python scripts/list_envs.py
```

## Structure

```
source/rl_manager/rl_manager/tasks/manager_based/navigation/
├── __init__.py               # Task registration
├── navigation_env_cfg.py     # Environment config
├── mdp/
│   ├── rewards.py            # Custom reward functions
│   └── command.py            # Waypoint command generator
├── assets/
│   ├── navigation.py         # Robot & scene config
│   └── wall.usd              # Wall obstacle model
└── agents/
    └── rsl_rl_ppo_cfg.py     # PPO training config
```

## Configuration

### Environment (`navigation_env_cfg.py`)

| Component | Details |
|-----------|---------|
| Scene | Ground plane, A1 robot, LiDAR, contact sensors, walls |
| Actions | Joint position control (scale 0.25) |
| Observations | Goal distance, heading error, base velocity, joint states |
| Rewards | Alive bonus, velocity tracking, position/heading alignment, waypoint bonus |
| Terminations | Timeout, illegal body contact |

### Training (`rsl_rl_ppo_cfg.py`)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (adaptive LR) |
| Actor | [512, 256, 128] |
| Critic | [1024, 512, 256] |
| Max iterations | 1500 |
| Steps per env | 24 |

### Waypoints

Default sequence (configurable):
1. `(2.8, 0.0)` → `(2.8, -4.3)` → `(-1.6, -4.3)` → `(-1.6, 0.0)`