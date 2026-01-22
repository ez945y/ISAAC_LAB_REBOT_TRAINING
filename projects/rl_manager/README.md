# RL Manager-based A1 Navigation

A custom Isaac Lab extension for Unitree A1 robot waypoint navigation using Manager-based RL.

## Features

- **Waypoint Navigation**: Train A1 robot to navigate through multiple waypoints in sequence
- **360° LiDAR Sensing**: Spinning LiDAR for obstacle detection
- **Contact Sensors**: Foot contact detection for gait analysis
- **Custom Rewards**: Position progress, heading alignment, velocity tracking, and waypoint bonuses
- **Domain Randomization**: Mass randomization for robust sim-to-real transfer

## Installation

Follow the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
cd projects/rl_manager
```

## Available Tasks

| Task ID | Description |
|---------|-------------|
| `Isaac-Waypoint-Navigation-v0` | A1 robot waypoint navigation with 4 waypoints |

## Usage

### Training
```bash
# Basic training
python scripts/rsl_rl/train.py --task=Isaac-Waypoint-Navigation-v0
```

### Play Trained Policy
```bash
python scripts/rsl_rl/play.py --task=Isaac-Waypoint-Navigation--Play-v0
```

### Testing
```bash
# Zero-action agent
python scripts/zero_agent.py --task=Isaac-Waypoint-Navigation-v0

# Random-action agent
python scripts/random_agent.py --task=Isaac-Waypoint-Navigation-v0

# List all available environments
python scripts/list_envs.py
```

## Project Structure

```
rl_manager/
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py          # RSL-RL training script
│   │   └── play.py           # Policy playback script
│   ├── random_agent.py       # Random action testing
│   ├── zero_agent.py         # Zero action testing
│   └── list_envs.py          # List available environments
│
└── source/rl_manager/rl_manager/
    └── tasks/manager_based/navigation/
        ├── __init__.py           # Task registration
        ├── navigation_env_cfg.py # Environment configuration
        ├── mdp/
        │   ├── __init__.py
        │   ├── rewards.py        # Custom reward functions
        │   └── command.py        # Waypoint command generator
        ├── assets/
        │   ├── navigation.py     # Robot & scene assets config
        │   └── wall.usd          # Wall obstacle model
        └── agents/
            └── rsl_rl_ppo_cfg.py # PPO training hyperparameters
```

## Configuration

### Environment Settings (`navigation_env_cfg.py`)

- **Scene**: Ground plane, A1 robot, LiDAR, contact sensors, wall obstacles
- **Actions**: Joint position control with 0.25 scale
- **Observations**: Goal distance, heading error, base velocity, joint states
- **Rewards**: Alive bonus, velocity tracking, position/heading alignment, waypoint bonus
- **Terminations**: Timeout, illegal body contact

### Training Settings (`rsl_rl_ppo_cfg.py`)

- **Algorithm**: PPO with adaptive learning rate
- **Network**: Actor [512, 256, 128], Critic [1024, 512, 256]
- **Max Iterations**: 1500
- **Steps per Env**: 24

## Waypoints

Default waypoint sequence (configurable in `navigation_env_cfg.py`):
1. (2.8, 0.0) - Start position
2. (2.8, -4.3) - First turn
3. (-1.6, -4.3) - Second turn  
4. (-1.6, 0.0) - Return

## License

BSD-3-Clause