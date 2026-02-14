# Direct RL — Hand Locomotion

Robotic hand locomotion and balance control using **Direct RL** environment. Modified from the Isaac Lab Ant locomotion task.

## Task

| Task ID | Robot | DOF | Obs Dim |
|---------|-------|-----|---------|
| `Isaac-Hand-Direct-v0` | H1 Hand (Left) | 12 | 48 |

## Usage

```bash
cd projects/rl_direct

# Train
python scripts/rsl_rl/train.py --task=Isaac-Hand-Direct-v0

# Play trained policy
python scripts/rsl_rl/play.py --task=Isaac-Hand-Direct-Play-v0

# Test with zero / random actions
python scripts/zero_agent.py --task=Isaac-Hand-Direct-v0
python scripts/random_agent.py --task=Isaac-Hand-Direct-v0
```

## Structure

```
source/rl_direct/rl_direct/tasks/direct/hand/
├── hand_env.py              # Environment implementation
├── assets/
│   ├── hand_cfg.py          # Robot config
│   └── h1_hand_left.usd     # Robot USD model
└── agents/
    └── rsl_rl_ppo_cfg.py    # PPO training config
```