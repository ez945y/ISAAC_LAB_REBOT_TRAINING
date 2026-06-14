# Control Scripts Library

A modular robot control library for Isaac Lab, providing unified interfaces for controllers, robot configurations, and input devices.

## Installation

Install this repository as a package so subpackages and bundled assets are
available at runtime:

```bash
pip install -e .
```

For quick local experiments, you can still add the repo tools directory to your
Python path:

```python
import sys
sys.path.insert(0, "/path/to/ISAAC_LAB_ROBOT_TRAINING/tools")

from controll_scripts import (
    ControllerFactory,
    ControllerType,
    KeyboardInputDevice,
    LeaderArmInputDevice,
    SOArm101Config,
)
```

## Module Structure

```
controll_scripts/
├── __init__.py           # Main exports
├── controllers/          # Robot controllers
│   ├── base.py          # BaseController abstract class
│   ├── ik_controller.py # Differential IK controller
│   ├── osc_controller.py# Operational Space controller
│   └── factory.py       # ControllerFactory
├── configs/             # Robot configurations
│   ├── base.py          # BaseRobotConfig
│   └── so_arm_101.py    # SOArm101Config
├── input_devices/       # Input device interfaces
│   ├── base.py          # BaseInputDevice abstract class
│   ├── keyboard.py      # KeyboardInputDevice
│   └── leader_arm.py    # LeaderArmInputDevice (socket-based)
├── safety/              # DAM safety wrapper and bundled stackfile
│   ├── dam_wrapper.py   # Joint/EE action filtering for Isaac loops
│   └── isaac_resolver.py# Isaac IK + Pinocchio FK resolver bridge
└── so_arm_101/          # SO-ARM-101 robot assets
    ├── SO-ARM101.usd    # USD robot model
    └── description/     # URDF and other descriptions
```

## Components

### Controllers

#### ControllerFactory
Factory class to create controllers.

```python
from controll_scripts import ControllerFactory, ControllerType

controller = ControllerFactory.create(
    controller_type=ControllerType.IK,
    robot=robot,
    robot_config=robot_config,
    device="cuda:0",
    num_envs=1,
)
```

#### ControllerType
Enum for controller types:
- `ControllerType.IK` - Differential Inverse Kinematics
- `ControllerType.OSC` - Operational Space Control

### Robot Configurations

#### SOArm101Config
Configuration for the SO-ARM-101 robot.

```python
from controll_scripts import SOArm101Config

robot_config = SOArm101Config()
articulation_cfg = robot_config.get_articulation_cfg(for_osc=False)
```

### Input Devices

All input devices implement the `BaseInputDevice` interface:

```python
def update(self) -> Tuple[torch.Tensor, float, bool]:
    """Returns (target_pose, gripper_pos, reset_requested)"""
    
def reset_target(self, pose: torch.Tensor) -> None:
    """Reset target pose"""
    
def sync_to_actual(self, actual_pose: torch.Tensor) -> None:
    """Sync target to actual pose when unreachable"""
```

#### KeyboardInputDevice
Keyboard control for robot manipulation.

```python
from controll_scripts import KeyboardInputDevice

input_device = KeyboardInputDevice(
    initial_pose=controller.current_ee_pose,
    device="cuda:0",
)
```

**Controls:**
| Key | Action |
|-----|--------|
| W/A/S/D | Move in XY plane |
| Q/E | Move up/down (Z axis) |
| Z/X | Roll rotation |
| T/G | Pitch rotation |
| C/V | Yaw rotation |
| B | Open gripper |
| N | Close gripper |
| L | Reset scene |

#### LeaderArmInputDevice
Socket-based input from a physical leader arm.

```python
from controll_scripts import LeaderArmInputDevice

input_device = LeaderArmInputDevice(
    initial_pose=controller.current_ee_pose,
    device="cuda:0",
    socket_host="127.0.0.1",
    socket_port=5359,
)
```

This device receives end-effector pose data from a socket server (e.g., `lerobot-teleoperate_port`).

**Expected JSON format:**
```json
{
    "x": 0.15,
    "y": 0.05,
    "z": 0.20,
    "qw": 1.0,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
    "gripper": 50.0
}
```

### DAM Safety

`DAMSafetyWrapper` filters SO-ARM-101 commands through a local `dam.SafetyGuard`
before they reach Isaac simulation. It currently supports one Isaac environment
at a time; `scripts/11_dam_safety_demo.py` rejects `--num_envs` values other than
`1` to avoid silently filtering only the first environment.

Install or expose the local DAM package that provides `dam.SafetyGuard`, then
use the bundled stackfile:

```python
from controll_scripts.safety import DAMSafetyWrapper

dam = DAMSafetyWrapper(
    stackfile="soarm_isaac_safety.yaml",
    robot_config=robot_config,
    device=sim.device,
)

safe_arm_targets = dam.filter(
    joint_pos_des,
    current_joint_pos,
    gripper_action=gripper_target,
    gripper_obs=current_gripper,
)
robot.set_joint_position_target_index(target=safe_arm_targets, joint_ids=arm_joint_ids)
robot.set_joint_position_target_index(
    target=torch.tensor([[dam.last_safe_gripper]], device=sim.device),
    joint_ids=gripper_joint_ids,
)
```

For EE-space targets, attach the live Isaac controller once. The wrapper uses
Isaac's IK controller to propose joints, validates those joints through DAM, and
uses the bundled SO-ARM-101 URDF with Pinocchio for FK:

```python
dam.attach_isaac_controller(robot, controller, robot_config)
safe_arm_targets = dam.filter_ee(
    target_pose,  # Isaac pose [x, y, z, qw, qx, qy, qz]
    current_joint_pos,
    gripper_action=gripper_target,
    gripper_obs=current_gripper,
)
```

`last_decision` reports the most severe DAM outcome from the last call using
`FAULT > REJECT > CLAMP > PASS`; `last_clamped` remains a boolean for whether
any guard clamped the command.

## Usage Example

```python
from controll_scripts import (
    ControllerFactory,
    ControllerType,
    KeyboardInputDevice,
    SOArm101Config,
)

# Create robot configuration
robot_config = SOArm101Config()

# Create controller
controller = ControllerFactory.create(
    controller_type=ControllerType.IK,
    robot=robot,
    robot_config=robot_config,
    device=sim.device,
    num_envs=1,
)

# Create input device
input_device = KeyboardInputDevice(
    initial_pose=controller.current_ee_pose,
    device=sim.device,
)

# Control loop
while simulation_app.is_running():
    target_pose, gripper_pos, reset_requested = input_device.update()
    
    if reset_requested:
        controller.reset()
        input_device.reset_target(controller.current_ee_pose)
        continue
    
    controller.compute(target_pose, gripper_pos)
    sim.step()
```
