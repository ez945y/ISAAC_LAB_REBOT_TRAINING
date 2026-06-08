# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Robot Control Library - 機器人控制庫

提供統一的控制器介面，支持多種控制器類型和機器人配置。
"""

from .controllers import BaseController, IKController, OSCController, ControllerFactory, ControllerType
from .configs import BaseRobotConfig, SOArm101Config
from .input_devices import (
    BaseInputDevice,
    KeyboardInputDevice,
    LeaderArmInputDevice,
    Se3LeaderArm,
    Se3LeaderArmCfg,
)

__all__ = [
    # Controllers
    "BaseController",
    "IKController",
    "OSCController",
    "ControllerFactory",
    "ControllerType",
    # Configs
    "BaseRobotConfig",
    "SOArm101Config",
    # Input Devices
    "BaseInputDevice",
    "KeyboardInputDevice",
    "LeaderArmInputDevice",
    # Isaac Lab DeviceBase compatible
    "Se3LeaderArm",
    "Se3LeaderArmCfg",
    # Safety (requires local DAM package exposing dam.SafetyGuard)
    # from controll_scripts.safety import DAMSafetyWrapper
]
