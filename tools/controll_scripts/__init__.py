# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Robot Control Library - 機器人控制庫

提供統一的控制器介面，支持多種控制器類型和機器人配置。
"""

_EXPORT_MODULES = {
    # Controllers
    "BaseController": ".controllers",
    "IKController": ".controllers",
    "OSCController": ".controllers",
    "ControllerFactory": ".controllers",
    "ControllerType": ".controllers",
    # Configs
    "BaseRobotConfig": ".configs",
    "SOArm101Config": ".configs",
    # Input Devices
    "BaseInputDevice": ".input_devices",
    "KeyboardInputDevice": ".input_devices",
    "LeaderArmInputDevice": ".input_devices",
    "Se3LeaderArm": ".input_devices",
    "Se3LeaderArmCfg": ".input_devices",
}

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


def __getattr__(name: str):
    """Lazy-load Isaac-dependent exports only when callers ask for them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
