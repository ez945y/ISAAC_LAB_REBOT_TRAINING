from __future__ import annotations

import sys
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "tools" / "controll_scripts"
SAFETY_DIR = PACKAGE_DIR / "safety"

controll_pkg = ModuleType("controll_scripts")
controll_pkg.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("controll_scripts", controll_pkg)
safety_pkg = ModuleType("controll_scripts.safety")
safety_pkg.__path__ = [str(SAFETY_DIR)]
sys.modules.setdefault("controll_scripts.safety", safety_pkg)


def _load_safety_module(module_name: str):
    spec = util.spec_from_file_location(
        f"controll_scripts.safety.{module_name}",
        SAFETY_DIR / f"{module_name}.py",
    )
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


isaac_resolver = _load_safety_module("isaac_resolver")
dam_wrapper = _load_safety_module("dam_wrapper")

DAMSafetyWrapper = dam_wrapper.DAMSafetyWrapper
dam_xyzw_to_isaac_wxyz = isaac_resolver.dam_xyzw_to_isaac_wxyz
isaac_wxyz_to_dam_xyzw = isaac_resolver.isaac_wxyz_to_dam_xyzw


class _FakeRobotConfig:
    arm_joint_names = ["j0", "j1"]
    gripper_joint_name = "gripper"


class _Decision:
    name = "PASS"


class _FakeJointGuard:
    def __init__(self, *args, **kwargs) -> None:
        self.last_results = [SimpleNamespace(decision=_Decision())]

    def __call__(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        safe = action.clone()
        safe[2] = 0.25
        return safe

    def close(self) -> None:
        pass


class _FakeDam:
    SafetyGuard = _FakeJointGuard


class _FakeResolver:
    current_ee_pose_dam = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])

    def __init__(self) -> None:
        self.last_safe_joint_positions = None
        self.gripper_target = None

    def set_gripper_target(self, gripper_target: float | None) -> None:
        self.gripper_target = gripper_target


class _FakeEEGuard:
    def __init__(self, resolver: _FakeResolver) -> None:
        self.resolver = resolver
        self.last_results = [SimpleNamespace(decision=_Decision())]
        self.ee_pose = None

    def set_ee_pose(self, ee_pose) -> None:
        self.ee_pose = ee_pose

    def __call__(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        assert action.shape == (7,)
        assert obs.shape == (3,)
        self.resolver.last_safe_joint_positions = np.array([0.4, 0.5, 0.6])
        return torch.zeros(7, dtype=action.dtype, device=action.device)

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def fake_dam_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "dam", _FakeDam())


def test_quaternion_conventions_round_trip() -> None:
    dam_pose = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9])

    isaac_pose = dam_xyzw_to_isaac_wxyz(dam_pose)

    np.testing.assert_allclose(isaac_pose, [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(isaac_wxyz_to_dam_xyzw(isaac_pose), dam_pose)


def test_joint_filter_exposes_validated_gripper_target() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")

    safe_arm = wrapper.filter(
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.0, 0.0]),
        gripper_action=0.9,
        gripper_obs=0.0,
    )

    torch.testing.assert_close(safe_arm, torch.tensor([1.0, 2.0]))
    assert wrapper.last_safe_gripper == pytest.approx(0.25)


def test_ee_filter_returns_validated_joint_target_from_resolver() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")
    resolver = _FakeResolver()
    wrapper._ee_resolver = resolver
    wrapper._ee_guard = _FakeEEGuard(resolver)

    safe_arm = wrapper.filter_ee(
        torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        gripper_action=0.8,
        gripper_obs=0.0,
    )

    torch.testing.assert_close(safe_arm, torch.tensor([0.4, 0.5]))
    assert resolver.gripper_target == pytest.approx(0.8)
    assert wrapper.last_safe_gripper == pytest.approx(0.6)
