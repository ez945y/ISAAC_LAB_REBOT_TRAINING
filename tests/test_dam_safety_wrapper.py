from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from controll_scripts.safety import DAMSafetyWrapper
from controll_scripts.safety.isaac_resolver import (
    IsaacControllerKinematicsResolver,
    _PinocchioForwardKinematics,
    default_so101_urdf_path,
    dam_xyzw_to_isaac_wxyz,
    isaac_wxyz_to_dam_xyzw,
)


class _FakeRobotConfig:
    name = "fake"
    arm_joint_names = ["j0", "j1"]
    gripper_joint_name = "gripper"
    ee_body_name = "gripper"


class _FakeSO101Config:
    name = "SO-ARM-101"
    arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    gripper_joint_name = "gripper"
    ee_body_name = "gripper_link"


class _Decision:
    def __init__(self, name: str = "PASS") -> None:
        self.name = name


class _FakeJointGuard:
    stackfiles: list[str] = []
    seen_action_dtype = None
    seen_obs_dtype = None
    decision_names = ["PASS"]
    safe_override = None

    def __init__(self, *args, **kwargs) -> None:
        self.stackfiles.append(str(args[0]))
        self.last_results = [
            SimpleNamespace(decision=_Decision(name))
            for name in self.decision_names
        ]

    def __call__(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        self.seen_action_dtype = action.dtype
        self.seen_obs_dtype = obs.dtype
        if self.safe_override is not None:
            return torch.as_tensor(self.safe_override, dtype=action.dtype, device=action.device)
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


class _FakeNoFkEEGuard:
    last_results = [SimpleNamespace(decision=_Decision())]

    def set_ee_pose(self, ee_pose) -> None:
        pass

    def __call__(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(7, dtype=action.dtype, device=action.device)

    def close(self) -> None:
        pass


class _FakeShortTargetEEGuard:
    last_results = [SimpleNamespace(decision=_Decision())]

    def __init__(self, resolver: _FakeResolver) -> None:
        self.resolver = resolver

    def set_ee_pose(self, ee_pose) -> None:
        pass

    def __call__(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        self.resolver.last_safe_joint_positions = np.array([0.4, 0.5])
        return torch.zeros(7, dtype=action.dtype, device=action.device)

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def fake_dam_module(monkeypatch):
    _FakeJointGuard.stackfiles = []
    _FakeJointGuard.seen_action_dtype = None
    _FakeJointGuard.seen_obs_dtype = None
    _FakeJointGuard.decision_names = ["PASS"]
    _FakeJointGuard.safe_override = None
    monkeypatch.setitem(sys.modules, "dam", _FakeDam())


def test_quaternion_conventions_round_trip() -> None:
    dam_pose = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9])

    isaac_pose = dam_xyzw_to_isaac_wxyz(dam_pose)

    np.testing.assert_allclose(isaac_pose, [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(isaac_wxyz_to_dam_xyzw(isaac_pose), dam_pose)


def test_default_so101_fk_matches_bundled_urdf_conventions() -> None:
    fk = _PinocchioForwardKinematics(
        default_so101_urdf_path(),
        ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        "gripper",
    )

    pose = fk.compute(np.zeros(5))

    assert pose.shape == (7,)
    assert fk.model.nq == 5
    np.testing.assert_allclose(pose[:3], [0.02061531, -0.27747322, 0.26685203], atol=1e-7)
    assert np.linalg.norm(pose[3:]) == pytest.approx(1.0)


def test_so101_resolver_maps_isaac_names_to_bundled_urdf() -> None:
    resolver = IsaacControllerKinematicsResolver(
        robot=SimpleNamespace(),
        controller=SimpleNamespace(),
        robot_config=_FakeSO101Config(),
        device="cpu",
    )

    pose = resolver.forward_kinematics(np.zeros(6))

    assert resolver._fk.model.nq == 5
    np.testing.assert_allclose(pose[:3], [0.02061531, -0.27747322, 0.26685203], atol=1e-7)
    assert resolver.last_safe_joint_positions.shape == (6,)


def test_so101_resolver_rejects_short_validated_joint_vector_without_updating_cache() -> None:
    resolver = IsaacControllerKinematicsResolver(
        robot=SimpleNamespace(),
        controller=SimpleNamespace(),
        robot_config=_FakeSO101Config(),
        device="cpu",
    )
    resolver.last_safe_joint_positions = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="expected at least 5 joints"):
        resolver.forward_kinematics(np.zeros(4))

    np.testing.assert_allclose(resolver.last_safe_joint_positions, [1.0, 2.0, 3.0])


def test_fk_rejects_missing_joint_names_and_wrong_vector_size() -> None:
    with pytest.raises(ValueError, match="missing from FK URDF"):
        _PinocchioForwardKinematics(default_so101_urdf_path(), ["shoulder_pan"], "gripper")

    fk = _PinocchioForwardKinematics(
        default_so101_urdf_path(),
        ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        "gripper",
    )
    with pytest.raises(ValueError, match="expected 5 joint positions"):
        fk.compute(np.zeros(6))


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


def test_joint_filter_rejects_multi_env_batches() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")

    with pytest.raises(ValueError, match="one Isaac environment"):
        wrapper.filter(torch.zeros(2, 2), torch.zeros(2, 2))


def test_joint_filter_rejects_wrong_arm_width() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")

    with pytest.raises(ValueError, match="action width 2"):
        wrapper.filter(torch.zeros(3), torch.zeros(2))


def test_joint_filter_preserves_input_dtype_for_full_guard_vectors() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")

    safe_arm = wrapper.filter(
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        gripper_action=0.9,
        gripper_obs=0.0,
    )

    assert safe_arm.dtype == torch.float64
    assert wrapper._guard.seen_action_dtype == torch.float64
    assert wrapper._guard.seen_obs_dtype == torch.float64


def test_joint_filter_rejects_incomplete_guard_joint_target() -> None:
    _FakeJointGuard.safe_override = [0.1, 0.2]
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")

    with pytest.raises(RuntimeError, match="DAM joint guard.*expected 3 arm\\+gripper joints, got 2"):
        wrapper.filter(torch.zeros(2), torch.zeros(2))


def test_joint_filter_rejects_vector_gripper_action() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")

    with pytest.raises(ValueError, match="gripper_action.*single-element tensor"):
        wrapper.filter(
            torch.zeros(2),
            torch.zeros(2),
            gripper_action=torch.zeros(2),
        )


def test_joint_filter_reports_reject_over_clamp() -> None:
    _FakeJointGuard.decision_names = ["CLAMP", "REJECT"]
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")

    wrapper.filter(torch.zeros(2), torch.zeros(2))

    assert wrapper.last_clamped is True
    assert wrapper.last_decision == "REJECT"
    assert wrapper.clamp_rate == pytest.approx(1.0)


def test_joint_filter_reports_fault_over_reject() -> None:
    _FakeJointGuard.decision_names = ["CLAMP", "REJECT", "FAULT"]
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")

    wrapper.filter(torch.zeros(2), torch.zeros(2))

    assert wrapper.last_clamped is True
    assert wrapper.last_decision == "FAULT"


def test_wrapper_resolves_bundled_stackfile_by_name() -> None:
    wrapper = DAMSafetyWrapper("soarm_isaac_safety.yaml", _FakeSO101Config(), "cpu")

    stackfile = Path(_FakeJointGuard.stackfiles[-1])
    assert stackfile.name == "soarm_isaac_safety.yaml"
    assert stackfile.exists()
    wrapper.close()


def test_wrapper_preserves_existing_relative_stackfile() -> None:
    wrapper = DAMSafetyWrapper(
        "tools/controll_scripts/safety/soarm_isaac_safety.yaml",
        _FakeSO101Config(),
        "cpu",
    )

    assert _FakeJointGuard.stackfiles[-1] == "tools/controll_scripts/safety/soarm_isaac_safety.yaml"
    wrapper.close()


def test_wrapper_falls_back_to_bundled_name_for_package_style_path(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    wrapper = DAMSafetyWrapper(
        "controll_scripts/safety/soarm_isaac_safety.yaml",
        _FakeSO101Config(),
        "cpu",
    )

    stackfile = Path(_FakeJointGuard.stackfiles[-1])
    assert stackfile.name == "soarm_isaac_safety.yaml"
    assert stackfile.exists()
    wrapper.close()


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


def test_ee_filter_does_not_reuse_stale_resolver_joint_target() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")
    resolver = _FakeResolver()
    resolver.last_safe_joint_positions = np.array([9.0, 9.0, 9.0])
    wrapper._ee_resolver = resolver
    wrapper._ee_guard = _FakeNoFkEEGuard()

    with pytest.raises(RuntimeError, match="did not produce validated joint positions"):
        wrapper.filter_ee(
            torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        )


def test_ee_filter_rejects_incomplete_resolver_joint_target() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")
    resolver = _FakeResolver()
    wrapper._ee_resolver = resolver
    wrapper._ee_guard = _FakeShortTargetEEGuard(resolver)

    with pytest.raises(RuntimeError, match="expected 3 arm\\+gripper joints, got 2"):
        wrapper.filter_ee(
            torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        )


def test_ee_filter_rejects_vector_gripper_obs() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")
    resolver = _FakeResolver()
    wrapper._ee_resolver = resolver
    wrapper._ee_guard = _FakeEEGuard(resolver)

    with pytest.raises(ValueError, match="gripper_obs.*single-element tensor"):
        wrapper.filter_ee(
            torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            gripper_obs=torch.zeros(2),
        )


def test_ee_filter_rejects_multi_env_batches() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")
    resolver = _FakeResolver()
    wrapper._ee_resolver = resolver
    wrapper._ee_guard = _FakeEEGuard(resolver)

    with pytest.raises(ValueError, match="one Isaac environment"):
        wrapper.filter_ee(torch.zeros(2, 7), torch.zeros(2, 2))


def test_ee_filter_rejects_wrong_pose_width() -> None:
    wrapper = DAMSafetyWrapper("safety.yaml", _FakeRobotConfig(), "cpu")
    resolver = _FakeResolver()
    wrapper._ee_resolver = resolver
    wrapper._ee_guard = _FakeEEGuard(resolver)

    with pytest.raises(ValueError, match="target_pose width 7"):
        wrapper.filter_ee(torch.zeros(6), torch.zeros(2))
