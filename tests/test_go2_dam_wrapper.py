"""Unit tests for the Go2 squad DAM inter-dog distance guard.

Pure-logic (no Isaac): the HolonomicSolver rollout and the Go2DAMWrapper distance
band both run headless. Skipped if the local DAM package isn't installed.
"""

from __future__ import annotations

import math

import pytest
import torch

dam = pytest.importorskip("dam")  # noqa: F841 -- gate the whole module on DAM

from controll_scripts.safety import Go2DAMWrapper, HolonomicSolver  # noqa: E402


# -- HolonomicSolver ----------------------------------------------------------

def test_holonomic_forward_strafe_and_rotation():
    s = HolonomicSolver()
    # forward 1 m/s for 2 s, facing +x -> +2 x
    fwd = s.rollout([0, 0, 0.0], [1, 0, 0], dt=2.0)
    assert fwd[0].item() == pytest.approx(2.0, abs=1e-4)
    assert fwd[1].item() == pytest.approx(0.0, abs=1e-4)
    # strafe (vy) -> +y, no forward
    strafe = s.rollout([0, 0, 0.0], [0, 1, 0], dt=2.0)
    assert strafe[1].item() == pytest.approx(2.0, abs=1e-4)
    # facing +90deg, forward -> +y
    turned = s.rollout([0, 0, math.pi / 2], [1, 0, 0], dt=2.0)
    assert turned[1].item() == pytest.approx(2.0, abs=1e-4)


def test_holonomic_rollout_is_differentiable():
    s = HolonomicSolver()
    cmd = torch.tensor([0.5, 0.3, 0.1], requires_grad=True)
    nxt = s.rollout(torch.tensor([0.0, 0.0, 0.3]), cmd, dt=0.2)
    nxt[0].backward()
    assert cmd.grad is not None
    assert torch.isfinite(cmd.grad).all()
    assert cmd.grad.abs().sum() > 0


# -- Go2DAMWrapper distance band ----------------------------------------------

@pytest.fixture
def guard():
    return Go2DAMWrapper("go2_squad_safety.yaml")  # band [0.8, 4.0], dog at origin facing +x


def _filter(guard, cmd, neighbors, pose=(0.0, 0.0, 0.0)):
    return guard.filter(torch.tensor([cmd]), torch.tensor([pose]), neighbors)[0].tolist()


def test_in_band_passes_through(guard):
    safe = _filter(guard, [1.0, 0.0, 0.0], [(2.0, 0.0)])
    assert safe == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)
    assert guard.last_decision == "PASS"


def test_lone_dog_passes_through(guard):
    safe = _filter(guard, [1.0, 0.0, 0.0], [])
    assert safe == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)


def test_too_close_backs_off(guard):
    # neighbour 0.6 m straight ahead (< min 0.8) -> forward command reversed
    safe = _filter(guard, [1.0, 0.0, 0.0], [(0.6, 0.0)])
    assert safe[0] < 0.0
    assert guard.last_decision in {"CLAMP", "REJECT"}


def test_front_neighbour_triggers_sidestep(guard):
    # neighbour front-left -> holonomic guard strafes RIGHT (vy < 0) instead of only braking
    safe = _filter(guard, [1.0, 0.0, 0.0], [(0.5, 0.4)])
    assert safe[1] < -0.1


def test_too_far_pulls_in(guard):
    # nearest neighbour 6 m ahead (> max 4.0), idle -> drive toward it (+x)
    safe = _filter(guard, [0.0, 0.0, 0.0], [(6.0, 0.0)])
    assert safe[0] > 0.5


def test_far_lateral_neighbour_pulls_via_strafe(guard):
    # neighbour 6 m to the left, idle -> strafe left (vy > 0) to close the cohesion gap
    safe = _filter(guard, [0.0, 0.0, 0.0], [(0.0, 6.0)])
    assert safe[1] > 0.5
