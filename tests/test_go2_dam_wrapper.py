"""Unit tests for the Go2 squad DAM inter-dog distance guard.

Pure-logic (no Isaac): the HolonomicSolver rollout and the Go2DAMWrapper
priority-weighted distance band run headless. Skipped if DAM isn't installed.
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
    fwd = s.rollout([0, 0, 0.0], [1, 0, 0], dt=2.0)
    assert fwd[0].item() == pytest.approx(2.0, abs=1e-4)
    assert fwd[1].item() == pytest.approx(0.0, abs=1e-4)
    strafe = s.rollout([0, 0, 0.0], [0, 1, 0], dt=2.0)
    assert strafe[1].item() == pytest.approx(2.0, abs=1e-4)
    turned = s.rollout([0, 0, math.pi / 2], [1, 0, 0], dt=2.0)
    assert turned[1].item() == pytest.approx(2.0, abs=1e-4)


def test_holonomic_rollout_is_differentiable():
    s = HolonomicSolver()
    cmd = torch.tensor([0.5, 0.3, 0.1], requires_grad=True)
    nxt = s.rollout(torch.tensor([0.0, 0.0, 0.3]), cmd, dt=0.2)
    nxt[0].backward()
    assert cmd.grad is not None and torch.isfinite(cmd.grad).all() and cmd.grad.abs().sum() > 0


# -- Go2DAMWrapper distance band ----------------------------------------------

@pytest.fixture
def guard():
    return Go2DAMWrapper("go2_squad_safety.yaml")  # band [0.8, 4.0]


def _filter(guard, cmd, neighbors, pose=(0.0, 0.0, 0.0), self_priority=1.0):
    return guard.filter(torch.tensor([cmd]), torch.tensor([pose]), neighbors,
                        self_priority=self_priority)[0].tolist()


def test_in_band_passes_through(guard):
    # neighbour 2 m away, closing slowly -> within what the CBF allows -> unchanged
    safe = _filter(guard, [1.0, 0.0, 0.0], [(2.0, 0.0)])
    assert safe == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)
    assert guard.last_decision == "PASS"


def test_lone_dog_passes_through(guard):
    assert _filter(guard, [1.0, 0.0, 0.0], []) == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)


def test_too_close_slows_or_reverses(guard):
    # neighbour 0.6 m straight ahead (< min 0.8) -> forward command reduced
    safe = _filter(guard, [1.0, 0.0, 0.0], [(0.6, 0.0)])
    assert safe[0] < 0.9
    assert guard.last_decision in {"CLAMP", "REJECT"}


def test_front_neighbour_triggers_sidestep(guard):
    # neighbour front-left -> holonomic guard strafes RIGHT (vy < 0), not just braking
    safe = _filter(guard, [1.0, 0.0, 0.0], [(0.5, 0.4)])
    assert safe[1] < -0.05


def test_too_far_pulls_in(guard):
    safe = _filter(guard, [0.0, 0.0, 0.0], [(6.0, 0.0)])
    assert safe[0] > 0.4


def test_far_lateral_neighbour_pulls_via_strafe(guard):
    safe = _filter(guard, [0.0, 0.0, 0.0], [(0.0, 6.0)])
    assert safe[1] > 0.4


# -- emergent yielding from priority ------------------------------------------

def test_high_priority_deviates_less_than_low(guard):
    """Head-on closing: the high-priority dog corrects far less than the low one,
    so yielding emerges purely from the responsibility split (no yield logic).

    Both dogs drive FORWARD (vx=+1) at each other; the one facing -x (yaw=pi) thereby
    moves toward the origin. Deviation is measured against the same nominal."""
    nominal = [1.0, 0.0, 0.0]
    hi = _filter(guard, nominal, [(0.9, 0.0, 1.0)], pose=(0.0, 0.0, 0.0), self_priority=5.0)
    lo = _filter(guard, nominal, [(0.0, 0.0, 5.0)], pose=(0.9, 0.0, math.pi), self_priority=1.0)
    dev_hi = max(abs(hi[0] - 1.0), abs(hi[1]), abs(hi[2]))
    dev_lo = max(abs(lo[0] - 1.0), abs(lo[1]), abs(lo[2]))
    assert dev_lo > dev_hi          # the low-priority dog does the avoiding
    assert dev_lo > 2 * dev_hi      # and clearly more (it yields)


def test_cohesion_ignores_other_group(guard):
    """A far neighbour in ANOTHER group must NOT pull the dog (else groups merge)."""
    safe = _filter(guard, [0.0, 0.0, 0.0], [(6.0, 0.0, 1.0, 0.0)])  # same_group=0
    assert safe == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    assert guard.last_decision == "PASS"


def test_cohesion_pulls_toward_own_group(guard):
    """The same far neighbour, but in the SAME group, does pull the dog back in."""
    safe = _filter(guard, [0.0, 0.0, 0.0], [(6.0, 0.0, 1.0, 1.0)])  # same_group=1
    assert safe[0] > 0.4


def test_equal_priority_is_symmetric(guard):
    """With equal priority both dogs correct about the same (no one yields)."""
    a = _filter(guard, [1.0, 0.0, 0.0], [(0.9, 0.0, 1.0)], pose=(0.0, 0.0, 0.0), self_priority=1.0)
    b = _filter(guard, [1.0, 0.0, 0.0], [(0.0, 0.0, 1.0)], pose=(0.9, 0.0, math.pi), self_priority=1.0)
    dev_a = abs(a[0] - 1.0)
    dev_b = abs(b[0] - 1.0)
    assert dev_a == pytest.approx(dev_b, abs=0.1)
