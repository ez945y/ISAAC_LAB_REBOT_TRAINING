# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Quadruped squad dispatch: decoupled scheduler / mission / robot ABI.

Pure-logic layers (no Isaac dependency, unit-testable):
    formation, robot_agent, mission, scheduler

Isaac-bound layer (import only after the sim app starts):
    locomotion  (Go2Locomotion)
"""

from .formation import Formation, slots
from .mission import FormationMoveToArea, Mission
from .robot_agent import AgentState, LocomotionBackend, RobotAgent, velocity_command
from .scheduler import Dispatcher, Squad, split_evenly

__all__ = [
    "Formation", "slots",
    "Mission", "FormationMoveToArea",
    "AgentState", "LocomotionBackend", "RobotAgent", "velocity_command",
    "Dispatcher", "Squad", "split_evenly",
]
