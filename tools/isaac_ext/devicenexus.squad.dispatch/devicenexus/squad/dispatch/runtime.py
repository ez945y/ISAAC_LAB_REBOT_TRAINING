# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""In-process bridge between the running sim and the dispatch extension.

The extension must read live squad state and issue dispatch commands, but the
squad logic stays Isaac/UI-free. So the sim (scripts/11) registers a *handle*
implementing :class:`DispatchHandle`; the extension reads it each frame. This is
the same decoupling as the ROS/keyboard ingress — the extension is just another
ingress (+ a viewport/UI egress). The handle is a plain object, no Isaac or UI
imports here, so it is trivially unit-testable with a fake.

Lifecycle: the extension's ``on_startup`` may run BEFORE the sim builds its squad,
so ``get_handle()`` can return ``None`` and the extension renders nothing until a
handle appears.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DispatchHandle(Protocol):
    """What the extension needs from the running squad (provided by scripts/11)."""

    def snapshot(self) -> dict:
        """Current state for drawing + the panel. Shape::

            {
              "zones":    [{"name": str, "x": float, "y": float}, ...],
              "groups":   {gid: [agent_id, ...]},      # current grouping
              "selected": gid | None,                   # steered group (None = ALL)
              "formation": {gid: str},                  # per-group formation name
              "dogs":     [{"id","group","x","y","yaw","tx","ty","arrived"}, ...],
            }
        """

    # -- dispatch verbs (mirror SquadController) ----------------------------------
    def dispatch_zone(self, index: int) -> None: ...
    def select_next(self) -> None: ...
    def cycle_formation(self) -> None: ...
    def cycle_regroup(self) -> None: ...
    def toggle_patrol(self) -> None: ...
    def recall(self) -> None: ...
    def halt(self) -> None: ...


_handle: DispatchHandle | None = None


def set_handle(handle: DispatchHandle | None) -> None:
    """Called by the sim once its squad/controller exist (or None to clear)."""
    global _handle
    _handle = handle


def get_handle() -> DispatchHandle | None:
    return _handle
