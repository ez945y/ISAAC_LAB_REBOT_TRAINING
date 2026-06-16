# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Formation slot geometry — pure logic, no Isaac dependency (unit-testable).

A formation places ``n`` robots at fixed offsets around a group *anchor* (the
target the group is moving to), oriented by a *heading*. The scheduler asks for
``slots(anchor, heading, n)`` and assigns each member one world-space slot; the
locomotion layer then drives each robot there.
"""

from __future__ import annotations

import math
from enum import Enum


class Formation(str, Enum):
    ROW = "row"        # shoulder-to-shoulder, perpendicular to heading
    COLUMN = "column"  # single file, along heading
    WEDGE = "wedge"    # V / arrow, leader forward


def slots(
    anchor_xy: tuple[float, float],
    heading_rad: float,
    n: int,
    *,
    shape: Formation = Formation.WEDGE,
    spacing: float = 1.2,
) -> list[tuple[float, float]]:
    """World-space slot positions for ``n`` robots around ``anchor_xy``.

    ``heading_rad`` is the facing/advance direction. ``spacing`` is the gap
    between adjacent robots in meters.
    """
    if n <= 0:
        return []
    fwd = (math.cos(heading_rad), math.sin(heading_rad))
    lat = (math.cos(heading_rad + math.pi / 2.0), math.sin(heading_rad + math.pi / 2.0))
    ax, ay = anchor_xy

    out: list[tuple[float, float]] = []
    for i in range(n):
        centered = i - (n - 1) / 2.0  # symmetric about the anchor
        if shape is Formation.ROW:
            f, l = 0.0, centered * spacing
        elif shape is Formation.COLUMN:
            f, l = -i * spacing, 0.0
        else:  # WEDGE: spread laterally and trail back from the centre
            f, l = -abs(centered) * spacing * 0.7, centered * spacing
        out.append((ax + fwd[0] * f + lat[0] * l, ay + fwd[1] * f + lat[1] * l))
    return out
