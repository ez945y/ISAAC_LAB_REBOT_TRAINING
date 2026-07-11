# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Exact segment-segment closest-point query for planar capsule bodies.

A dog's elongated body is modelled as a CAPSULE: the sweep of a disc along the
spine segment ``centre ± half_length * heading``. The inter-body distance is then
the closest distance between the two spine segments (minus radii, which the
caller folds into ``min_dist``). This replaces the old 3-sphere sampling
(offsets {-h, 0, +h}), whose inter-sample gaps under-estimated the body on
oblique crossings.

Pure ``math`` implementation (no torch/numpy) so it is unit-testable anywhere
and adds no per-call allocation. Algorithm: standard clamped closest-point
between segments (Ericson, *Real-Time Collision Detection*, §5.1.9), re-expressed
in centre + unit-heading + half-length form so the returned parameters are the
body-frame spine offsets the DAM gradient needs.
"""

from __future__ import annotations

import math

_EPS = 1e-12


def seg_seg_closest(
    ax: float, ay: float, aux: float, auy: float, ah: float,
    bx: float, by: float, bux: float, buy: float, bh: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Closest points between spine segments A and B.

    Segment A is ``(ax, ay) + s * (aux, auy)`` with ``s in [-ah, ah]``; B likewise
    with parameter ``t in [-bh, bh]``. Directions must be unit vectors.

    Returns:
        ``(s, t, dist, pax, pay, pbx, pby)`` — the spine offsets of the closest
        pair, their distance, and the two closest points.
    """
    # Endpoint form: P1 = A - ah*u_a, d1 = 2ah*u_a (Ericson works on [0,1] params).
    p1x, p1y = ax - ah * aux, ay - ah * auy
    p2x, p2y = bx - bh * bux, by - bh * buy
    d1x, d1y = 2.0 * ah * aux, 2.0 * ah * auy
    d2x, d2y = 2.0 * bh * bux, 2.0 * bh * buy
    rx, ry = p1x - p2x, p1y - p2y

    a = d1x * d1x + d1y * d1y          # squared length of A
    e = d2x * d2x + d2y * d2y          # squared length of B
    f = d2x * rx + d2y * ry

    if a <= _EPS and e <= _EPS:        # both degenerate to points
        s01, t01 = 0.0, 0.0
    elif a <= _EPS:                    # A is a point
        s01 = 0.0
        t01 = min(1.0, max(0.0, f / e))
    else:
        c = d1x * rx + d1y * ry
        if e <= _EPS:                  # B is a point
            t01 = 0.0
            s01 = min(1.0, max(0.0, -c / a))
        else:
            b = d1x * d2x + d1y * d2y
            denom = a * e - b * b      # >= 0; 0 iff parallel
            if denom > _EPS:
                s01 = min(1.0, max(0.0, (b * f - c * e) / denom))
            else:
                s01 = 0.0              # parallel: pick A's start, clamp on B below
            t01 = (b * s01 + f) / e
            if t01 < 0.0:
                t01 = 0.0
                s01 = min(1.0, max(0.0, -c / a))
            elif t01 > 1.0:
                t01 = 1.0
                s01 = min(1.0, max(0.0, (b - c) / a))

    pax, pay = p1x + s01 * d1x, p1y + s01 * d1y
    pbx, pby = p2x + t01 * d2x, p2y + t01 * d2y
    dist = math.hypot(pax - pbx, pay - pby)
    # Back to body-frame spine offsets in [-h, h].
    s = (2.0 * s01 - 1.0) * ah
    t = (2.0 * t01 - 1.0) * bh
    return s, t, dist, pax, pay, pbx, pby
