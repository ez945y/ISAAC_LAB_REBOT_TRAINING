# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Post-ready summary helpers for DAM demo recordings."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class SegmentMetrics:
    """Recording metrics for one scripted replay segment."""

    mode: str
    steps: int = 0
    risky_frames: int = 0
    max_target_offset: float = 0.0
    min_target_z: float = float("inf")
    max_tracking_error: float = 0.0
    decisions: Counter = field(default_factory=Counter)

    @property
    def interventions(self) -> int:
        return sum(self.decisions[name] for name in ("CLAMP", "REJECT", "FAULT"))


def format_segment_summary(metrics: SegmentMetrics) -> str:
    """Return a compact one-line summary for terminal logs."""
    decision_text = ", ".join(
        f"{name}={metrics.decisions[name]}"
        for name in ("RAW", "PASS", "CLAMP", "REJECT", "FAULT")
        if metrics.decisions[name]
    ) or "none"
    return (
        f"[SUMMARY {metrics.mode.upper()}] "
        f"steps={metrics.steps} risky_frames={metrics.risky_frames} "
        f"max_offset={metrics.max_target_offset:.3f}m "
        f"min_z={metrics.min_target_z:.3f}m "
        f"max_err={metrics.max_tracking_error:.4f}m "
        f"decisions={decision_text}"
    )


def format_linkedin_summary(segment_metrics: list[SegmentMetrics]) -> str:
    """Return a LinkedIn-ready problem/proof/caption summary."""
    raw = next((item for item in segment_metrics if item.mode == "raw"), None)
    dam = next((item for item in segment_metrics if item.mode == "dam"), None)
    risky_frames = max((item.risky_frames for item in segment_metrics), default=0)
    interventions = dam.interventions if dam is not None else 0
    dam_steps = dam.steps if dam is not None else 0
    intervention_rate = interventions / dam_steps if dam_steps else 0.0

    lines = [
        "",
        "LINKEDIN DEMO SUMMARY",
        "Problem: robot policies and teleop streams can generate unsafe targets faster than humans can inspect.",
        "Demo: replay the same scripted risky command twice, first raw and then through DAM.",
        f"Risky command frames: {risky_frames}",
        f"DAM interventions: {interventions} ({intervention_rate:.1%} of DAM frames)",
    ]
    if raw is not None:
        lines.append(
            f"RAW max tracking error: {raw.max_tracking_error:.4f}m; "
            f"max target offset: {raw.max_target_offset:.3f}m"
        )
    if dam is not None:
        lines.append(
            f"DAM decisions: PASS={dam.decisions['PASS']} CLAMP={dam.decisions['CLAMP']} "
            f"REJECT={dam.decisions['REJECT']} FAULT={dam.decisions['FAULT']}"
        )
    if dam is not None and interventions == 0:
        lines.append(
            "Demo tuning note: no DAM intervention was observed; increase --unsafe-scale "
            "or tighten the stackfile before recording the final clip."
        )
    lines.extend(
        [
            "Caption angle: same command, safer robot. DAM turns safety constraints into a runtime control boundary.",
            "",
        ]
    )
    return "\n".join(lines)
