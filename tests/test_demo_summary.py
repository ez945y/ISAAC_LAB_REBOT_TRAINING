from __future__ import annotations

from collections import Counter

from controll_scripts.safety.demo_summary import (
    SegmentMetrics,
    format_linkedin_summary,
    format_segment_summary,
)


def test_segment_summary_includes_recording_metrics() -> None:
    metrics = SegmentMetrics(
        mode="dam",
        steps=120,
        risky_frames=42,
        max_target_offset=0.234,
        min_target_z=0.045,
        max_tracking_error=0.0123,
        decisions=Counter({"PASS": 100, "CLAMP": 20}),
    )

    summary = format_segment_summary(metrics)

    assert "[SUMMARY DAM]" in summary
    assert "risky_frames=42" in summary
    assert "max_offset=0.234m" in summary
    assert "PASS=100" in summary
    assert "CLAMP=20" in summary


def test_linkedin_summary_highlights_interventions() -> None:
    raw = SegmentMetrics(
        mode="raw",
        steps=120,
        risky_frames=60,
        max_target_offset=0.31,
        max_tracking_error=0.08,
        decisions=Counter({"RAW": 120}),
    )
    dam = SegmentMetrics(
        mode="dam",
        steps=120,
        risky_frames=60,
        max_target_offset=0.31,
        max_tracking_error=0.03,
        decisions=Counter({"PASS": 80, "CLAMP": 35, "REJECT": 5}),
    )

    summary = format_linkedin_summary([raw, dam])

    assert "LINKEDIN DEMO SUMMARY" in summary
    assert "Risky command frames: 60" in summary
    assert "DAM interventions: 40 (33.3% of DAM frames)" in summary
    assert "RAW max tracking error: 0.0800m" in summary
    assert "DAM decisions: PASS=80 CLAMP=35 REJECT=5 FAULT=0" in summary
    assert "same command, safer robot" in summary


def test_linkedin_summary_tells_operator_when_demo_has_no_intervention() -> None:
    dam = SegmentMetrics(
        mode="dam",
        steps=100,
        risky_frames=20,
        decisions=Counter({"PASS": 100}),
    )

    summary = format_linkedin_summary([dam])

    assert "DAM interventions: 0 (0.0% of DAM frames)" in summary
    assert "increase --unsafe-scale" in summary
