from types import SimpleNamespace

from zpds_prepare.decisions.segment_planner import plan_segments
from zpds_prepare.detectors.depth_coverage import detect_depth_coverage


def _stream(*timestamps: int):
    return SimpleNamespace(timestamps_ns=list(timestamps))


def test_depth_early_end_trims_candidate_to_visual_overlap() -> None:
    issues = detect_depth_coverage(
        video_streams={
            "camera0": _stream(0, 14_000_000_000),
            "camera1": _stream(0, 14_240_000_000),
            "camera2": _stream(0, 14_240_000_000),
        },
        depth_streams={
            "ego_depth": _stream(0, 12_440_000_000),
        },
        tolerance_ns=80_000_000,
    )

    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == "depth_early_end"
    assert issue.decision == "trim"
    assert issue.start_ns == 12_440_000_000
    assert issue.end_ns == 14_000_000_000
    assert issue.details["missing_tail_ns"] == 1_560_000_000
    assert issue.details["coverage_ratio"] == 0.888571

    candidates = plan_segments(
        issues=issues,
        session_start_ns=0,
        session_end_ns=14_000_000_000,
    )
    assert len(candidates) == 1
    assert candidates[0].source_start_ns == 0
    assert candidates[0].source_end_ns == 12_440_000_000
    assert candidates[0].reason == "trimmed_by_quality_boundary"


def test_depth_boundary_difference_within_tolerance_is_allowed() -> None:
    issues = detect_depth_coverage(
        video_streams={"camera0": _stream(0, 1_000_000_000)},
        depth_streams={"ego_depth": _stream(0, 950_000_000)},
        tolerance_ns=80_000_000,
    )

    assert issues == []
