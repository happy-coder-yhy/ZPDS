from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zpds_prepare.detectors.umi import (
    analyze_magnetic_encoder,
    analyze_stream_timeline,
    analyze_vio_quality,
    build_dual_alignment,
)
from zpds_prepare.readers.session_model import TimeSeriesStream


def _vio_stream(
    tmp_path: Path,
    timestamps_ns: list[int],
    *,
    translations: list[float] | None = None,
    quaternions: list[tuple[float, float, float, float]] | None = None,
    frames: list[str] | None = None,
) -> TimeSeriesStream:
    count = len(timestamps_ns)
    translations = translations or [float(index) for index in range(count)]
    quaternions = quaternions or [(0.0, 0.0, 0.0, 1.0)] * count
    frames = frames or ["world"] * count
    return TimeSeriesStream(
        stream_id="robot0_vio_pose",
        modality="vio_pose",
        role="state",
        source_path=tmp_path / "source.mcap",
        timestamps_ns=timestamps_ns,
        rows=pd.DataFrame(
            {
                "tx": translations,
                "ty": [0.0] * count,
                "tz": [0.0] * count,
                "qx": [value[0] for value in quaternions],
                "qy": [value[1] for value in quaternions],
                "qz": [value[2] for value in quaternions],
                "qw": [value[3] for value in quaternions],
                "source_frame_id": pd.Series(frames, dtype="string"),
            }
        ),
        fields=[],
        expected_rate_hz=30.0,
        metadata={
            "robot_id": "robot0",
            "source_topic": "/robot0/vio/eef_pose",
            "translation_unit": "unknown",
            "semantic_status": "raw_unverified",
        },
    )


def _encoder_stream(
    tmp_path: Path,
    timestamps_ns: list[int],
    values: list[float],
) -> TimeSeriesStream:
    return TimeSeriesStream(
        stream_id="robot0_magnetic_encoder",
        modality="magnetic_encoder",
        role="sensor",
        source_path=tmp_path / "source.mcap",
        timestamps_ns=timestamps_ns,
        rows=pd.DataFrame({"raw_value": values}),
        fields=[{"name": "raw_value", "unit": "unknown"}],
        expected_rate_hz=250.0,
        metadata={
            "robot_id": "robot0",
            "unit": "unknown",
            "semantic_status": "raw_unverified",
        },
    )


def test_timeline_preserves_source_order_and_splits_boundaries() -> None:
    timestamps = [100, 110, 110, 90, 100, 1_000]

    evidence, summary, issues = analyze_stream_timeline(
        "robot0_vio_pose",
        timestamps,
        minimum_gap_ns=100,
    )

    assert evidence["timestamp_ns"].tolist() == timestamps
    assert evidence["continuity_group"].tolist() == [0, 0, 1, 2, 2, 3]
    assert evidence["continuity_start_reason"].tolist() == [
        "stream_start",
        "",
        "timestamp_duplicate",
        "timestamp_rollback",
        "",
        "timestamp_gap",
    ]
    assert summary["source_order_preserved"] is True
    assert summary["interpolation_across_groups"] == "forbidden"
    assert {issue.issue_type for issue in issues} == {
        "umi_timestamp_duplicate",
        "umi_timestamp_rollback",
        "umi_timestamp_gap",
    }


def test_dual_alignment_does_not_cross_non_overlapping_groups() -> None:
    robot0 = [0, 10, 1_000, 1_010]
    robot1 = [1, 11, 2_000, 2_010]

    alignment, summary = build_dual_alignment(
        robot0,
        robot1,
        robot0_groups=[0, 0, 1, 1],
        robot1_groups=[0, 0, 1, 1],
        max_residual_ns=5,
    )

    assert alignment["mapping_method"].tolist() == [
        "inferred",
        "inferred",
        "unavailable",
        "unavailable",
    ]
    assert alignment["robot1_sample_index"].tolist()[:2] == [0, 1]
    assert pd.isna(alignment.loc[2, "robot1_sample_index"])
    assert alignment["interpolated"].eq(False).all()
    assert summary["mapped_ratio"] == 0.5
    assert summary["cross_group_mapping"] == "forbidden"


def test_dual_alignment_reports_residual_statistics() -> None:
    alignment, summary = build_dual_alignment(
        [100, 200, 300],
        [102, 205, 309],
        max_residual_ns=10,
        mapping_method="direct",
    )

    assert alignment["residual_ns"].tolist() == [2, 5, 9]
    assert alignment["mapping_method"].tolist() == ["direct"] * 3
    assert summary["residual_p50_ns"] == 5
    assert summary["residual_max_ns"] == 9
    assert summary["interpolation_used"] is False


def test_vio_quality_marks_gaps_frames_and_invalid_quaternions(
    tmp_path: Path,
) -> None:
    timestamps = [0, 10, 1_000, 1_010]
    stream = _vio_stream(
        tmp_path,
        timestamps,
        quaternions=[
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ],
        frames=["world", "world", "world", "map"],
    )

    evidence, summary, issues = analyze_vio_quality(
        stream,
        minimum_gap_ns=100,
    )

    assert evidence["continuity_group"].tolist() == [0, 0, 1, 2]
    assert evidence["quaternion_valid"].tolist() == [True, True, False, True]
    assert summary["explicit_reset_signal_available"] is False
    assert summary["interpolation_across_groups"] == "forbidden"
    issue_types = {issue.issue_type for issue in issues}
    assert "umi_vio_timestamp_gap" in issue_types
    assert "umi_vio_invalid_quaternion" in issue_types
    assert "umi_vio_reference_frame_change" in issue_types


def test_vio_translation_jump_is_only_a_statistical_candidate(
    tmp_path: Path,
) -> None:
    stream = _vio_stream(
        tmp_path,
        [0, 10, 20, 30, 40, 50],
        translations=[0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
    )

    _, summary, issues = analyze_vio_quality(stream, minimum_gap_ns=1_000)

    candidates = [
        issue
        for issue in issues
        if issue.issue_type == "umi_vio_translation_step_candidate"
    ]
    assert len(candidates) == 1
    assert candidates[0].decision == "keep_with_flag"
    assert candidates[0].details["physical_semantics"] == "unavailable"
    assert summary["translation_unit"] == "unknown"


def test_vio_quality_flags_header_topic_channel_mismatch(
    tmp_path: Path,
) -> None:
    stream = _vio_stream(tmp_path, [0, 10, 20])
    stream.rows["source_header_topic"] = "/robot1/vio/eef_pose"

    evidence, summary, issues = analyze_vio_quality(stream)

    assert evidence["header_topic_matches_channel"].eq(False).all()
    assert summary["header_topic_mismatch_count"] == 3
    mismatch = next(
        issue
        for issue in issues
        if issue.issue_type == "umi_vio_header_topic_mismatch"
    )
    assert mismatch.decision == "keep_with_flag"
    assert mismatch.details["physical_pose_values_modified"] is False


def test_magnetic_encoder_stays_raw_unverified(tmp_path: Path) -> None:
    timestamps = list(range(12))
    stream = _encoder_stream(tmp_path, timestamps, [0.25] * 12)

    evidence, summary, issues = analyze_magnetic_encoder(
        stream,
        freeze_min_samples=10,
        minimum_gap_ns=100,
    )

    assert evidence["semantic_status"].unique().tolist() == ["raw_unverified"]
    assert evidence["freeze_candidate"].all()
    assert summary["gripper_action_generated"] is False
    assert summary["open_close_event_generated"] is False
    assert summary["stall_generated"] is False
    assert summary["physical_interpretation"] == "unavailable"
    assert any(
        issue.issue_type == "umi_magnetic_encoder_freeze_candidate"
        and issue.details["physical_interpretation"] == "unavailable"
        for issue in issues
    )


def test_magnetic_encoder_reports_non_finite_without_semantic_claim(
    tmp_path: Path,
) -> None:
    stream = _encoder_stream(
        tmp_path,
        [0, 4, 8, 12, 16],
        [0.1, np.nan, 0.2, np.inf, 0.3],
    )

    evidence, summary, issues = analyze_magnetic_encoder(
        stream,
        minimum_gap_ns=100,
    )

    assert evidence["value_finite"].tolist() == [True, False, True, False, True]
    assert summary["finite_ratio"] == 0.6
    assert summary["semantic_status"] == "raw_unverified"
    assert sum(
        issue.issue_type == "umi_magnetic_encoder_non_finite"
        for issue in issues
    ) == 2
