from __future__ import annotations

from pathlib import Path

import pandas as pd

from zpds_prepare.detectors.umi import analyze_umi_session
from zpds_prepare.readers.session_model import (
    ImuStream,
    Session,
    TimeSeriesStream,
    VideoStream,
)


def _video(robot_id: str, timestamps: list[int]) -> VideoStream:
    return VideoStream(
        stream_id=f"{robot_id}_camera0",
        timestamps_ns=timestamps,
        index_frames=[],
        video_path="unused.mp4",
        fps=30.0,
        frame_count=len(timestamps),
    )


def _imu(robot_id: str, timestamps: list[int]) -> ImuStream:
    return ImuStream(
        stream_id=f"{robot_id}_imu",
        dataframe=pd.DataFrame({"timestamp_ns": timestamps}),
        sample_rate_hz=100.0,
    )


def _time_series(
    tmp_path: Path,
    robot_id: str,
    modality: str,
    timestamps: list[int],
) -> TimeSeriesStream:
    if modality == "vio_pose":
        rows = pd.DataFrame(
            {
                "tx": range(len(timestamps)),
                "ty": 0.0,
                "tz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "source_frame_id": pd.Series(
                    ["world"] * len(timestamps), dtype="string"
                ),
            }
        )
    else:
        rows = pd.DataFrame({"raw_value": [0.25] * len(timestamps)})
    return TimeSeriesStream(
        stream_id=f"{robot_id}_{modality}",
        modality=modality,
        role="sensor",
        source_path=tmp_path / "source.mcap",
        timestamps_ns=timestamps,
        rows=rows,
        expected_rate_hz=100.0,
        metadata={
            "robot_id": robot_id,
            "unit": "unknown",
            "translation_unit": "unknown",
            "semantic_status": "raw_unverified",
        },
    )


def _session(tmp_path: Path) -> Session:
    timestamps0 = [0, 10, 20, 30]
    timestamps1 = [1, 11, 21, 31]
    return Session(
        session_id="umi-demo",
        source_path=str(tmp_path / "source.mcap"),
        meta={},
        video_streams={
            "robot0_camera0": _video("robot0", timestamps0),
            "robot1_camera0": _video("robot1", timestamps1),
        },
        imu_streams={
            "robot0_imu": _imu("robot0", timestamps0),
            "robot1_imu": _imu("robot1", timestamps1),
        },
        time_series_streams={
            stream.stream_id: stream
            for stream in (
                _time_series(tmp_path, "robot0", "vio_pose", timestamps0),
                _time_series(tmp_path, "robot1", "vio_pose", timestamps1),
                _time_series(
                    tmp_path, "robot0", "magnetic_encoder", timestamps0
                ),
                _time_series(
                    tmp_path, "robot1", "magnetic_encoder", timestamps1
                ),
            )
        },
    )


def test_orchestrator_analyzes_all_stream_categories_and_pairs(
    tmp_path: Path,
) -> None:
    bundle = analyze_umi_session(
        _session(tmp_path),
        minimum_gap_ns=100,
        alignment_max_residual_ns=5,
        encoder_freeze_min_samples=3,
    )

    assert set(bundle.timeline_evidence["video"]) == {
        "robot0_camera0",
        "robot1_camera0",
    }
    assert set(bundle.timeline_evidence["imu"]) == {"robot0_imu", "robot1_imu"}
    assert len(bundle.timeline_evidence["time_series"]) == 4
    assert set(bundle.vio_evidence) == {"robot0_vio_pose", "robot1_vio_pose"}
    assert set(bundle.magnetic_encoder_evidence) == {
        "robot0_magnetic_encoder",
        "robot1_magnetic_encoder",
    }
    assert set(bundle.dual_alignments) == {
        "video:camera0",
        "imu:imu",
        "time_series:vio_pose",
        "time_series:magnetic_encoder",
    }
    assert all(
        summary["mapped_ratio"] == 1.0
        for summary in bundle.dual_alignment_summaries.values()
    )


def test_orchestrator_does_not_invent_encoder_semantics_or_write_files(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    before = set(tmp_path.iterdir())

    bundle = analyze_umi_session(
        session,
        minimum_gap_ns=100,
        encoder_freeze_min_samples=3,
    )

    assert set(tmp_path.iterdir()) == before
    assert all(
        summary["semantic_status"] == "raw_unverified"
        and summary["gripper_action_generated"] is False
        and summary["open_close_event_generated"] is False
        and summary["stall_generated"] is False
        for summary in bundle.magnetic_encoder_summaries.values()
    )
    assert not hasattr(bundle, "manifest")
    assert not hasattr(bundle, "quality_views")


def test_orchestrator_uses_vio_boundaries_and_skips_missing_pair(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    session.video_streams.pop("robot1_camera0")
    session.time_series_streams["robot1_vio_pose"].rows.loc[
        2:, "source_frame_id"
    ] = "map"

    bundle = analyze_umi_session(
        session,
        minimum_gap_ns=100,
        alignment_max_residual_ns=5,
    )

    assert "video:camera0" not in bundle.dual_alignments
    alignment = bundle.dual_alignments["time_series:vio_pose"]
    assert alignment["interpolated"].eq(False).all()
    assert bundle.dual_alignment_summaries["time_series:vio_pose"][
        "cross_group_mapping"
    ] == "forbidden"


def test_orchestrator_rejects_imu_without_source_timestamps(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    session.imu_streams["robot0_imu"].dataframe = pd.DataFrame({"ax": [0.0]})

    try:
        analyze_umi_session(session)
    except ValueError as exc:
        assert str(exc) == "robot0_imu IMU dataframe missing timestamp_ns"
    else:
        raise AssertionError("missing IMU timestamps must not be guessed")
