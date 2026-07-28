from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from segment.segment_writer import build_segment_json
from segment.vio_pose_validator import validate_vio_pose_streams
from segment.vio_pose_writer import (
    assign_continuity_groups,
    write_vio_pose_stream,
)
from zpds_prepare.readers.session_model import TimeSeriesStream


def _stream(
    tmp_path: Path,
    robot_id: str,
    base_ns: int,
) -> TimeSeriesStream:
    timestamps_ns = [
        base_ns,
        base_ns + 33_333_333,
        base_ns + 633_333_333,
        base_ns + 666_666_666,
    ]
    return TimeSeriesStream(
        stream_id=f"{robot_id}_vio_pose",
        modality="vio_pose",
        role="state",
        source_path=tmp_path / "01767.mcap",
        timestamps_ns=timestamps_ns,
        rows=pd.DataFrame(
            {
                "log_time_ns": pd.Series(
                    [timestamp + 1 for timestamp in timestamps_ns],
                    dtype="int64",
                ),
                "publish_time_ns": pd.Series(
                    [timestamp + 2 for timestamp in timestamps_ns],
                    dtype="int64",
                ),
                "tx": pd.Series([0.0, 0.1, 0.2, 0.3], dtype="float64"),
                "ty": pd.Series([0.0, 0.0, 0.0, 0.0], dtype="float64"),
                "tz": pd.Series([0.0, 0.0, 0.0, 0.0], dtype="float64"),
                "qx": pd.Series([0.0, 0.0, 0.0, 0.0], dtype="float64"),
                "qy": pd.Series([0.0, 0.0, 0.0, 0.0], dtype="float64"),
                "qz": pd.Series([0.0, 0.0, 0.0, 0.0], dtype="float64"),
                "qw": pd.Series([1.0, 1.0, 1.0, 1.0], dtype="float64"),
                "source_frame_id": pd.Series(
                    ["world"] * 4,
                    dtype="string",
                ),
                "source_header_topic": pd.Series(
                    ["/robot0/vio/eef_pose"] * 4,
                    dtype="string",
                ),
            }
        ),
        fields=[
            {"name": "translation", "shape": [3], "dtype": "float64"},
            {
                "name": "orientation",
                "shape": [4],
                "dtype": "float64",
                "representation": "quaternion_xyzw",
            },
        ],
        expected_rate_hz=30.0,
        frame_id="world",
        metadata={
            "robot_id": robot_id,
            "source_topic": f"/{robot_id}/vio/eef_pose",
            "source_schema": "foxglove.PoseInFrame",
            "source_asset_id": "raw_mcap",
            "source_field": "pose",
            "translation_unit": "unknown",
            "orientation_representation": "quaternion_xyzw",
            "semantic_status": "raw_unverified",
            "child_frame": "unknown",
            "transform_direction": "unknown",
            "source_topic_authority": "mcap_channel",
        },
    )


def _segment(
    tmp_path: Path,
    base_ns: int,
    results: list[dict],
) -> dict:
    return build_segment_json(
        dataset_path=str(tmp_path / "source.bin"),
        span={
            "source_start_ns": base_ns,
            "source_end_ns": base_ns + 666_666_666,
        },
        source_assets=[
            {
                "source_asset_id": "raw_mcap",
                "uri": "01767.mcap",
                "sha256": "fixture",
            }
        ],
        profile="umi",
        time_series_results=results,
    )


def test_write_and_validate_dual_vio_pose_streams(
    tmp_path: Path,
) -> None:
    base_ns = 1_767_657_922_000_000_000
    segment_dir = tmp_path / "prepared" / "seg_000001"
    results = [
        write_vio_pose_stream(
            _stream(tmp_path, robot_id, base_ns),
            output_dir=str(segment_dir),
            source_start_ns=base_ns,
            source_end_ns=base_ns + 666_666_666,
        )
        for robot_id in ("robot0", "robot1")
    ]
    segment = _segment(tmp_path, base_ns, results)

    validation = validate_vio_pose_streams(segment_dir, segment)

    assert validation["status"] == "pass"
    assert validation["checks"]["vio_pose_dual_robot"] == "pass"
    assert validation["checks"]["vio_pose_source_match"] == "skip"
    for result in results:
        frame = pd.read_parquet(segment_dir / result["uri"])
        assert frame["continuity_group_id"].tolist() == [0, 0, 1, 1]
        assert frame["continuity_start_reason"].tolist() == [
            "segment_start",
            "",
            "timestamp_gap",
            "",
        ]
        assert frame["parent_frame"].unique().tolist() == ["world"]
        assert frame["child_frame"].unique().tolist() == ["unknown"]
        assert frame["translation_unit"].unique().tolist() == ["unknown"]
        assert frame["semantic_status"].unique().tolist() == [
            "raw_unverified"
        ]


def test_continuity_groups_detect_clock_and_frame_breaks() -> None:
    timestamps = np.asarray([10, 20, 15, 25], dtype=np.int64)
    frames = np.asarray(["world", "world", "world", "map"])
    quaternions = np.asarray(
        [[0.0, 0.0, 0.0, 1.0]] * 4,
        dtype=np.float64,
    )

    groups, reasons, _ = assign_continuity_groups(
        timestamps,
        frames,
        quaternions,
    )

    assert groups.tolist() == [0, 0, 1, 2]
    assert reasons == [
        "segment_start",
        "",
        "timestamp_non_increasing",
        "reference_frame_change",
    ]


def test_validator_rejects_invalid_quaternion(tmp_path: Path) -> None:
    base_ns = 1_767_657_922_000_000_000
    segment_dir = tmp_path / "prepared" / "seg_000001"
    results = [
        write_vio_pose_stream(
            _stream(tmp_path, robot_id, base_ns),
            output_dir=str(segment_dir),
            source_start_ns=base_ns,
            source_end_ns=base_ns + 666_666_666,
        )
        for robot_id in ("robot0", "robot1")
    ]
    robot0_path = segment_dir / results[0]["uri"]
    robot0 = pd.read_parquet(robot0_path)
    robot0.loc[1, ["qx", "qy", "qz", "qw"]] = 0.0
    robot0.to_parquet(robot0_path, index=False)

    validation = validate_vio_pose_streams(
        segment_dir,
        _segment(tmp_path, base_ns, results),
    )

    assert validation["status"] == "fail"
    assert any(
        "Quaternion norm is invalid" in error
        for error in validation["errors"]
    )
