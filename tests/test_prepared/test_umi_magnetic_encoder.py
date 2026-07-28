from __future__ import annotations

from pathlib import Path

import pandas as pd

from segment.magnetic_encoder_writer import (
    write_magnetic_encoder_stream,
)
from segment.segment_writer import build_segment_json
from segment.validator import validate_magnetic_encoder_streams
from zpds_prepare.readers.session_model import TimeSeriesStream


def _stream(
    tmp_path: Path,
    robot_id: str,
    base_ns: int,
) -> TimeSeriesStream:
    return TimeSeriesStream(
        stream_id=f"{robot_id}_magnetic_encoder",
        modality="magnetic_encoder",
        role="sensor",
        source_path=tmp_path / "01767.mcap",
        timestamps_ns=[base_ns, base_ns + 4_000_000],
        rows=pd.DataFrame(
            {
                "log_time_ns": pd.Series(
                    [base_ns + 1, base_ns + 4_000_001],
                    dtype="int64",
                ),
                "publish_time_ns": pd.Series(
                    [base_ns + 2, base_ns + 4_000_002],
                    dtype="int64",
                ),
                "raw_value": pd.Series(
                    [0.10300199687480927, 0.10400199687480927],
                    dtype="float64",
                ),
            }
        ),
        fields=[
            {"name": "log_time_ns", "dtype": "int64", "unit": "ns"},
            {"name": "publish_time_ns", "dtype": "int64", "unit": "ns"},
            {"name": "raw_value", "dtype": "float64", "unit": "unknown"},
        ],
        expected_rate_hz=250.0,
        metadata={
            "robot_id": robot_id,
            "source_topic": f"/{robot_id}/sensor/magnetic_encoder",
            "source_asset_id": "raw_mcap",
            "source_field": "value",
            "unit": "unknown",
            "semantic_status": "raw_unverified",
        },
    )


def test_write_and_validate_dual_magnetic_encoder_streams(
    tmp_path: Path,
) -> None:
    base_ns = 1_767_657_922_000_000_000
    segment_dir = tmp_path / "prepared" / "seg_000001"
    results = [
        write_magnetic_encoder_stream(
            _stream(tmp_path, robot_id, base_ns),
            output_dir=str(segment_dir),
            source_start_ns=base_ns,
            source_end_ns=base_ns + 4_000_000,
        )
        for robot_id in ("robot0", "robot1")
    ]
    segment = build_segment_json(
        dataset_path=str(tmp_path / "source.bin"),
        span={
            "source_start_ns": base_ns,
            "source_end_ns": base_ns + 4_000_000,
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

    validation = validate_magnetic_encoder_streams(
        segment_dir,
        segment,
    )

    assert validation["status"] == "pass"
    assert validation["checks"]["magnetic_encoder_dual_robot"] == "pass"
    assert validation["checks"]["magnetic_encoder_source_match"] == "skip"
    for result in results:
        frame = pd.read_parquet(segment_dir / result["uri"])
        assert frame["timestamp_ns"].tolist() == [0, 4_000_000]
        assert frame["source_timestamp_ns"].tolist() == [
            base_ns,
            base_ns + 4_000_000,
        ]
        assert frame["log_time_ns"].dtype == "int64"
        assert frame["raw_value"].dtype == "float64"
        assert frame["unit"].unique().tolist() == ["unknown"]
        assert frame["semantic_status"].unique().tolist() == [
            "raw_unverified"
        ]


def test_validator_rejects_robot_mixing(tmp_path: Path) -> None:
    base_ns = 1_767_657_922_000_000_000
    segment_dir = tmp_path / "prepared" / "seg_000001"
    results = [
        write_magnetic_encoder_stream(
            _stream(tmp_path, robot_id, base_ns),
            output_dir=str(segment_dir),
            source_start_ns=base_ns,
            source_end_ns=base_ns + 4_000_000,
        )
        for robot_id in ("robot0", "robot1")
    ]
    robot0_path = segment_dir / results[0]["uri"]
    robot0 = pd.read_parquet(robot0_path)
    robot0.loc[1, "robot_id"] = "robot1"
    robot0.to_parquet(robot0_path, index=False)
    segment = build_segment_json(
        dataset_path=str(tmp_path / "source.bin"),
        span={
            "source_start_ns": base_ns,
            "source_end_ns": base_ns + 4_000_000,
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

    validation = validate_magnetic_encoder_streams(
        segment_dir,
        segment,
    )

    assert validation["status"] == "fail"
    assert any("Robot rows are mixed" in error for error in validation["errors"])
