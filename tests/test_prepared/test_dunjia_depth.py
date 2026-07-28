from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from segment import depth_writer
from segment.calibration import (
    extract_calibration_from_mcap,
    write_calibration,
)
from segment.depth_writer import write_depth_stream
from segment.validator import (
    validate_depth_streams,
    validate_source_hashes,
)
from zpds_prepare.readers.session_model import DepthStream


def test_write_dunjia_depth_preserves_both_source_clocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "session.mcap"
    source.write_bytes(b"fixture")
    frames = [
        {
            "seq": 0,
            "timestamp_ns": 1_000_000_000,
            "log_time_ns": 1_000_000_003,
            "publish_time_ns": 1_000_000_005,
            "source_frame_position": 0,
        },
        {
            "seq": 1,
            "timestamp_ns": 1_040_000_000,
            "log_time_ns": 1_040_000_003,
            "publish_time_ns": 1_040_000_005,
            "source_frame_position": 1,
        },
    ]
    stream = DepthStream(
        stream_id="ego_depth",
        timestamps_ns=[frame["timestamp_ns"] for frame in frames],
        index_frames=frames,
        source_files=[source],
        source_kind="mcap_compressed_image",
        fps=25.0,
        width=4,
        height=3,
        frame_count=2,
        dtype="uint16",
        unit="unknown",
        invalid_value=None,
        frame_id="headcam_depth_optical_frame",
        metadata={
            "topic": "/robot0/sensor/depth/compressed",
            "unit_status": "unverified",
            "source_asset_id": "raw_mcap",
            "operation": "trim_decode_embedded_png",
        },
    )

    def fake_iter(_stream, sample_map):
        for row in sample_map.itertuples(index=False):
            yield row, np.full((3, 4), row.output_frame_index + 100, np.uint16)

    monkeypatch.setattr(
        depth_writer,
        "_iter_mcap_compressed_images",
        fake_iter,
    )
    segment_dir = tmp_path / "prepared" / "seg_000001"

    result = write_depth_stream(
        stream,
        output_dir=str(segment_dir),
        source_start_ns=1_000_000_000,
        source_end_ns=1_040_000_000,
    )

    assert result["frames"] == 2
    assert result["rate_hz"] == 25.0
    assert result["source_asset_id"] == "raw_mcap"
    assert result["operation"] == "trim_decode_embedded_png"
    assert result["unit"] == "unknown"
    sample_map = pd.read_parquet(
        segment_dir / "maps" / "ego_depth_sample_map.parquet"
    )
    assert sample_map["source_timestamp_ns"].tolist() == [
        1_000_000_000,
        1_040_000_000,
    ]
    assert sample_map["source_log_time_ns"].tolist() == [
        1_000_000_003,
        1_040_000_003,
    ]
    assert sample_map["source_publish_time_ns"].tolist() == [
        1_000_000_005,
        1_040_000_005,
    ]
    assert sample_map["mapping_method"].tolist() == ["identity", "identity"]
    first = cv2.imread(
        str(segment_dir / "data" / "depth" / "ego_depth" / "00000000.png"),
        cv2.IMREAD_UNCHANGED,
    )
    assert first is not None
    assert first.dtype == np.uint16
    assert first.shape == (3, 4)

    calibration = extract_calibration_from_mcap(
        {
            "width": 4,
            "height": 3,
            "frame_id": "headcam_center_optical_frame",
            "K": [4.0, 0.0, 2.0, 0.0, 3.0, 1.5, 0.0, 0.0, 1.0],
        },
        multi_cam={
            "camera0": {
                "width": 4,
                "height": 3,
                "frame_id": "headcam_center_optical_frame",
                "K": [4.0, 0.0, 2.0, 0.0, 3.0, 1.5, 0.0, 0.0, 1.0],
            },
            "depth": {
                "width": 4,
                "height": 3,
                "frame_id": "headcam_depth_optical_frame",
                "K": [4.0, 0.0, 2.0, 0.0, 3.0, 1.5, 0.0, 0.0, 1.0],
            },
        },
    )
    write_calibration(calibration, str(segment_dir))
    segment = {
        "timeline": {"start_ns": 0, "end_ns": 40_000_000},
        "calibration_uri": "calibration/calibration.json",
        "source_assets": [
            {
                "source_asset_id": "raw_mcap",
                "uri": source.name,
                "sha256": "unused",
            }
        ],
        "streams": [
            {
                "stream_id": "ego_depth",
                "modality": "depth",
                "uri": result["uri"],
                "format": "png_sequence",
                "shape": [3, 4],
                "dtype": "uint16",
                "unit": "unknown",
                "frame_id": "headcam_depth_optical_frame",
                "origin": {
                    "source_asset_id": "raw_mcap",
                    "operation": "trim_decode_embedded_png",
                    "sample_map_uri": result["sample_map_uri"],
                },
            }
        ],
    }
    validation = validate_depth_streams(segment_dir, segment)
    assert validation["errors"] == []
    assert validation["checks"]["depth_ego_depth_dual_clock"] == "pass"
    assert validation["checks"]["depth_ego_depth_boundary"] == "pass"
    assert validation["checks"]["depth_ego_depth_calibration"] == "pass"


def test_source_hash_validation_detects_matching_raw(tmp_path: Path) -> None:
    source = tmp_path / "session.mcap"
    source.write_bytes(b"dunjia")
    import hashlib

    expected = hashlib.sha256(b"dunjia").hexdigest()
    result = validate_source_hashes(
        {
            "source_session": {"session_uri": str(source)},
            "source_assets": [
                {
                    "source_asset_id": "raw_mcap",
                    "uri": source.name,
                    "sha256": expected,
                }
            ],
        }
    )

    assert result["checks"]["source_hashes"] == "pass"
    assert result["statistics"]["source_hashes_verified"] == 1
    assert result["errors"] == []
