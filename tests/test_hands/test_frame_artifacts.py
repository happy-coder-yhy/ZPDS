from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zpds.hands.contracts import FrameInferenceRecord
from zpds.hands.frame_artifacts import (
    InferenceArtifactContext,
    ParquetBBoxWriter,
    ParquetFrameStatusWriter,
    validate_wilor_frame_artifacts,
)
from zpds.hands.schemas import (
    HandBBox,
    HandKeypoints,
    PreparedFrame,
    RawHandResult,
)


def _context() -> InferenceArtifactContext:
    return InferenceArtifactContext(
        prep_revision="r0001",
        segment_id="seg_000001",
        video_stream_id="ego_rgb",
        model_name="wilor",
        model_version="wilor_cvpr2025",
        checkpoint_sha256="checkpoint-sha",
        config_sha256="config-sha",
        device="cuda:0",
    )


def _hand() -> RawHandResult:
    return RawHandResult(
        handedness="Right",
        handedness_score=0.9,
        keypoints=HandKeypoints(
            normalized=[(0.2, 0.3, 0.0)] * 21,
            pixel=[(20.0, 30.0)] * 21,
        ),
        bbox=HandBBox(10.0, 20.0, 40.0, 60.0),
        detection_score=0.8,
    )


def _frame(index: int) -> PreparedFrame:
    return PreparedFrame(
        frame_rgb=np.zeros((80, 100, 3), dtype=np.uint8),
        output_frame_index=index,
        timestamp_ns=index * 33_333_333,
        source_frame_index=index + 10,
        source_timestamp_ns=1_000_000_000 + index * 33_333_333,
    )


def _write_sample_map(path: Path, count: int) -> None:
    pd.DataFrame(
        {
            "output_frame_index": list(range(count)),
            "output_timestamp_ns": [
                index * 33_333_333 for index in range(count)
            ],
            "source_frame_index": [
                index + 10 for index in range(count)
            ],
            "source_timestamp_ns": [
                1_000_000_000 + index * 33_333_333
                for index in range(count)
            ],
        }
    ).to_parquet(path, index=False)


def test_formal_writers_preserve_full_frame_status_and_primary_bbox(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "wilor_frame_status.parquet"
    bbox_path = tmp_path / "wilor_hands_bbox.parquet"
    status_writer = ParquetFrameStatusWriter(status_path, _context())
    bbox_writer = ParquetBBoxWriter(bbox_path, _context())
    records = [
        FrameInferenceRecord(
            frame=_frame(0),
            inference_status="detected",
            raw_hands=(_hand(),),
            effective_hands=(_hand(),),
            active_backend="wilor",
            inference_ms=2.5,
        ),
        FrameInferenceRecord(
            frame=_frame(1),
            inference_status="no_hand",
            active_backend="wilor",
            inference_ms=1.5,
        ),
        FrameInferenceRecord(
            frame=_frame(2),
            inference_status="failed",
            effective_hands=(_hand(),),
            failure_reason="WiLoR failure; MediaPipe fallback used",
            active_backend="wilor",
            inference_ms=3.5,
        ),
    ]
    for record in records:
        status_writer.write(record)
        bbox_writer.write(record)
    status_writer.close()
    bbox_writer.close()

    status = pd.read_parquet(status_path)
    bbox = pd.read_parquet(bbox_path)
    assert status["inference_status"].tolist() == [
        "detected",
        "no_hand",
        "failed",
    ]
    assert status["hand_count"].tolist() == [1, 0, 0]
    assert len(bbox) == 1
    assert bbox.iloc[0]["handedness"] == "right"


def test_frame_artifact_validator_checks_sample_map_and_bbox_counts(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "wilor_frame_status.parquet"
    bbox_path = tmp_path / "wilor_hands_bbox.parquet"
    sample_map_path = tmp_path / "sample_map.parquet"
    _write_sample_map(sample_map_path, 2)
    status_writer = ParquetFrameStatusWriter(status_path, _context())
    bbox_writer = ParquetBBoxWriter(bbox_path, _context())
    for record in (
        FrameInferenceRecord(
            frame=_frame(0),
            inference_status="detected",
            raw_hands=(_hand(),),
            active_backend="wilor",
            inference_ms=1.0,
        ),
        FrameInferenceRecord(
            frame=_frame(1),
            inference_status="no_hand",
            active_backend="wilor",
            inference_ms=1.0,
        ),
    ):
        status_writer.write(record)
        bbox_writer.write(record)
    status_writer.close()
    bbox_writer.close()

    report = validate_wilor_frame_artifacts(
        status_path,
        bbox_path,
        sample_map_path,
    )

    assert report["status"] == "pass", report
    assert report["checks"]["sample_map_alignment"] == "pass"
    assert report["checks"]["bbox_contract"] == "pass"


def test_frame_artifact_validator_rejects_timestamp_drift(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "wilor_frame_status.parquet"
    bbox_path = tmp_path / "wilor_hands_bbox.parquet"
    sample_map_path = tmp_path / "sample_map.parquet"
    _write_sample_map(sample_map_path, 1)
    status_writer = ParquetFrameStatusWriter(status_path, _context())
    bbox_writer = ParquetBBoxWriter(bbox_path, _context())
    record = FrameInferenceRecord(
        frame=PreparedFrame(
            frame_rgb=np.zeros((80, 100, 3), dtype=np.uint8),
            output_frame_index=0,
            timestamp_ns=999,
            source_frame_index=10,
            source_timestamp_ns=1_000_000_000,
        ),
        inference_status="no_hand",
        active_backend="wilor",
        inference_ms=1.0,
    )
    status_writer.write(record)
    bbox_writer.write(record)
    status_writer.close()
    bbox_writer.close()

    report = validate_wilor_frame_artifacts(
        status_path,
        bbox_path,
        sample_map_path,
    )

    assert report["status"] == "fail"
    assert report["checks"]["sample_map_alignment"] == "fail"


def test_frame_artifact_validator_enforces_failure_ratio(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "wilor_frame_status.parquet"
    bbox_path = tmp_path / "wilor_hands_bbox.parquet"
    sample_map_path = tmp_path / "sample_map.parquet"
    _write_sample_map(sample_map_path, 2)
    status_writer = ParquetFrameStatusWriter(status_path, _context())
    bbox_writer = ParquetBBoxWriter(bbox_path, _context())
    for index in range(2):
        record = FrameInferenceRecord(
            frame=_frame(index),
            inference_status="failed",
            failure_reason="synthetic failure",
            active_backend="wilor",
            inference_ms=1.0,
        )
        status_writer.write(record)
        bbox_writer.write(record)
    status_writer.close()
    bbox_writer.close()

    report = validate_wilor_frame_artifacts(
        status_path,
        bbox_path,
        sample_map_path,
    )

    assert report["status"] == "fail"
    assert report["checks"]["coverage_quality"] == "fail"
