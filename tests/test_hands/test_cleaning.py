from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from zpds.hands.cleaning import (
    HandVideoCleaningConfig,
    analyze_hand_video,
    clean_hand_video,
)
from zpds.hands.manipulation import is_valid_manipulation
from zpds.hands.metrics import compute_hand_presence_ratio


def _config(**overrides) -> HandVideoCleaningConfig:
    values = {
        "min_video_duration_s": 2.0,
        "min_kept_span_duration_s": 0.2,
        "black_gray_mean_max": 12.0,
        "black_pixel_value_max": 16,
        "black_dark_ratio_min": 0.98,
        "black_min_duration_s": 0.3,
        "blur_laplacian_var_max": 10.0,
        "blur_min_duration_s": 0.3,
        "no_hand_min_duration_s": 0.3,
        "no_operation_center_speed_max_per_s": 0.02,
        "no_operation_flow_magnitude_max_px": 0.2,
        "no_operation_min_duration_s": 0.8,
        "occlusion_pose_completeness_max": 0.7,
        "occlusion_clipped_ratio_min": 0.35,
        "occlusion_min_duration_s": 0.3,
        "flow_consistency_scale_px": 1.5,
        "flow_resize_width": 80,
        "flow_grid_stride": 4,
        "smoothness_normalized_jerk_scale": 1.0,
        "merge_gap_s": 0.0,
        "output_codec": "mp4v",
    }
    values.update(overrides)
    return HandVideoCleaningConfig(**values)


def _frame(index: int, *, flat: int | None = None, static: bool = False) -> np.ndarray:
    if flat is not None:
        return np.full((64, 80, 3), flat, dtype=np.uint8)
    yy, xx = np.indices((64, 80))
    checker = (((xx // 4 + yy // 4) % 2) * 120 + 50).astype(np.uint8)
    image = np.repeat(checker[:, :, None], 3, axis=2)
    offset = 8 if static else 4 + (index * 2) % 45
    cv2.rectangle(image, (offset, 20), (offset + 20, 45), (20, 220, 40), -1)
    return image


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (80, 64)
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _hand_row(
    frame_index: int,
    *,
    clipped_count: int = 0,
    static: bool = False,
    center_x: float | None = None,
    handedness: str = "Right",
    detection_id: int = 0,
) -> dict:
    if center_x is None:
        center_x = 18.0 if static else 14.0 + (frame_index * 2) % 45
    points = [
        [center_x + (point % 5), 25.0 + (point // 5) * 3]
        for point in range(21)
    ]
    return {
        "output_frame_index": frame_index,
        "timestamp_ns": frame_index * 100_000_000,
        "handedness": handedness,
        "detection_id": detection_id,
        "bbox_x1": center_x - 5,
        "bbox_y1": 18.0,
        "bbox_x2": center_x + 15,
        "bbox_y2": 46.0,
        "keypoints_2d": points,
        "keypoints_clipped_count": clipped_count,
    }


def _write_mixed_fixture(tmp_path: Path) -> tuple[Path, Path]:
    frames: list[np.ndarray] = []
    rows: list[dict] = []
    for index in range(60):
        if 10 <= index < 15:
            frame = _frame(index, flat=0)
        elif 25 <= index < 30:
            frame = _frame(index, flat=110)
        elif 40 <= index < 50:
            frame = _frame(index, static=True)
        else:
            frame = _frame(index)
        frames.append(frame)
        if not 35 <= index < 40:
            rows.append(
                _hand_row(
                    index,
                    clipped_count=15 if 55 <= index < 60 else 0,
                    static=40 <= index < 50,
                )
            )
    video = tmp_path / "mixed.avi"
    hands = tmp_path / "hands_2d.parquet"
    _write_video(video, frames)
    pd.DataFrame(rows).to_parquet(hands, index=False)
    return video, hands


def test_analyze_and_clean_all_requested_failure_types(tmp_path: Path) -> None:
    video, hands = _write_mixed_fixture(tmp_path)
    original_hash = hashlib.sha256(video.read_bytes()).hexdigest()

    result = clean_hand_video(video, hands, tmp_path / "derived", _config())

    assert hashlib.sha256(video.read_bytes()).hexdigest() == original_hash
    assert result.cleaned_video_path is not None
    assert result.cleaned_video_path.is_file()
    assert result.report_path.is_file()
    assert result.frame_metrics_path.is_file()
    assert result.sample_map_path.is_file()
    reasons = {
        reason
        for span in result.report["excluded_spans"]
        for reason in span["reasons"]
    }
    assert {
        "black_frame",
        "blur_detected",
        "hand_absent",
        "no_operation",
        "hand_occluded",
    } <= reasons
    metrics = result.report["quality_metrics"]
    assert metrics["hand_presence_ratio"] == pytest.approx(55 / 60)
    assert 0.0 <= metrics["motion_smoothness"] <= 1.0
    assert 0.0 <= metrics["optical_flow_consistency"] <= 1.0
    assert 0.0 <= metrics["hand_pose_completeness"] <= 1.0
    sample_map = pd.read_parquet(result.sample_map_path)
    frame_metrics = pd.read_parquet(result.frame_metrics_path)
    assert len(sample_map) == result.report["summary"]["kept_frames"]
    assert len(frame_metrics) == 60
    assert frame_metrics["is_excluded"].sum() == result.report["summary"]["excluded_frames"]
    capture = cv2.VideoCapture(str(result.cleaned_video_path))
    decoded_cleaned = 0
    while capture.read()[0]:
        decoded_cleaned += 1
    capture.release()
    assert decoded_cleaned == len(sample_map)


def test_too_short_video_is_rejected_without_empty_video_artifact(tmp_path: Path) -> None:
    video = tmp_path / "short.avi"
    hands = tmp_path / "hands.parquet"
    _write_video(video, [_frame(index) for index in range(10)])
    pd.DataFrame([_hand_row(index) for index in range(10)]).to_parquet(hands, index=False)

    result = clean_hand_video(video, hands, tmp_path / "derived", _config())

    assert result.cleaned_video_path is None
    assert result.report["summary"]["overall_disposition"] == "reject"
    assert "video_too_short" in result.report["excluded_spans"][0]["reasons"]
    assert pd.read_parquet(result.sample_map_path).empty


def test_analyze_requires_hand_schema(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    hands = tmp_path / "hands.parquet"
    _write_video(video, [_frame(index) for index in range(20)])
    pd.DataFrame({"output_frame_index": [0]}).to_parquet(hands, index=False)

    with pytest.raises(ValueError, match="缺少字段"):
        analyze_hand_video(video, hands, _config())


def test_frame_status_distinguishes_no_hand_from_inference_failure(tmp_path: Path) -> None:
    video = tmp_path / "status.avi"
    hands = tmp_path / "hands.parquet"
    status = tmp_path / "frame_status.parquet"
    _write_video(video, [_frame(index) for index in range(30)])
    pd.DataFrame([_hand_row(index) for index in range(20)]).to_parquet(hands, index=False)
    pd.DataFrame(
        {
            "output_frame_index": list(range(30)),
            "inference_status": ["detected"] * 20 + ["no_hand"] * 5 + ["failed"] * 5,
        }
    ).to_parquet(status, index=False)

    analysis = analyze_hand_video(video, hands, _config(), status)

    reasons = {
        reason for span in analysis.report["excluded_spans"] for reason in span["reasons"]
    }
    assert "hand_absent" in reasons
    assert "hand_track_lost" in reasons
    assert analysis.frame_metrics["is_no_hand"].sum() == 5
    assert analysis.frame_metrics["is_hand_track_lost"].sum() == 5


def test_two_hand_tracks_do_not_follow_swapping_detection_ids(tmp_path: Path) -> None:
    video = tmp_path / "two_hands.avi"
    hands = tmp_path / "hands.parquet"
    _write_video(video, [_frame(0, static=True) for _ in range(30)])
    rows = []
    for index in range(30):
        rows.extend(
            [
                _hand_row(
                    index,
                    center_x=18.0,
                    handedness="Left",
                    detection_id=index % 2,
                ),
                _hand_row(
                    index,
                    center_x=55.0,
                    handedness="Right",
                    detection_id=(index + 1) % 2,
                ),
            ]
        )
    pd.DataFrame(rows).to_parquet(hands, index=False)

    analysis = analyze_hand_video(video, hands, _config(min_kept_span_duration_s=0.05))

    assert analysis.frame_metrics["is_no_operation"].sum() == 29
    assert analysis.report["quality_metrics"]["motion_smoothness"] == pytest.approx(1.0)


def test_default_config_loads_and_rejects_unknown_fields(tmp_path: Path) -> None:
    config = HandVideoCleaningConfig.load("configs/hands/cleaning_default.yaml")
    assert config.output_codec == "mp4v"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "hand_video_cleaning:\n  unexpected: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="字段不匹配"):
        HandVideoCleaningConfig.load(invalid)


def test_presence_and_manipulation_helpers() -> None:
    observations = [
        {"output_frame_index": 1},
        {"output_frame_index": 1},
        {"output_frame_index": 3},
    ]
    assert compute_hand_presence_ratio(observations, total_frames=5) == pytest.approx(0.4)
    assert compute_hand_presence_ratio([True, False, True]) == pytest.approx(2 / 3)
    assert is_valid_manipulation({"center_speed_normalized_per_s": 0.02})
    assert is_valid_manipulation({}, {"delta": 0.02})
    assert not is_valid_manipulation({}, {"delta": 0.0})
