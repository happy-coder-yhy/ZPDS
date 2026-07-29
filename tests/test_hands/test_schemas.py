from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from zpds.hands.base import RawHandResult as LegacyRawHandResult
from zpds.hands.schemas import (
    HAND_KEYPOINT_COUNT,
    HandBBox,
    HandKeypoints,
    HandObservation,
    PreparedFrame,
    RawHandResult,
)


def _keypoints_2d() -> list[tuple[float, float]]:
    return [(float(index), float(index + 1)) for index in range(HAND_KEYPOINT_COUNT)]


def _observation(**overrides) -> HandObservation:
    values = {
        "segment_id": "seg_000001",
        "video_stream_id": "ego_rgb",
        "output_frame_index": 3,
        "timestamp_ns": 100_000_000,
        "source_frame_index": 4,
        "source_timestamp_ns": 101_000_000,
        "detection_id": 0,
        "handedness": "left",
        "handedness_score": 0.9,
        "bbox_xyxy": (10.0, 20.0, 100.0, 120.0),
        "keypoints_2d": _keypoints_2d(),
        "keypoints_z_relative": [0.0] * HAND_KEYPOINT_COUNT,
        "model_name": "mediapipe",
        "model_version": "hand_landmarker_v1",
    }
    values.update(overrides)
    return HandObservation(**values)


def test_base_module_keeps_raw_result_compatibility() -> None:
    assert LegacyRawHandResult is RawHandResult


def test_hand_keypoints_requires_exactly_21_points() -> None:
    with pytest.raises(ValueError, match="关键点数量"):
        HandKeypoints(
            normalized=[(0.0, 0.0, 0.0)] * 20,
            pixel=[(0.0, 0.0)] * HAND_KEYPOINT_COUNT,
        )


def test_hand_bbox_uses_absolute_xyxy_coordinates() -> None:
    bbox = HandBBox(10.0, 20.0, 30.0, 50.0)

    assert bbox.width == 20.0
    assert bbox.height == 30.0
    assert bbox.area == 600.0
    assert bbox.is_valid


def test_hand_observation_accepts_frozen_v1_contract() -> None:
    observation = _observation(source_frame_index=None, source_timestamp_ns=None)

    assert observation.handedness == "left"
    assert len(observation.keypoints_2d) == HAND_KEYPOINT_COUNT
    with pytest.raises(FrozenInstanceError):
        observation.detection_id = 1  # type: ignore[misc]


@pytest.mark.parametrize("handedness", ["Left", "RIGHT", "", "both"])
def test_hand_observation_rejects_noncanonical_handedness(handedness: str) -> None:
    with pytest.raises(ValueError, match="handedness"):
        _observation(handedness=handedness)


def test_hand_observation_rejects_invalid_bbox() -> None:
    with pytest.raises(ValueError, match="bbox_xyxy"):
        _observation(bbox_xyxy=(100.0, 20.0, 10.0, 120.0))


def test_hand_observation_rejects_wrong_keypoint_count() -> None:
    with pytest.raises(ValueError, match="keypoints_2d"):
        _observation(keypoints_2d=[(0.0, 0.0)] * 20)


def test_hand_observation_rejects_nonfinite_values() -> None:
    keypoints = _keypoints_2d()
    keypoints[5] = (float("nan"), 0.0)

    with pytest.raises(ValueError, match="有限数值"):
        _observation(keypoints_2d=keypoints)


def test_prepared_frame_accepts_rgb_uint8_contract() -> None:
    frame = PreparedFrame(
        frame_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
        output_frame_index=0,
        timestamp_ns=0,
        source_frame_index=None,
        source_timestamp_ns=None,
    )

    assert frame.frame_rgb.shape == (24, 32, 3)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((24, 32), dtype=np.uint8),
        np.zeros((24, 32, 3), dtype=np.float32),
        np.zeros((0, 32, 3), dtype=np.uint8),
    ],
)
def test_prepared_frame_rejects_invalid_image(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        PreparedFrame(
            frame_rgb=image,
            output_frame_index=0,
            timestamp_ns=0,
            source_frame_index=0,
            source_timestamp_ns=0,
        )
