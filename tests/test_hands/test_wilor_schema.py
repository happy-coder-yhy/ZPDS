import numpy as np
import pytest

from zpds.hands.wilor_schema import (
    WiLoRDetection,
    WiLoRImageTransform,
    WiLoRReconstructionResult,
)


def test_wilor_image_transform_from_resize() -> None:
    transform = WiLoRImageTransform.from_resize(
        original_width=1920,
        original_height=1080,
        detector_width=640,
        detector_height=384,
        letterbox_left=0.0,
        letterbox_top=42.0,
    )

    assert transform.original_width == 1920
    assert transform.original_height == 1080
    assert transform.detector_width == 640
    assert transform.detector_height == 384
    assert transform.resize_scale_x == pytest.approx(640 / 1920)  # 0.33
    assert transform.resize_scale_y == pytest.approx(384 / 1080)  # 0.356
    assert transform.letterbox_top == 42.0
    assert transform.is_padded
    assert transform.maintain_aspect


def test_wilor_image_transform_no_padding() -> None:
    transform = WiLoRImageTransform.from_resize(
        original_width=640,
        original_height=480,
        detector_width=640,
        detector_height=480,
    )
    assert not transform.is_padded
    assert transform.resize_scale_x == 1.0
    assert transform.resize_scale_y == 1.0


def test_wilor_detection_fields() -> None:
    transform = WiLoRImageTransform.from_resize(
        original_width=640,
        original_height=480,
        detector_width=256,
        detector_height=256,
        letterbox_left=10.0,
        letterbox_top=20.0,
    )
    detection = WiLoRDetection(
        handedness="Right",
        handedness_score=0.88,
        detection_score=0.92,
        bbox_xyxy_px=(100.0, 150.0, 300.0, 400.0),
        raw_keypoints_2d=None,
        transform=transform,
    )

    assert detection.detection_score == 0.92
    assert detection.handedness == "Right"
    assert detection.handedness_score == 0.88
    assert detection.bbox_xyxy_px == (100.0, 150.0, 300.0, 400.0)
    assert not detection.clipped
    assert detection.transform is not None


def test_wilor_detection_with_raw_keypoints() -> None:
    joints = np.array([[i * 10.0, i * 5.0] for i in range(21)], dtype=np.float32)
    transform = WiLoRImageTransform.from_resize(
        original_width=640, original_height=480,
        detector_width=256, detector_height=256,
    )
    detection = WiLoRDetection(
        handedness="Left",
        handedness_score=0.91,
        detection_score=0.95,
        bbox_xyxy_px=(50.0, 60.0, 200.0, 250.0),
        raw_keypoints_2d=joints,
        raw_keypoint_format="wilor_original",
        transform=transform,
    )

    assert detection.raw_keypoints_2d.shape == (21, 2)
    assert detection.raw_keypoint_format == "wilor_original"


def test_wilor_detection_clipped() -> None:
    detection = WiLoRDetection(
        handedness="Left",
        handedness_score=0.9,
        detection_score=0.9,
        bbox_xyxy_px=(0.0, 0.0, 200.0, 250.0),
        clipped=True,
    )
    assert detection.clipped


def test_wilor_reconstruction_default_not_attempted() -> None:
    result = WiLoRReconstructionResult()

    assert not result.reconstruction_attempted
    assert result.keypoints_3d is None
    assert result.mano_pose is None
    assert not result.pose_valid


def test_wilor_reconstruction_with_mano_params() -> None:
    result = WiLoRReconstructionResult(
        keypoints_3d=np.random.default_rng(1).uniform(0, 1, (21, 3)).astype(np.float32),
        mano_pose=np.zeros(48, dtype=np.float32),
        mano_shape=np.ones(10, dtype=np.float32) * 0.1,
        camera_translation=np.array([0.1, -0.2, 0.5], dtype=np.float32),
        coordinate_frame="camera",
        scale_status="metric",
        reprojection_error_px=2.3,
        pose_valid=True,
        reconstruction_attempted=True,
    )

    assert result.keypoints_3d.shape == (21, 3)
    assert result.pose_valid
    assert result.scale_status == "metric"
    assert result.reprojection_error_px == 2.3
    assert result.reconstruction_attempted
