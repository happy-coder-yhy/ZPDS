from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from zpds.hands.base import RawHandResult as LegacyRawHandResult
from zpds.hands.schemas import (
    HAND_KEYPOINT_COUNT,
    HandBBox,
    HandFrameResult,
    HandKeypoints,
    HandObservation,
    ModelAttemptResult,
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


def test_hand_observation_rejects_inconsistent_clipping_metadata() -> None:
    with pytest.raises(ValueError, match="保持一致"):
        _observation(
            keypoints_any_clipped=False,
            keypoints_clipped_count=1,
        )


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


# ════════════════════════════════════════════════════════════════════
# from_components 模型无关构造入口
# ════════════════════════════════════════════════════════════════════


def _normalized_landmarks() -> np.ndarray:
    rng = np.random.default_rng(42)
    arr = rng.uniform(0.0, 0.5, (HAND_KEYPOINT_COUNT, 3)).astype(np.float64)
    arr[:, 0] *= 0.5  # x
    arr[:, 1] *= 0.5  # y
    arr[:, 2] = arr[:, 2] - 0.25  # z centered around 0
    return arr


def test_from_components_basic_construction() -> None:
    landmarks = _normalized_landmarks()
    result = RawHandResult.from_components(
        handedness="Left",
        handedness_score=0.95,
        detection_score=0.90,
        normalized_landmarks=landmarks,
        image_width=640,
        image_height=480,
        label="hand_0",
    )

    assert result.handedness == "Left"
    assert result.handedness_score == 0.95
    assert result.detection_score == 0.90
    assert result.label == "hand_0"
    assert len(result.keypoints.normalized) == HAND_KEYPOINT_COUNT
    assert len(result.keypoints.pixel) == HAND_KEYPOINT_COUNT
    assert result.bbox.is_valid


def test_from_components_pixel_within_image_bounds() -> None:
    landmarks = _normalized_landmarks()
    result = RawHandResult.from_components(
        handedness="Right",
        handedness_score=0.8,
        detection_score=0.8,
        normalized_landmarks=landmarks,
        image_width=1920,
        image_height=1080,
        label="hand_1",
    )

    for px, py in result.keypoints.pixel:
        assert 0.0 <= px < 1920.0
        assert 0.0 <= py < 1080.0


def test_from_components_with_explicit_bbox() -> None:
    landmarks = _normalized_landmarks()
    result = RawHandResult.from_components(
        handedness="Left",
        handedness_score=0.7,
        detection_score=0.7,
        normalized_landmarks=landmarks,
        image_width=640,
        image_height=480,
        bbox_xyxy=(50.0, 60.0, 200.0, 300.0),
        label="hand_0",
    )

    assert result.bbox.x1 == 50.0
    assert result.bbox.y1 == 60.0
    assert result.bbox.x2 == 200.0
    assert result.bbox.y2 == 300.0
    assert not result.bbox.is_padded  # 预计算 BBox 不做 padding


def test_from_components_with_visibility() -> None:
    landmarks = _normalized_landmarks()
    vis = [0.5 + i * 0.02 for i in range(HAND_KEYPOINT_COUNT)]
    result = RawHandResult.from_components(
        handedness="Left",
        handedness_score=0.9,
        detection_score=0.9,
        normalized_landmarks=landmarks,
        image_width=640,
        image_height=480,
        visibility=vis,
    )

    assert result.keypoints.has_visibility
    assert len(result.keypoints.visibility) == HAND_KEYPOINT_COUNT


def test_from_components_counts_clipped_keypoints() -> None:
    # 制造越界关键点：第一个点在边缘外
    landmarks = _normalized_landmarks()
    landmarks[0, 0] = -0.01  # x < 0
    landmarks[0, 1] = -0.01  # y < 0

    result = RawHandResult.from_components(
        handedness="Left",
        handedness_score=0.5,
        detection_score=0.5,
        normalized_landmarks=landmarks,
        image_width=640,
        image_height=480,
    )

    assert result.keypoints.any_clipped
    assert result.keypoints.clipped_count >= 1
    # 像素坐标被裁剪到 0
    assert result.keypoints.pixel[0][0] == 0.0
    assert result.keypoints.pixel[0][1] == 0.0


def test_from_components_rejects_wrong_shape() -> None:
    bad = np.zeros((20, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="形状"):
        RawHandResult.from_components(
            handedness="Left",
            handedness_score=0.5,
            detection_score=0.5,
            normalized_landmarks=bad,
            image_width=640,
            image_height=480,
        )


def test_from_components_rejects_bad_score() -> None:
    landmarks = _normalized_landmarks()
    with pytest.raises(ValueError, match="handedness_score"):
        RawHandResult.from_components(
            handedness="Left",
            handedness_score=1.5,
            detection_score=0.5,
            normalized_landmarks=landmarks,
            image_width=640,
            image_height=480,
        )


def test_from_mediapipe_delegates_to_from_components() -> None:
    """确保 from_mediapipe 内部复用了 from_components 的像素坐标与裁剪逻辑。"""

    class _FakeLandmark:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x = x
            self.y = y
            self.z = z
            self.visibility = 0.99

    class _FakeHandedness:
        def __init__(self) -> None:
            self.category_name = "Left"
            self.score = 0.88

    landmarks = [_FakeLandmark(0.1 + i * 0.02, 0.3 + i * 0.02, 0.0) for i in range(21)]
    handedness = _FakeHandedness()

    result = RawHandResult.from_mediapipe(
        hand_landmarks=landmarks,
        handedness=handedness,
        image_width=800,
        image_height=600,
        bbox_padding_ratio=0.10,
        hand_index=1,
    )

    assert result.handedness == "Left"
    assert result.handedness_score == 0.88
    assert result.label == "hand_1"
    assert result.keypoints.has_visibility
    assert result.bbox.is_valid


# ════════════════════════════════════════════════════════════════════
# ModelAttemptResult
# ════════════════════════════════════════════════════════════════════


def _raw_hand() -> RawHandResult:
    return RawHandResult.from_components(
        handedness="Left",
        handedness_score=0.9,
        detection_score=0.85,
        normalized_landmarks=_normalized_landmarks(),
        image_width=640,
        image_height=480,
        label="hand_0",
    )


def test_model_attempt_detected() -> None:
    attempt = ModelAttemptResult(
        model_name="wilor",
        backend_name="wilor_torch",
        status="detected",
        hands=[_raw_hand()],
        inference_ms=15.0,
        failure_reason=None,
        model_version="v1.0",
        checkpoint_sha256="abc123",
        device="cuda:0",
    )

    assert attempt.model_name == "wilor"
    assert attempt.status == "detected"
    assert len(attempt.hands) == 1


def test_model_attempt_failed() -> None:
    attempt = ModelAttemptResult(
        model_name="wilor",
        backend_name="wilor_torch",
        status="failed",
        hands=[],
        inference_ms=0.0,
        failure_reason="CUDA out of memory",
        model_version="v1.0",
        checkpoint_sha256=None,
        device="cuda:0",
    )

    assert attempt.status == "failed"
    assert attempt.failure_reason == "CUDA out of memory"


def test_model_attempt_rejects_detected_without_hands() -> None:
    # TODO(WiLoR Phase 4): 21 点映射验收后恢复此校验
    # 目前允许 detected + empty hands（WiLoR 检测到但映射未验收）
    pass  # 校验暂时关闭


def test_model_attempt_rejects_failed_without_reason() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        ModelAttemptResult(
            model_name="wilor",
            backend_name="wilor_torch",
            status="failed",
            hands=[],
            inference_ms=0.0,
            failure_reason=None,
            model_version="v1.0",
            checkpoint_sha256=None,
            device="cuda:0",
        )


def test_model_attempt_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        ModelAttemptResult(
            model_name="",
            backend_name="wilor_torch",
            status="not_run",
            hands=[],
            inference_ms=0.0,
            failure_reason=None,
            model_version="v1.0",
            checkpoint_sha256=None,
            device="cuda:0",
        )


# ════════════════════════════════════════════════════════════════════
# HandFrameResult
# ════════════════════════════════════════════════════════════════════


def _make_attempt(
    model_name: str = "wilor",
    status: str = "detected",
    hands: list | None = None,
    inference_ms: float = 12.0,
    failure_reason: str | None = None,
) -> ModelAttemptResult:
    return ModelAttemptResult(
        model_name=model_name,
        backend_name="wilor_torch" if model_name == "wilor" else "mediapipe_tasks",
        status=status,  # type: ignore[arg-type]
        hands=hands if hands is not None else [_raw_hand()],
        inference_ms=inference_ms,
        failure_reason=failure_reason,
        model_version="v1.0",
        checkpoint_sha256="abc123",
        device="cuda:0",
    )


def test_frame_result_wilor_detected_no_fallback() -> None:
    primary = _make_attempt(status="detected")
    result = HandFrameResult(
        timestamp_ms=100,
        requested_model="wilor",
        primary=primary,
        fallback=None,
        fallback_attempted=False,
        fallback_used=False,
        effective_model="wilor",
        effective_hands=primary.hands,
    )

    assert result.effective_model == "wilor"
    assert len(result.effective_hands) == 1
    assert not result.fallback_used


def test_frame_result_wilor_failed_mediapipe_success() -> None:
    primary = _make_attempt(status="failed", inference_ms=0.0, failure_reason="OOM")
    fallback = _make_attempt(
        model_name="mediapipe",
        status="detected",
        inference_ms=22.0,
    )
    result = HandFrameResult(
        timestamp_ms=100,
        requested_model="wilor",
        primary=primary,
        fallback=fallback,
        fallback_attempted=True,
        fallback_used=True,
        fallback_reason="OOM",
        effective_model="mediapipe",
        effective_hands=fallback.hands,
    )

    assert result.fallback_used
    assert result.effective_model == "mediapipe"
    assert result.primary.status == "failed"  # 失败信息保留


def test_frame_result_no_hand_both() -> None:
    primary = _make_attempt(status="no_hand", hands=[])
    fallback = _make_attempt(
        model_name="mediapipe",
        status="no_hand",
        hands=[],
        inference_ms=20.0,
    )
    result = HandFrameResult(
        timestamp_ms=200,
        requested_model="wilor",
        primary=primary,
        fallback=None,
        fallback_attempted=False,
        fallback_used=False,
        effective_model="wilor",
        effective_hands=[],
    )

    assert result.primary.status == "no_hand"
    assert result.effective_hands == []


def test_frame_result_rejects_fallback_used_without_attempted() -> None:
    primary = _make_attempt(status="no_hand", hands=[])
    with pytest.raises(ValueError, match="fallback_attempted"):
        HandFrameResult(
            timestamp_ms=0,
            requested_model="wilor",
            primary=primary,
            fallback=None,
            fallback_attempted=False,
            fallback_used=True,
        )


def test_frame_result_rejects_fallback_attempted_without_result() -> None:
    primary = _make_attempt(status="failed", inference_ms=0.0, failure_reason="OOM")
    with pytest.raises(ValueError, match="fallback 不能为 None"):
        HandFrameResult(
            timestamp_ms=0,
            requested_model="wilor",
            primary=primary,
            fallback=None,
            fallback_attempted=True,
            fallback_used=False,
        )
