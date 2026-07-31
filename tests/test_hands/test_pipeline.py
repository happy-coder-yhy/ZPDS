from collections.abc import Iterator

import numpy as np
import pytest

from zpds.hands.pipeline import HandsPipeline, HandsPipelineError
from zpds.hands.schemas import (
    HAND_KEYPOINT_COUNT,
    HandBBox,
    HandFrameResult,
    HandKeypoints,
    ModelAttemptResult,
    PreparedFrame,
    RawHandResult,
)


class FakeReader:
    def __init__(self, frames: list[PreparedFrame]) -> None:
        self._frames = frames

    @property
    def segment_id(self) -> str:
        return "seg_000001"

    @property
    def video_stream_id(self) -> str:
        return "ego_rgb"

    def __iter__(self) -> Iterator[PreparedFrame]:
        return iter(self._frames)


class FakeEstimator:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.timestamps_ms: list[int] = []
        self.frames: list[np.ndarray] = []

    def estimate(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[RawHandResult]:
        self.frames.append(frame_rgb)
        self.timestamps_ms.append(timestamp_ms)
        response = self._responses[len(self.timestamps_ms) - 1]
        if isinstance(response, Exception):
            raise response
        return response  # type: ignore[return-value]


def _frame(
    output_frame_index: int,
    timestamp_ns: int,
    *,
    source_frame_index: int | None = None,
    source_timestamp_ns: int | None = None,
) -> PreparedFrame:
    return PreparedFrame(
        frame_rgb=np.full(
            (24, 32, 3),
            output_frame_index,
            dtype=np.uint8,
        ),
        output_frame_index=output_frame_index,
        timestamp_ns=timestamp_ns,
        source_frame_index=source_frame_index,
        source_timestamp_ns=source_timestamp_ns,
    )


def _raw_hand(
    handedness: str = "Left",
    *,
    handedness_score: float = 0.9,
    offset: float = 0.0,
) -> RawHandResult:
    normalized = [
        (
            index / 100.0,
            (index + 1) / 100.0,
            -(index + 2) / 100.0,
        )
        for index in range(HAND_KEYPOINT_COUNT)
    ]
    pixel = [(offset + index + 0.25, offset + index + 0.75) for index in range(HAND_KEYPOINT_COUNT)]
    return RawHandResult(
        handedness=handedness,
        handedness_score=handedness_score,
        keypoints=HandKeypoints(normalized=normalized, pixel=pixel),
        bbox=HandBBox(
            x1=10.0 + offset,
            y1=20.0 + offset,
            x2=100.0 + offset,
            y2=120.0 + offset,
        ),
        detection_score=0.8,
    )


def _pipeline(
    frames: list[PreparedFrame],
    responses: list[object],
) -> tuple[HandsPipeline, FakeEstimator]:
    estimator = FakeEstimator(responses)
    pipeline = HandsPipeline(
        reader=FakeReader(frames),
        estimator=estimator,
        model_name="mediapipe",
        model_version="hand_landmarker_v1",
    )
    return pipeline, estimator


def test_pipeline_handles_frames_without_hands() -> None:
    pipeline, estimator = _pipeline(
        [_frame(0, 0), _frame(1, 33_333_333)],
        [[], []],
    )

    assert list(pipeline) == []
    assert estimator.timestamps_ms == [0, 33]
    assert pipeline.stats.frames_processed == 2
    assert pipeline.stats.observations_created == 0
    assert pipeline.stats.frames_with_hands == 0
    assert pipeline.stats.average_fps > 0


def test_pipeline_can_return_observation_list() -> None:
    pipeline, _ = _pipeline(
        [_frame(0, 0)],
        [[_raw_hand("Left")]],
    )

    observations = pipeline.run_to_list()

    assert isinstance(observations, list)
    assert len(observations) == 1
    assert observations[0].handedness == "left"


def test_pipeline_stops_after_max_frames() -> None:
    frames = [_frame(index, index * 33_333_333) for index in range(4)]
    estimator = FakeEstimator([[], [], [], []])
    pipeline = HandsPipeline(
        reader=FakeReader(frames),
        estimator=estimator,
        model_name="mediapipe",
        model_version="hand_landmarker_v1",
        max_frames=2,
    )

    assert list(pipeline) == []
    assert estimator.timestamps_ms == [0, 33]
    assert pipeline.stats.frames_processed == 2


@pytest.mark.parametrize("max_frames", [0, -1])
def test_pipeline_rejects_invalid_max_frames(max_frames: int) -> None:
    with pytest.raises(ValueError, match="max_frames"):
        HandsPipeline(
            reader=FakeReader([_frame(0, 0)]),
            estimator=FakeEstimator([[]]),
            model_name="mediapipe",
            model_version="hand_landmarker_v1",
            max_frames=max_frames,
        )


def test_pipeline_converts_raw_result_and_preserves_provenance() -> None:
    pipeline, _ = _pipeline(
        [
            _frame(
                7,
                233_333_331,
                source_frame_index=19,
                source_timestamp_ns=1_233_333_331,
            )
        ],
        [[_raw_hand("Left")]],
    )

    observations = list(pipeline)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.segment_id == "seg_000001"
    assert observation.video_stream_id == "ego_rgb"
    assert observation.output_frame_index == 7
    assert observation.timestamp_ns == 233_333_331
    assert observation.source_frame_index == 19
    assert observation.source_timestamp_ns == 1_233_333_331
    assert observation.detection_id == 0
    assert observation.handedness == "left"
    assert observation.handedness_score == 0.9
    assert observation.bbox_xyxy == (10.0, 20.0, 100.0, 120.0)
    assert observation.keypoints_2d[0] == (0.25, 0.75)
    assert observation.keypoints_z_relative[0] == -0.02
    assert len(observation.keypoints_2d) == HAND_KEYPOINT_COUNT
    assert len(observation.keypoints_z_relative) == HAND_KEYPOINT_COUNT
    assert observation.model_name == "mediapipe"
    assert observation.model_version == "hand_landmarker_v1"


def test_pipeline_detection_ids_restart_for_each_frame() -> None:
    pipeline, _ = _pipeline(
        [_frame(0, 0), _frame(1, 33_333_333)],
        [
            [_raw_hand("Left"), _raw_hand("Right", offset=5.0)],
            [_raw_hand("Right")],
        ],
    )

    observations = list(pipeline)

    assert [item.detection_id for item in observations] == [0, 1, 0]
    assert [item.handedness for item in observations] == ["left", "right", "right"]
    assert pipeline.stats.frames_processed == 2
    assert pipeline.stats.observations_created == 3
    assert pipeline.stats.frames_with_hands == 2


def test_pipeline_preserves_frame_level_wilor_fallback_attribution() -> None:
    class RichEstimator:
        def estimate_frame(
            self,
            frame_rgb: np.ndarray,
            timestamp_ms: int,
        ) -> HandFrameResult:
            primary = ModelAttemptResult(
                model_name="wilor",
                backend_name="wilor",
                status="failed",
                hands=[],
                inference_ms=1.0,
                failure_reason="synthetic WiLoR failure",
                model_version="wilor_cvpr2025",
                checkpoint_sha256="wilor-sha",
                device="cuda:0",
            )
            fallback = ModelAttemptResult(
                model_name="mediapipe",
                backend_name="mediapipe",
                status="detected",
                hands=[_raw_hand("Right")],
                inference_ms=1.0,
                failure_reason=None,
                model_version="hand_landmarker_v1",
                checkpoint_sha256=None,
                device="cpu",
            )
            return HandFrameResult(
                timestamp_ms=timestamp_ms,
                requested_model="wilor",
                primary=primary,
                fallback=fallback,
                fallback_attempted=True,
                fallback_used=True,
                fallback_reason=primary.failure_reason,
                effective_model="mediapipe",
                effective_hands=fallback.hands,
            )

    pipeline = HandsPipeline(
        reader=FakeReader([_frame(0, 0)]),
        estimator=RichEstimator(),
        model_name="wilor",
        model_version="wilor_cvpr2025",
    )

    observation = next(iter(pipeline))

    assert observation.model_name == "mediapipe"
    assert observation.backend_requested == "wilor"
    assert observation.backend_active == "mediapipe"
    assert observation.backend_fallback_used is True
    assert observation.backend_fallback_reason == "synthetic WiLoR failure"


@pytest.mark.parametrize(
    ("model_label", "expected"),
    [
        ("Left", "left"),
        (" RIGHT ", "right"),
        ("unknown", "unknown"),
        ("Both", "unknown"),
        ("", "unknown"),
    ],
)
def test_pipeline_normalizes_handedness(model_label: str, expected: str) -> None:
    pipeline, _ = _pipeline(
        [_frame(0, 0)],
        [[_raw_hand(model_label)]],
    )

    assert next(iter(pipeline)).handedness == expected


def test_pipeline_makes_model_timestamps_strictly_increasing() -> None:
    pipeline, estimator = _pipeline(
        [
            _frame(0, 0),
            _frame(1, 500_000),
            _frame(2, 2_000_000),
            _frame(3, 2_100_000),
        ],
        [[], [], [], []],
    )

    list(pipeline)

    assert estimator.timestamps_ms == [0, 1, 2, 3]


def test_pipeline_wraps_estimator_failure_with_frame_context() -> None:
    pipeline, _ = _pipeline(
        [_frame(0, 0), _frame(1, 33_333_333)],
        [[], RuntimeError("model unavailable")],
    )

    with pytest.raises(HandsPipelineError) as error_info:
        list(pipeline)

    message = str(error_info.value)
    assert "model unavailable" in message
    assert "segment=seg_000001" in message
    assert "stream=ego_rgb" in message
    assert "output_frame_index=1" in message
    assert "timestamp_ns=33333333" in message
    assert isinstance(error_info.value.__cause__, RuntimeError)
    assert pipeline.stats.frames_processed == 1


@pytest.mark.parametrize("invalid_response", [None, (), ["not-a-result"]])
def test_pipeline_rejects_invalid_estimator_output(invalid_response: object) -> None:
    pipeline, _ = _pipeline([_frame(0, 0)], [invalid_response])

    with pytest.raises(HandsPipelineError):
        list(pipeline)


def test_pipeline_is_one_shot_because_video_estimator_has_state() -> None:
    pipeline, _ = _pipeline([_frame(0, 0)], [[]])

    assert list(pipeline) == []
    with pytest.raises(HandsPipelineError, match="不能重复运行"):
        list(pipeline)


@pytest.mark.parametrize(
    ("model_name", "model_version"),
    [("", "v1"), ("mediapipe", "")],
)
def test_pipeline_requires_model_identity(model_name: str, model_version: str) -> None:
    with pytest.raises(ValueError):
        HandsPipeline(
            reader=FakeReader([_frame(0, 0)]),
            estimator=FakeEstimator([[]]),
            model_name=model_name,
            model_version=model_version,
        )
