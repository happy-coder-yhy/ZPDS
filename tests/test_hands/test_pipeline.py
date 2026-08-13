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


class FakeStructuredEstimator:
    def __init__(self, responses: list[HandFrameResult]) -> None:
        self._responses = responses
        self.timestamps_ms: list[int] = []
        self.closed = False

    def estimate_frame(
        self,
        _frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> HandFrameResult:
        self.timestamps_ms.append(timestamp_ms)
        return self._responses[len(self.timestamps_ms) - 1]

    def estimate(
        self,
        _frame_rgb: np.ndarray,
        _timestamp_ms: int,
    ) -> list[RawHandResult]:
        raise AssertionError(
            "structured estimator must use estimate_frame(), not estimate()"
        )

    def close(self) -> None:
        self.closed = True


def _structured_result(
    status: str,
    *,
    hands: list[RawHandResult] | None = None,
    failure_reason: str | None = None,
    inference_ms: float = 1.25,
) -> HandFrameResult:
    primary = ModelAttemptResult(
        model_name="wilor",
        backend_name="wilor",
        status=status,  # type: ignore[arg-type]
        hands=hands or [],
        inference_ms=inference_ms,
        failure_reason=failure_reason,
        model_version="test",
        checkpoint_sha256="abc",
        device="cuda:0",
    )
    return HandFrameResult(
        timestamp_ms=0,
        requested_model="wilor",
        primary=primary,
        fallback=None,
        effective_model=(
            "wilor" if status == "detected" else None
        ),
        effective_hands=list(primary.hands),
    )


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
        model_name="wilor",
        model_version="wilor_cvpr2025",
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
    assert pipeline.stats.frames_no_hand == 2
    assert pipeline.stats.frames_failed == 0
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
    assert observation.model_name == "wilor"
    assert observation.model_version == "wilor_cvpr2025"


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
    assert pipeline.stats.frames_processed == 2
    assert pipeline.stats.frames_failed == 1
    assert pipeline.frame_statistics.to_manifest() == {
        "requested": 2,
        "detected": 0,
        "no_hand": 1,
        "failed": 1,
        "skipped_invalid_input": 0,
    }


def test_run_frames_records_every_status_and_continues_after_failure() -> None:
    frames = [
        _frame(
            index,
            index * 33_333_333,
            source_frame_index=index + 10,
            source_timestamp_ns=1_000_000_000 + index * 33_333_333,
        )
        for index in range(4)
    ]
    estimator = FakeEstimator(
        [
            [_raw_hand("Left")],
            [],
            RuntimeError("single frame failure"),
            [_raw_hand("Right")],
        ]
    )
    pipeline = HandsPipeline(
        reader=FakeReader(frames),
        estimator=estimator,
        model_name="wilor",
        model_version="test",
        active_backend="wilor",
    )

    records = pipeline.run_frames_to_list()

    assert [record.inference_status for record in records] == [
        "detected",
        "no_hand",
        "failed",
        "detected",
    ]
    assert estimator.timestamps_ms == [0, 33, 66, 99]
    assert [record.frame.output_frame_index for record in records] == [0, 1, 2, 3]
    assert [record.frame.timestamp_ns for record in records] == [
        0,
        33_333_333,
        66_666_666,
        99_999_999,
    ]
    assert [record.frame.source_frame_index for record in records] == [
        10,
        11,
        12,
        13,
    ]
    assert records[2].failure_reason == (
        "RuntimeError: single frame failure"
    )
    assert all(record.active_backend == "wilor" for record in records)
    assert all(record.inference_ms >= 0 for record in records)
    assert pipeline.stats.frames_processed == 4
    assert pipeline.stats.frames_with_hands == 2
    assert pipeline.stats.frames_no_hand == 1
    assert pipeline.stats.frames_failed == 1
    assert pipeline.stats.observations_created == 2
    assert pipeline.frame_statistics.requested == 4
    assert pipeline.frame_statistics.accounted == 4
    assert pipeline.frame_statistics.is_complete


def test_pipeline_preserves_structured_wilor_frame_statuses() -> None:
    frames = [
        _frame(index, index * 33_333_333)
        for index in range(4)
    ]
    estimator = FakeStructuredEstimator(
        [
            _structured_result(
                "detected",
                hands=[_raw_hand("Left")],
                inference_ms=2.0,
            ),
            _structured_result("no_hand", inference_ms=3.0),
            _structured_result(
                "failed",
                failure_reason="WiLoRInferenceError: bad frame",
                inference_ms=4.0,
            ),
            _structured_result(
                "skipped_invalid_input",
                failure_reason="invalid RGB frame",
                inference_ms=0.0,
            ),
        ]
    )
    pipeline = HandsPipeline(
        reader=FakeReader(frames),
        estimator=estimator,
        model_name="wilor",
        model_version="test",
        active_backend="wilor",
    )

    records = pipeline.run_frames_to_list()

    assert [record.inference_status for record in records] == [
        "detected",
        "no_hand",
        "failed",
        "skipped_invalid_input",
    ]
    assert [record.inference_ms for record in records] == [
        2.0,
        3.0,
        4.0,
        0.0,
    ]
    assert records[2].failure_reason == (
        "WiLoRInferenceError: bad frame"
    )
    assert records[3].failure_reason == "invalid RGB frame"
    assert estimator.timestamps_ms == [0, 33, 66, 99]
    assert pipeline.stats.frames_with_hands == 1
    assert pipeline.stats.frames_no_hand == 1
    assert pipeline.stats.frames_failed == 1
    assert pipeline.stats.frames_skipped_invalid_input == 1
    assert pipeline.frame_statistics.to_manifest() == {
        "requested": 4,
        "detected": 1,
        "no_hand": 1,
        "failed": 1,
        "skipped_invalid_input": 1,
    }


@pytest.mark.parametrize(
    "result",
    [
        _structured_result("detected"),
        _structured_result(
            "not_run",
            failure_reason="run aborted",
        ),
    ],
)
def test_pipeline_never_turns_invalid_structured_status_into_no_hand(
    result: HandFrameResult,
) -> None:
    pipeline = HandsPipeline(
        reader=FakeReader([_frame(0, 0)]),
        estimator=FakeStructuredEstimator([result]),
        model_name="wilor",
        model_version="test",
        active_backend="wilor",
    )

    record = pipeline.run_frames_to_list()[0]

    assert record.inference_status == "failed"
    assert record.failure_reason
    assert pipeline.stats.frames_failed == 1
    assert pipeline.stats.frames_no_hand == 0


def test_run_frames_turns_invalid_model_output_into_failed_status() -> None:
    pipeline, estimator = _pipeline(
        [_frame(0, 0), _frame(1, 33_333_333)],
        [["not-a-result"], []],
    )

    records = pipeline.run_frames_to_list()

    assert [record.inference_status for record in records] == [
        "failed",
        "no_hand",
    ]
    assert "RawHandResult" in (records[0].failure_reason or "")
    assert estimator.timestamps_ms == [0, 33]
    assert pipeline.stats.frames_processed == 2
    assert pipeline.stats.frames_failed == 1
    assert pipeline.stats.frames_no_hand == 1


def test_pipeline_interfaces_share_one_shot_guard() -> None:
    pipeline, _ = _pipeline([_frame(0, 0)], [[]])

    assert pipeline.run_frames_to_list()[0].inference_status == "no_hand"
    with pytest.raises(HandsPipelineError, match="不能重复运行"):
        list(pipeline)


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


# ════════════════════════════════════════════════════════════════════
# 批量推理路径（回归：estimate_batch 链路曾传 PreparedFrame 而非
# frame_rgb，真实 WiLoR 输入校验抛 TypeError 导致整批失败）
# ════════════════════════════════════════════════════════════════════


class FakeBatchEstimator(FakeStructuredEstimator):
    """带批量接口的 estimator：与真实 WiLoR 相同，校验入参为 ndarray。"""

    supports_batch = True

    def estimate_batch(
        self,
        frames_rgb: list[np.ndarray],
        timestamps_ms: list[int],
    ) -> list[HandFrameResult]:
        # 回归锚点：pipeline 曾把 PreparedFrame 对象传进来
        for f in frames_rgb:
            if not isinstance(f, np.ndarray):
                raise TypeError(
                    f"frame_rgb 必须是 np.ndarray, 实际 {type(f).__name__}"
                )
        self.timestamps_ms.extend(timestamps_ms)
        return [
            self._responses[i % len(self._responses)]
            for i in range(len(timestamps_ms))
        ]


def _batch_pipeline(
    responses: list[HandFrameResult],
    frame_count: int,
) -> HandsPipeline:
    frames = [_frame(i, i * 33_333_000) for i in range(frame_count)]
    return HandsPipeline(
        reader=FakeReader(frames),
        estimator=FakeBatchEstimator(responses),
        model_name="wilor",
        model_version="test",
        active_backend="wilor",
    )


def test_batch_path_passes_ndarray_and_flushes_tail() -> None:
    """不足一批的尾帧也能 flush（回归：残帧永远不处理会挂死）。"""
    responses = [_structured_result("no_hand")] * 5
    pipeline = _batch_pipeline(responses, frame_count=5)
    obs = list(pipeline)
    est = pipeline._estimator
    assert len(est.timestamps_ms) == 5  # 全部帧都送达 estimator
    assert len(obs) == 0  # no_hand 帧不产出观测


def test_batch_path_flushes_full_batches() -> None:
    """满批 + 残批都处理（16 帧一批），detected 帧产出观测。"""
    n = 16 + 5
    responses = [_structured_result("no_hand"), _structured_result("detected", hands=[
        RawHandResult(
            handedness="right", handedness_score=0.9,
            keypoints=HandKeypoints(
                normalized=[(0.0, 0.0, 0.0)] * HAND_KEYPOINT_COUNT,
                pixel=[(0.0, 0.0)] * HAND_KEYPOINT_COUNT,
            ),
            bbox=HandBBox(x1=0.0, y1=0.0, x2=4.0, y2=5.0),
        )
    ])]
    pipeline = _batch_pipeline(responses, frame_count=n)
    obs = list(pipeline)
    # 响应循环 [no_hand, detected] → 奇数帧 detected（i=1,3,...,19）→ 10 条观测
    assert len(obs) == 10
    assert len(pipeline._estimator.timestamps_ms) == n  # noqa: SLF001


def test_batch_path_skipped_invalid_input_matches_frames() -> None:
    """批量路径中无效帧也返回与输入等长的结果（不越界、不错位）。"""
    responses = [_structured_result("no_hand")] * 3
    est = FakeBatchEstimator(responses)
    frames = [
        _frame(0, 0),
        _frame(1, 33_333_000),
        _frame(2, 66_666_000),
    ]
    pipe = HandsPipeline(FakeReader(frames), est, model_name="wilor",
                         model_version="test", active_backend="wilor")
    obs = list(pipe)
    assert len(obs) == 0
    assert len(est.timestamps_ms) == 3
