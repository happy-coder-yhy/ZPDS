"""Prepared RGB 帧到统一手部观测的 Hands V1 Pipeline。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from zpds.hands.schemas import Handedness, HandObservation, PreparedFrame, RawHandResult


class HandEstimator(Protocol):
    """人员 B 提供的模型适配器接口。"""

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]:
        """对单帧 RGB 图像推理，无检测时返回空列表。"""


class PreparedFrameSource(Protocol):
    """Pipeline 所需的 Prepared Segment Reader 最小接口。"""

    @property
    def segment_id(self) -> str: ...

    @property
    def video_stream_id(self) -> str: ...

    def __iter__(self) -> Iterator[PreparedFrame]: ...


@dataclass(frozen=True)
class PipelineStats:
    """一次 Pipeline 运行的统计信息。"""

    frames_processed: int = 0
    observations_created: int = 0
    frames_with_hands: int = 0
    elapsed_seconds: float = 0.0

    @property
    def average_fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


class HandsPipelineError(RuntimeError):
    """Hands Pipeline 在某个具体 Prepared 帧失败。"""

    def __init__(
        self,
        message: str,
        *,
        segment_id: str,
        video_stream_id: str,
        output_frame_index: int | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        context = [
            f"segment={segment_id}",
            f"stream={video_stream_id}",
        ]
        if output_frame_index is not None:
            context.append(f"output_frame_index={output_frame_index}")
        if timestamp_ns is not None:
            context.append(f"timestamp_ns={timestamp_ns}")
        super().__init__(f"{message}: {', '.join(context)}")
        self.segment_id = segment_id
        self.video_stream_id = video_stream_id
        self.output_frame_index = output_frame_index
        self.timestamp_ns = timestamp_ns


class HandsPipeline:
    """运行一次 Prepared Segment 手部推理并产生 ``HandObservation``。

    MediaPipe VIDEO estimator 本身带有时间状态，因此 Pipeline 是一次性的。
    如需重新运行，请创建新的 estimator 和 Pipeline 实例。
    """

    def __init__(
        self,
        reader: PreparedFrameSource,
        estimator: HandEstimator,
        *,
        model_name: str,
        model_version: str,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name 不能为空")
        if not model_version.strip():
            raise ValueError("model_version 不能为空")
        self._reader = reader
        self._estimator = estimator
        self._model_name = model_name
        self._model_version = model_version
        self._stats = PipelineStats()
        self._started = False

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    def __iter__(self) -> Iterator[HandObservation]:
        if self._started:
            raise HandsPipelineError(
                "HandsPipeline 实例不能重复运行",
                segment_id=self._reader.segment_id,
                video_stream_id=self._reader.video_stream_id,
            )
        self._started = True

        frames_processed = 0
        observations_created = 0
        frames_with_hands = 0
        previous_timestamp_ms: int | None = None
        started_at = time.perf_counter()

        try:
            for frame in self._reader:
                timestamp_ms = self._model_timestamp_ms(
                    frame.timestamp_ns,
                    previous_timestamp_ms,
                )
                previous_timestamp_ms = timestamp_ms

                try:
                    raw_results = self._estimator.estimate(
                        frame.frame_rgb,
                        timestamp_ms,
                    )
                    if not isinstance(raw_results, list):
                        raise TypeError(
                            "estimator.estimate() 必须返回 list[RawHandResult]，"
                            f"实际为 {type(raw_results).__name__}"
                        )

                    observations = [
                        self._to_observation(frame, raw_result, detection_id)
                        for detection_id, raw_result in enumerate(raw_results)
                    ]
                except Exception as error:
                    raise HandsPipelineError(
                        f"Hands Pipeline 帧处理失败: {error}",
                        segment_id=self._reader.segment_id,
                        video_stream_id=self._reader.video_stream_id,
                        output_frame_index=frame.output_frame_index,
                        timestamp_ns=frame.timestamp_ns,
                    ) from error

                frames_processed += 1
                observations_created += len(observations)
                if observations:
                    frames_with_hands += 1

                yield from observations
        finally:
            self._stats = PipelineStats(
                frames_processed=frames_processed,
                observations_created=observations_created,
                frames_with_hands=frames_with_hands,
                elapsed_seconds=time.perf_counter() - started_at,
            )

    def _to_observation(
        self,
        frame: PreparedFrame,
        raw_result: RawHandResult,
        detection_id: int,
    ) -> HandObservation:
        if not isinstance(raw_result, RawHandResult):
            raise TypeError(
                f"estimator 结果必须是 RawHandResult，实际为 {type(raw_result).__name__}"
            )

        return HandObservation(
            segment_id=self._reader.segment_id,
            video_stream_id=self._reader.video_stream_id,
            output_frame_index=frame.output_frame_index,
            timestamp_ns=frame.timestamp_ns,
            source_frame_index=frame.source_frame_index,
            source_timestamp_ns=frame.source_timestamp_ns,
            detection_id=detection_id,
            handedness=self._normalize_handedness(raw_result.handedness),
            handedness_score=float(raw_result.handedness_score),
            bbox_xyxy=(
                float(raw_result.bbox.x1),
                float(raw_result.bbox.y1),
                float(raw_result.bbox.x2),
                float(raw_result.bbox.y2),
            ),
            keypoints_2d=[(float(x), float(y)) for x, y in raw_result.keypoints.pixel],
            keypoints_z_relative=[float(z) for _, _, z in raw_result.keypoints.normalized],
            model_name=self._model_name,
            model_version=self._model_version,
        )

    @staticmethod
    def _model_timestamp_ms(
        timestamp_ns: int,
        previous_timestamp_ms: int | None,
    ) -> int:
        timestamp_ms = timestamp_ns // 1_000_000
        if previous_timestamp_ms is not None:
            timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
        return timestamp_ms

    @staticmethod
    def _normalize_handedness(value: str) -> Handedness:
        normalized = value.strip().lower()
        if normalized == "left":
            return "left"
        if normalized == "right":
            return "right"
        return "unknown"


__all__ = [
    "HandEstimator",
    "HandsPipeline",
    "HandsPipelineError",
    "PipelineStats",
    "PreparedFrameSource",
]
