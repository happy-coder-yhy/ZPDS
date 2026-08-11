"""Prepared RGB 帧到统一手部观测的 Hands V1 Pipeline。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from zpds.hands.contracts import (
    FrameInferenceRecord,
    HandEstimator,
    InferenceStatus,
    RunFrameStatistics,
)
from zpds.hands.schemas import (
    Handedness,
    HandFrameResult,
    HandObservation,
    ModelAttemptResult,
    PreparedFrame,
    RawHandResult,
)

# 跨帧 batch 推理的缓冲帧数（estimator 支持 estimate_batch 时启用）
_BATCH_FRAMES = 16


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
    frames_no_hand: int = 0
    frames_failed: int = 0
    frames_skipped_invalid_input: int = 0
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
        active_backend: str | None = None,
        max_frames: int | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name 不能为空")
        if not model_version.strip():
            raise ValueError("model_version 不能为空")
        if active_backend is not None and not active_backend.strip():
            raise ValueError("active_backend 不能为空")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames 必须大于 0")
        self._reader = reader
        self._estimator = estimator
        self._model_name = model_name
        self._model_version = model_version
        self._active_backend = active_backend or model_name
        self._max_frames = max_frames
        self._stats = PipelineStats()
        self._frame_statistics = RunFrameStatistics()
        self._frame_errors: dict[int, Exception] = {}
        self._started = False

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    @property
    def frame_statistics(self) -> RunFrameStatistics:
        """逐帧状态计数；完整消费 Pipeline 后可直接写入 manifest。"""
        return self._frame_statistics

    def run_to_list(self) -> list[HandObservation]:
        """完整运行 Pipeline 并返回截图约定的 ``list[HandObservation]``。

        大数据生产写出仍建议直接把 Pipeline 作为迭代器交给 Writer，以减少
        中间结果驻留内存。
        """
        return list(self)

    def run_frames_to_list(self) -> list[FrameInferenceRecord]:
        """完整运行并返回每个 Prepared 帧对应的一条状态记录。"""
        return list(self.run_frames())

    def __iter__(self) -> Iterator[HandObservation]:
        """兼容 Hands V1：展平 detected 帧，失败帧保持 fail-fast。"""
        for record in self.run_frames():
            yield from self.observations_for_record(
                record,
                fail_on_error=True,
            )

    def observations_for_record(
        self,
        record: FrameInferenceRecord,
        *,
        fail_on_error: bool,
    ) -> tuple[HandObservation, ...]:
        """将逐帧记录转换成 Hands V1 观测。

        WiLoR 编排可传 ``fail_on_error=False`` 保留失败后继续；MediaPipe
        兼容路径传 ``True``，维持原有 fail-fast 行为。
        """
        hands = record.effective_hands or record.raw_hands
        if (
            record.inference_status in {"failed", "skipped_invalid_input"}
            and not hands
        ):
            if not fail_on_error:
                return ()
            error = HandsPipelineError(
                f"Hands Pipeline 帧处理失败: {record.failure_reason}",
                segment_id=self._reader.segment_id,
                video_stream_id=self._reader.video_stream_id,
                output_frame_index=record.frame.output_frame_index,
                timestamp_ns=record.frame.timestamp_ns,
            )
            cause = self._frame_errors.get(record.frame.output_frame_index)
            if cause is not None:
                raise error from cause
            raise error

        return tuple(
            self._to_observation(
                record.frame,
                raw_result,
                detection_id,
                frame_result=record.frame_result,
            )
            for detection_id, raw_result in enumerate(hands)
        )

    def run_frames(self) -> Iterator[FrameInferenceRecord]:
        """逐帧执行模型，单帧失败留痕后继续处理后续帧。

        正常返回空列表记为 ``no_hand``，非空列表记为 ``detected``；
        estimator 或后处理的单帧异常记为 ``failed``。Reader 初始化、视频
        解码和 Sample Map 对齐等输入流级错误仍向上抛出，使整个 run 失败。
        """
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
        frames_no_hand = 0
        frames_failed = 0
        frames_skipped_invalid_input = 0
        previous_timestamp_ms: int | None = None
        started_at = time.perf_counter()

        # 跨帧 batch：estimator 同时具备 estimate_batch 且声明支持时启用，
        # 否则退回逐帧（MediaPipe 等轻量后端不受影响）。
        _estimate_batch = getattr(self._estimator, "estimate_batch", None)
        batchable = callable(_estimate_batch) and bool(
            getattr(self._estimator, "supports_batch", False)
        )
        pending: list[tuple[PreparedFrame, int]] = []

        def _apply_record(record: FrameInferenceRecord) -> None:
            """更新帧级统计（单帧与批量路径共用）。"""
            nonlocal observations_created, frames_with_hands
            nonlocal frames_no_hand, frames_failed, frames_skipped_invalid_input
            if record.effective_hands:
                frames_with_hands += 1
                observations_created += len(record.effective_hands)
            if record.inference_status == "no_hand":
                frames_no_hand += 1
            elif record.inference_status == "failed":
                frames_failed += 1
            elif record.inference_status == "skipped_invalid_input":
                frames_skipped_invalid_input += 1

        try:
            for frame in self._reader:
                if (
                    self._max_frames is not None
                    and frames_processed >= self._max_frames
                ):
                    break
                timestamp_ms = self._model_timestamp_ms(
                    frame.timestamp_ns,
                    previous_timestamp_ms,
                )
                previous_timestamp_ms = timestamp_ms

                if batchable:
                    pending.append((frame, timestamp_ms))
                    if len(pending) >= _BATCH_FRAMES:
                        for record in self._process_pending(pending):
                            frames_processed += 1
                            _apply_record(record)
                            self._frame_statistics.add(record)
                            yield record
                        pending = []
                    continue

                record = self._run_single_frame(frame, timestamp_ms)
                frames_processed += 1
                _apply_record(record)
                self._frame_statistics.add(record)
                yield record

            # 尾部不足一批的残帧
            if batchable and pending:
                for record in self._process_pending(pending):
                    frames_processed += 1
                    _apply_record(record)
                    self._frame_statistics.add(record)
                    yield record
        finally:
            self._stats = PipelineStats(
                frames_processed=frames_processed,
                observations_created=observations_created,
                frames_with_hands=frames_with_hands,
                frames_no_hand=frames_no_hand,
                frames_failed=frames_failed,
                frames_skipped_invalid_input=frames_skipped_invalid_input,
                elapsed_seconds=time.perf_counter() - started_at,
            )

    def _run_single_frame(
        self,
        frame: PreparedFrame,
        timestamp_ms: int,
    ) -> FrameInferenceRecord:
        """单帧推理并构造 FrameInferenceRecord。

        try/except 语义与旧主循环一致：模型适配器属于第三方边界，
        任何单帧异常都必须转为 failed，不能使后续 Prepared 帧丢失。
        """
        inference_started_at = time.perf_counter()
        try:
            (
                inference_status,
                primary_results,
                effective_results,
                failure_reason,
                inference_ms,
                frame_result,
            ) = self._estimate_frame(
                frame,
                timestamp_ms,
            )
        except Exception as error:  # noqa: BLE001
            inference_ms = (
                time.perf_counter() - inference_started_at
            ) * 1000.0
            self._frame_errors[frame.output_frame_index] = error
            return FrameInferenceRecord(
                frame=frame,
                inference_status="failed",
                failure_reason=f"{type(error).__name__}: {error}",
                active_backend=self._active_backend,
                inference_ms=inference_ms,
            )
        return FrameInferenceRecord(
            frame=frame,
            inference_status=inference_status,
            raw_hands=tuple(primary_results),
            effective_hands=tuple(effective_results),
            frame_result=frame_result,
            failure_reason=failure_reason,
            active_backend=self._active_backend,
            inference_ms=inference_ms,
        )

    def _process_pending(
        self,
        pending: list[tuple[PreparedFrame, int]],
    ) -> list[FrameInferenceRecord]:
        """对一批缓冲帧执行跨帧批量推理，构造逐帧 FrameInferenceRecord。

        批级异常（运行级/未分类）整批转 failed，不中断后续批次；
        单帧级失败由 estimator 内部逐帧记录为 failed 状态。
        """
        frames = [frame.frame_rgb for frame, _ in pending]
        timestamps_ms = [ts for _, ts in pending]
        try:
            results = self._estimator.estimate_batch(frames, timestamps_ms)
        except Exception as error:  # noqa: BLE001
            for frame, _ in pending:
                self._frame_errors[frame.output_frame_index] = error
            return [
                FrameInferenceRecord(
                    frame=frame,
                    inference_status="failed",
                    failure_reason=f"{type(error).__name__}: {error}",
                    active_backend=self._active_backend,
                    inference_ms=0.0,
                )
                for frame, _ in pending
            ]

        records: list[FrameInferenceRecord] = []
        for (frame, timestamp_ms), result in zip(pending, results):
            if not isinstance(result, HandFrameResult):
                raise TypeError(
                    "estimator.estimate_batch() 必须返回 HandFrameResult，"
                    f"实际为 {type(result).__name__}"
                )
            (
                inference_status,
                primary_results,
                effective_results,
                failure_reason,
                inference_ms,
                frame_result,
            ) = self._extract_frame_result(frame, timestamp_ms, result)
            records.append(
                FrameInferenceRecord(
                    frame=frame,
                    inference_status=inference_status,
                    raw_hands=tuple(primary_results),
                    effective_hands=tuple(effective_results),
                    frame_result=frame_result,
                    failure_reason=failure_reason,
                    active_backend=self._active_backend,
                    inference_ms=inference_ms,
                )
            )
        return records

    def _estimate_frame(
        self,
        frame: PreparedFrame,
        timestamp_ms: int,
    ) -> tuple[
        InferenceStatus,
        list[RawHandResult],
        list[RawHandResult],
        str | None,
        float,
        HandFrameResult,
    ]:
        """Run either the structured WiLoR API or the legacy list API."""
        started_at = time.perf_counter()
        estimate_frame = getattr(self._estimator, "estimate_frame", None)
        if callable(estimate_frame):
            result = estimate_frame(
                frame.frame_rgb,
                timestamp_ms,
            )
            if not isinstance(result, HandFrameResult):
                raise TypeError(
                    "estimator.estimate_frame() 必须返回 HandFrameResult，"
                    f"实际为 {type(result).__name__}"
                )
        else:
            raw_results = self._estimator.estimate(
                frame.frame_rgb,
                timestamp_ms,
            )
            inference_status = "detected" if raw_results else "no_hand"
            effective_results = raw_results
            failure_reason = None
            inference_ms = (time.perf_counter() - started_at) * 1000.0
            result = HandFrameResult(
                timestamp_ms=timestamp_ms,
                requested_model=self._model_name,
                primary=ModelAttemptResult(
                    model_name=self._model_name,
                    backend_name="legacy_protocol",
                    status=inference_status,
                    hands=raw_results,
                    inference_ms=inference_ms,
                    failure_reason=None,
                    model_version=self._model_version,
                    checkpoint_sha256=None,
                    device="legacy_protocol",
                ),
                fallback=None,
                effective_model=(
                    self._model_name if raw_results else None
                ),
                effective_hands=raw_results,
            )

        return self._extract_frame_result(frame, timestamp_ms, result)

    def _extract_frame_result(
        self,
        frame: PreparedFrame,
        timestamp_ms: int,
        result: HandFrameResult,
    ) -> tuple[
        InferenceStatus,
        list[RawHandResult],
        list[RawHandResult],
        str | None,
        float,
        HandFrameResult,
    ]:
        """从 HandFrameResult 提取 6 元组并执行公共契约校验。

        单帧路径（_estimate_frame）与批量路径（_process_pending）共用，
        保证两条路径对模型结果的校验与转换完全一致。
        """
        primary = result.primary
        if primary.status == "not_run":
            raise RuntimeError(
                primary.failure_reason
                or "WiLoR run aborted before frame inference"
            )
        # 先做 list 类型检查再 list()，保证 () / None 等非法输出
        # 被拒（与旧版对原始返回值的校验语义一致）。
        if not isinstance(primary.hands, list):
            raise TypeError(
                "estimator 必须返回 list[RawHandResult]，"
                f"实际为 {type(primary.hands).__name__}"
            )
        if not isinstance(result.effective_hands, list):
            raise TypeError(
                "HandFrameResult.effective_hands 必须是 list[RawHandResult]，"
                f"实际为 {type(result.effective_hands).__name__}"
            )
        inference_status: InferenceStatus = primary.status
        raw_results = list(primary.hands)
        effective_results = list(result.effective_hands)
        failure_reason = primary.failure_reason
        inference_ms = float(primary.inference_ms)

        if not isinstance(raw_results, list):
            raise TypeError(
                "estimator 必须返回 list[RawHandResult]，"
                f"实际为 {type(raw_results).__name__}"
            )
        if not isinstance(effective_results, list):
            raise TypeError(
                "HandFrameResult.effective_hands 必须是 list[RawHandResult]，"
                f"实际为 {type(effective_results).__name__}"
            )
        if any(
            not isinstance(raw_result, RawHandResult)
            for raw_result in [*raw_results, *effective_results]
        ):
            invalid_type = next(
                type(raw_result).__name__
                for raw_result in [*raw_results, *effective_results]
                if not isinstance(raw_result, RawHandResult)
            )
            raise TypeError(
                "estimator 结果必须全部是 RawHandResult，"
                f"发现 {invalid_type}"
            )

        if inference_status == "detected":
            if not raw_results:
                raise ValueError("detected 状态必须至少包含一只手")
            # 在生成 detected 状态前执行公共契约转换，避免把无法写入
            # Hands V1 的后处理异常误记为有效检测。
            for detection_id, raw_result in enumerate(effective_results):
                self._to_observation(
                    frame,
                    raw_result,
                    detection_id,
                    frame_result=result,
                )
        elif raw_results:
            raise ValueError(
                f"{inference_status} 状态不能包含手部结果"
            )

        if inference_status in {"failed", "skipped_invalid_input"}:
            if not (failure_reason or "").strip():
                raise ValueError(
                    f"{inference_status} 状态必须提供 failure_reason"
                )
        elif failure_reason is not None:
            raise ValueError(
                f"{inference_status} 状态不应提供 failure_reason"
            )

        return (
            inference_status,
            raw_results,
            effective_results,
            failure_reason,
            inference_ms,
            result,
        )

    def _to_observation(
        self,
        frame: PreparedFrame,
        raw_result: RawHandResult,
        detection_id: int,
        *,
        frame_result: HandFrameResult | None = None,
    ) -> HandObservation:
        if not isinstance(raw_result, RawHandResult):
            raise TypeError(
                f"estimator 结果必须是 RawHandResult，实际为 {type(raw_result).__name__}"
            )

        active_attempt = (
            self._active_attempt(frame_result)
            if frame_result is not None
            else None
        )
        has_backend_provenance = (
            active_attempt is not None
            and active_attempt.backend_name != "legacy_protocol"
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
            model_name=(
                active_attempt.model_name
                if active_attempt is not None
                else self._model_name
            ),
            model_version=(
                active_attempt.model_version
                if active_attempt is not None
                else self._model_version
            ),
            keypoints_any_clipped=raw_result.keypoints.any_clipped,
            keypoints_clipped_count=raw_result.keypoints.clipped_count,
            backend_requested=(
                frame_result.requested_model
                if frame_result is not None and has_backend_provenance
                else ""
            ),
            backend_active=(
                active_attempt.backend_name
                if has_backend_provenance
                else ""
            ),
            backend_fallback_used=(
                frame_result.fallback_used
                if frame_result is not None
                else False
            ),
            backend_fallback_reason=(
                frame_result.fallback_reason or ""
                if frame_result is not None
                else ""
            ),
        )

    @staticmethod
    def _active_attempt(
        frame_result: HandFrameResult,
    ) -> ModelAttemptResult | None:
        if frame_result.effective_model == frame_result.primary.model_name:
            return frame_result.primary
        if (
            frame_result.fallback is not None
            and frame_result.effective_model
            == frame_result.fallback.model_name
        ):
            return frame_result.fallback
        return None

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
