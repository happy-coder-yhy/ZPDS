"""WiLoR 估计器阶段 3 测试。

验证：
- 四种帧状态：detected / no_hand / failed / skipped_invalid_input
- HandFrameResult 组装：primary 单记录（单后端恒 WiLoR，无回退）
- 帧统计恒等式对齐
- 运行级错误 / 阈值 / Run Report
"""

from __future__ import annotations

import numpy as np
import pytest

from zpds.hands.schemas import (
    HandFrameResult,
    ModelAttemptResult,
    RawHandResult,
)
from zpds.hands.wilor_estimator import (
    WiLoREstimatorConfig,
    WiLoRFrameStats,
    WiLoRHandEstimator,
)
from zpds.hands.wilor_schema import (
    WiLoRDetection,
    WiLoRImageTransform,
    WiLoRInferenceError,
    WiLoRModelInfo,
)


# ════════════════════════════════════════════════════════════════════
# 辅助 — Mock 对象
# ════════════════════════════════════════════════════════════════════


_MOCK_TRANSFORM = WiLoRImageTransform.from_resize(
    original_width=640, original_height=480,
    detector_width=256, detector_height=256,
)


def _make_detection(**overrides) -> WiLoRDetection:
    values = {
        "handedness": "Right",
        "handedness_score": 0.9,
        "detection_score": 0.85,
        "bbox_xyxy_px": (100.0, 150.0, 300.0, 400.0),
        "transform": _MOCK_TRANSFORM,
    }
    values.update(overrides)
    return WiLoRDetection(**values)


def _make_model_info() -> WiLoRModelInfo:
    return WiLoRModelInfo(
        model_version="v1.0",
        checkpoint_sha256="a" * 64,
        device="cpu",
        init_time_ms=500.0,
    )


class _MockAdapter:
    """模拟 WiLoRAdapter，返回可控的 WiLoRDetection 列表。"""

    def __init__(
        self,
        detections: list[WiLoRDetection] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.detections: list[WiLoRDetection] = detections if detections is not None else []
        self.exc = exc
        self.close_called = False
        self.call_count = 0

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[WiLoRDetection]:
        self.call_count += 1
        if self.exc is not None:
            raise self.exc
        return list(self.detections)

    def close(self) -> None:
        self.close_called = True


def _make_frame() -> np.ndarray:
    return np.full((480, 640, 3), 128, dtype=np.uint8)


# ════════════════════════════════════════════════════════════════════
# 状态：detected
# ════════════════════════════════════════════════════════════════════


def test_status_detected() -> None:
    adapter = _MockAdapter(detections=[_make_detection()])
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(_make_frame(), 0)

    assert result.primary.status == "detected"
    assert result.primary.model_name == "wilor"
    assert result.effective_model == "wilor"
    assert not result.fallback_used

    assert estimator.frame_stats.detected == 1
    assert estimator.frame_stats.total_frames == 1
    estimator.close()


# ════════════════════════════════════════════════════════════════════
# 状态：no_hand
# ════════════════════════════════════════════════════════════════════


def test_status_no_hand() -> None:
    adapter = _MockAdapter(detections=[])  # 无检测
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(_make_frame(), 0)

    assert result.primary.status == "no_hand"
    assert result.primary.hands == []
    assert result.effective_model == "wilor"
    assert not result.fallback_used

    assert estimator.frame_stats.no_hand == 1
    estimator.close()


# ════════════════════════════════════════════════════════════════════
# 状态：failed
# ════════════════════════════════════════════════════════════════════


def test_status_failed() -> None:
    adapter = _MockAdapter(exc=WiLoRInferenceError("CUDA OOM"))
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(_make_frame(), 0)

    assert result.primary.status == "failed"
    assert "CUDA OOM" in (result.primary.failure_reason or "")
    assert result.primary.hands == []

    assert estimator.frame_stats.failed == 1
    estimator.close()


def test_status_failed_due_to_not_implemented() -> None:
    """阶段 1/2 占位 — WiLoRAdapter.detect 抛出 NotImplementedError。"""
    adapter = _MockAdapter(exc=WiLoRInferenceError("尚未实现"))
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(_make_frame(), 0)

    assert result.primary.status == "failed"
    assert "尚未实现" in (result.primary.failure_reason or "")
    estimator.close()


# ════════════════════════════════════════════════════════════════════
# 状态：skipped_invalid_input
# ════════════════════════════════════════════════════════════════════


def test_status_skipped_invalid_input_none() -> None:
    estimator = WiLoRHandEstimator(
        adapter=_MockAdapter(),
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(None, 0)  # type: ignore[arg-type]

    assert result.primary.status == "skipped_invalid_input"
    assert result.effective_model is None
    assert not result.fallback_used
    assert estimator.frame_stats.skipped_invalid_input == 1
    estimator.close()


def test_status_skipped_invalid_input_empty_frame() -> None:
    estimator = WiLoRHandEstimator(
        adapter=_MockAdapter(),
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(np.array([], dtype=np.uint8), 0)

    assert result.primary.status == "skipped_invalid_input"
    estimator.close()


def test_status_skipped_invalid_input_wrong_ndim() -> None:
    estimator = WiLoRHandEstimator(
        adapter=_MockAdapter(),
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(np.zeros((480, 640), dtype=np.uint8), 0)

    assert result.primary.status == "skipped_invalid_input"
    estimator.close()


def test_status_skipped_invalid_input_wrong_dtype() -> None:
    estimator = WiLoRHandEstimator(
        adapter=_MockAdapter(),
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(np.zeros((480, 640, 3), dtype=np.float32), 0)

    assert result.primary.status == "skipped_invalid_input"
    estimator.close()


def test_status_skipped_invalid_input_negative_timestamp() -> None:
    estimator = WiLoRHandEstimator(
        adapter=_MockAdapter(),
        model_info=_make_model_info(),
    )
    result = estimator.estimate_frame(_make_frame(), -5)

    assert result.primary.status == "skipped_invalid_input"
    estimator.close()


# ════════════════════════════════════════════════════════════════════
# 时间戳单调性
# ════════════════════════════════════════════════════════════════════


def test_timestamp_strictly_increasing() -> None:
    adapter = _MockAdapter(detections=[_make_detection()])
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    r1 = estimator.estimate_frame(_make_frame(), 0)
    assert r1.primary.status == "detected"

    r2 = estimator.estimate_frame(_make_frame(), 33)
    assert r2.primary.status == "detected"

    # 相同时间戳 → invalid
    r3 = estimator.estimate_frame(_make_frame(), 33)
    assert r3.primary.status == "skipped_invalid_input"

    # 倒退 → invalid
    r4 = estimator.estimate_frame(_make_frame(), 10)
    assert r4.primary.status == "skipped_invalid_input"
    estimator.close()


# ════════════════════════════════════════════════════════════════════
# 帧统计恒等式
# ════════════════════════════════════════════════════════════════════


def test_frame_stats_invariant() -> None:
    """total = detected + no_hand + failed + skipped_invalid_input"""
    adapter = _MockAdapter(detections=[_make_detection()])
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )

    # detected
    estimator.estimate_frame(_make_frame(), 0)
    # no_hand — 需要换 adapter
    estimator2 = WiLoRHandEstimator(
        adapter=_MockAdapter(detections=[]),
        model_info=_make_model_info(),
    )
    estimator2.estimate_frame(_make_frame(), 0)
    # failed
    estimator3 = WiLoRHandEstimator(
        adapter=_MockAdapter(exc=WiLoRInferenceError("err")),
        model_info=_make_model_info(),
    )
    estimator3.estimate_frame(_make_frame(), 0)
    # skipped
    estimator4 = WiLoRHandEstimator(
        adapter=_MockAdapter(),
        model_info=_make_model_info(),
    )
    estimator4.estimate_frame(None, 0)  # type: ignore[arg-type]

    s1 = estimator.frame_stats
    s2 = estimator2.frame_stats
    s3 = estimator3.frame_stats
    s4 = estimator4.frame_stats

    assert s1.total_frames == 1 and s1.detected == 1
    assert s2.total_frames == 1 and s2.no_hand == 1
    assert s3.total_frames == 1 and s3.failed == 1
    assert s4.total_frames == 1 and s4.skipped_invalid_input == 1

    for stats in (s1, s2, s3, s4):
        assert stats.total_frames == (
            stats.detected + stats.no_hand + stats.failed + stats.skipped_invalid_input
        )
    estimator.close()
    estimator2.close()
    estimator3.close()
    estimator4.close()


# ════════════════════════════════════════════════════════════════════
# 前后一致
# ════════════════════════════════════════════════════════════════════


def test_estimate_method_compat() -> None:
    """estimate() 兼容 HandEstimator Protocol，返回 effective_hands。"""
    adapter = _MockAdapter(detections=[_make_detection()])
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    hands = estimator.estimate(_make_frame(), 0)

    assert isinstance(hands, list)
    # 21 点映射未验收前恒为空
    assert hands == []
    estimator.close()


def test_close_calls_adapter() -> None:
    adapter = _MockAdapter()
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    estimator.close()

    assert adapter.close_called


# ════════════════════════════════════════════════════════════════════
# 阶段 6：运行级错误 / 阈值 / Run Report
# ════════════════════════════════════════════════════════════════════


def test_consecutive_failures_trigger_abort() -> None:
    adapter = _MockAdapter(exc=WiLoRInferenceError("fail"))
    from zpds.hands.wilor_schema import WiLoRRunThresholds
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
        thresholds=WiLoRRunThresholds(
            max_consecutive_frame_failures=3,
            max_failure_ratio=1.0,  # 禁用比例检查
        ),
    )
    for ts in range(3):
        result = estimator.estimate_frame(_make_frame(), ts)
        assert result.primary.status == "failed"

    assert estimator.is_aborted
    assert "连续失败 3 帧" in (estimator.abort_reason or "")


def test_aborted_run_skips_frames() -> None:
    adapter = _MockAdapter(exc=WiLoRInferenceError("fail"))
    from zpds.hands.wilor_schema import WiLoRRunThresholds
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
        thresholds=WiLoRRunThresholds(
            max_consecutive_frame_failures=2,
            max_failure_ratio=1.0,
        ),
    )
    estimator.estimate_frame(_make_frame(), 0)  # fail 1
    estimator.estimate_frame(_make_frame(), 33)  # fail 2 → abort

    # 第 3 帧被跳过
    result = estimator.estimate_frame(_make_frame(), 66)
    assert result.primary.status == "not_run"
    assert result.effective_model is None
    estimator.close()


def test_run_level_error_raises() -> None:
    """运行级异常直接抛出，不转化为 failed。"""
    from zpds.hands.wilor_schema import CheckpointIntegrityError
    adapter = _MockAdapter(exc=CheckpointIntegrityError("checkpoint corrupted"))
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    with pytest.raises(CheckpointIntegrityError):
        estimator.estimate_frame(_make_frame(), 0)
    estimator.close()


def test_build_run_report() -> None:
    adapter = _MockAdapter(detections=[_make_detection()])
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    estimator.estimate_frame(_make_frame(), 0)
    report = estimator.build_run_report()

    assert report.requested_model == "wilor"
    assert report.coverage["decoded_frames"] == 1
    assert report.coverage["detected_frames"] == 1
    assert report.quality["coordinate_frame"] == "model_camera"
    assert report.quality["scale_status"] == "uncalibrated"
    assert report.completion["wilor_full_frame_requirement_met"]
    estimator.close()


def test_run_report_records_failures() -> None:
    adapter = _MockAdapter(exc=WiLoRInferenceError("OOM"))
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
    )
    estimator.estimate_frame(_make_frame(), 0)
    report = estimator.build_run_report()

    assert report.coverage["failed_frames"] == 1
    assert not report.completion["wilor_full_frame_requirement_met"]
    # 错误记录在 errors 列表中
    assert len(report.errors) == 1
    assert "OOM" in report.errors[0]["failure_reason"]
    estimator.close()


def test_classify_error_frame_level() -> None:
    from zpds.hands.wilor_schema import WiLoRInferenceError, CoordinateTransformError
    assert WiLoRHandEstimator.classify_error(WiLoRInferenceError("")) == "frame_level"
    assert WiLoRHandEstimator.classify_error(CoordinateTransformError("")) == "frame_level"


def test_classify_error_run_level() -> None:
    from zpds.hands.wilor_schema import CheckpointIntegrityError, WiLoRInitializationError
    assert WiLoRHandEstimator.classify_error(CheckpointIntegrityError("")) == "run_level"
    assert WiLoRHandEstimator.classify_error(WiLoRInitializationError("")) == "run_level"


def test_run_thresholds_defaults() -> None:
    from zpds.hands.wilor_schema import WiLoRRunThresholds
    t = WiLoRRunThresholds()
    assert t.max_consecutive_frame_failures == 5
    assert t.max_failure_ratio == 0.02


def test_run_thresholds_rejects_invalid() -> None:
    from zpds.hands.wilor_schema import WiLoRRunThresholds
    with pytest.raises(ValueError):
        WiLoRRunThresholds(max_consecutive_frame_failures=0)
    with pytest.raises(ValueError):
        WiLoRRunThresholds(max_failure_ratio=0.0)
    with pytest.raises(ValueError):
        WiLoRRunThresholds(max_failure_ratio=1.5)
