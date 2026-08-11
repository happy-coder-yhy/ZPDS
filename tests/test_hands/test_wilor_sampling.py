"""WiLoR 抽帧（bbox_fps 时间窗）测试。

验证：
- ego_bbox_every_frame=True（默认）每帧推理，行为不变
- False 时按 1000/bbox_fps ms 时间窗推理，中间帧复用上一推理帧结果
- 首帧强制推理
- 逐帧 / 批量两条路径的传播语义一致
- 统计恒等式与 report 如实反映推理/传播
"""

from __future__ import annotations

import numpy as np
import pytest

from zpds.hands.schemas import RawHandResult
from zpds.hands.wilor_estimator import (
    WiLoREstimatorConfig,
    WiLoRHandEstimator,
)
from zpds.hands.wilor_schema import (
    WiLoRDetection,
    WiLoRFallbackPolicy,
    WiLoRImageTransform,
    WiLoRModelInfo,
)


# ════════════════════════════════════════════════════════════════════
# 辅助
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


class _CountingAdapter:
    """统计 detect / detect_batch 调用次数与输入帧数。"""

    def __init__(
        self,
        detections: list[WiLoRDetection] | None = None,
    ) -> None:
        self.detections = detections if detections is not None else []
        self.detect_count = 0
        self.detect_batch_count = 0
        self.detect_batch_frames: list[int] = []

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[WiLoRDetection]:
        self.detect_count += 1
        return list(self.detections)

    def detect_batch(
        self,
        frames_rgb: list[np.ndarray],
        timestamps_ms: list[int],
    ) -> list[list[WiLoRDetection]]:
        self.detect_batch_count += 1
        self.detect_batch_frames = list(timestamps_ms)
        return [list(self.detections) for _ in frames_rgb]

    def close(self) -> None:
        pass


def _make_frame() -> np.ndarray:
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def _make_estimator(
    adapter: _CountingAdapter,
    *,
    ego_bbox_every_frame: bool,
    bbox_fps: float = 10.0,
) -> WiLoRHandEstimator:
    return WiLoRHandEstimator(
        adapter=adapter,
        model_info=_make_model_info(),
        fallback_estimator=None,
        config=WiLoREstimatorConfig(
            fallback_policy=WiLoRFallbackPolicy(
                on_wilor_init_failure=False,
                on_wilor_frame_failure=False,
                on_wilor_no_hand=False,
                on_invalid_input=False,
            ),
            ego_bbox_every_frame=ego_bbox_every_frame,
            bbox_fps=bbox_fps,
        ),
    )


# 30fps 帧时间戳（每帧 ~33ms）
_FRAMES_30FPS_7 = [0, 33, 66, 100, 133, 166, 200]


# ════════════════════════════════════════════════════════════════════
# 默认每帧模式：行为不变
# ════════════════════════════════════════════════════════════════════


def test_every_frame_default_infers_all_frames() -> None:
    adapter = _CountingAdapter(detections=[_make_detection()])
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=True, bbox_fps=10.0,
    )

    results = [
        estimator.estimate_frame(_make_frame(), ts)
        for ts in _FRAMES_30FPS_7
    ]

    assert adapter.detect_count == 7  # 每帧推理
    assert all(r.primary.propagated is False for r in results)
    assert estimator.frame_stats.propagated_frames == 0


# ════════════════════════════════════════════════════════════════════
# 抽帧：逐帧路径
# ════════════════════════════════════════════════════════════════════


def test_sampling_frame_path_infers_on_window() -> None:
    """bbox_fps=10 → 100ms 窗口：0/100/200 推理，中间帧传播。"""
    adapter = _CountingAdapter(detections=[_make_detection()])
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=False, bbox_fps=10.0,
    )

    results = [
        estimator.estimate_frame(_make_frame(), ts)
        for ts in _FRAMES_30FPS_7
    ]

    assert adapter.detect_count == 3  # 0 / 100 / 200
    propagated = [i for i, r in enumerate(results) if r.primary.propagated]
    assert propagated == [1, 2, 4, 5]
    # 传播帧无推理耗时，但状态与结果与推理帧一致（detected）
    for i in propagated:
        assert results[i].primary.inference_ms == 0.0
        assert results[i].primary.status == "detected"
        assert results[i].primary.hands == results[0].primary.hands
    # 推理帧保留真实耗时
    for i in [0, 3, 6]:
        assert results[i].primary.inference_ms > 0.0
        assert results[i].primary.propagated is False


def test_sampling_first_frame_forced() -> None:
    """首帧（无可用结果）强制推理，不依赖时间窗。"""
    adapter = _CountingAdapter()
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=False, bbox_fps=0.1,  # 窗口 10s
    )

    result = estimator.estimate_frame(_make_frame(), timestamp_ms=0)

    assert adapter.detect_count == 1
    assert result.primary.propagated is False


def test_sampling_no_hand_propagates_no_hand() -> None:
    """无手帧的传播：status 保持 no_hand，统计正确。"""
    adapter = _CountingAdapter(detections=[])
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=False, bbox_fps=10.0,
    )

    results = [
        estimator.estimate_frame(_make_frame(), ts)
        for ts in _FRAMES_30FPS_7
    ]

    assert adapter.detect_count == 3
    assert all(r.primary.status == "no_hand" for r in results)
    stats = estimator.frame_stats
    assert stats.no_hand == 7
    assert stats.detected == 0
    assert stats.propagated_frames == 4


# ════════════════════════════════════════════════════════════════════
# 抽帧：批量路径
# ════════════════════════════════════════════════════════════════════


def test_sampling_batch_path_infers_subset() -> None:
    """estimate_batch 只把推理帧交给 detect_batch，其余传播。"""
    adapter = _CountingAdapter(detections=[_make_detection()])
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=False, bbox_fps=10.0,
    )

    results = estimator.estimate_batch(
        [_make_frame() for _ in _FRAMES_30FPS_7],
        list(_FRAMES_30FPS_7),
    )

    assert adapter.detect_batch_count == 1
    assert adapter.detect_batch_frames == [0, 100, 200]  # 只传推理帧
    propagated = [i for i, r in enumerate(results) if r.primary.propagated]
    assert propagated == [1, 2, 4, 5]
    for i in propagated:
        assert results[i].primary.inference_ms == 0.0
    assert estimator.frame_stats.propagated_frames == 4


def test_sampling_batch_tail_after_frame_path() -> None:
    """逐帧与批量混用：传播状态跨路径保持一致。"""
    adapter = _CountingAdapter(detections=[_make_detection()])
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=False, bbox_fps=10.0,
    )

    estimator.estimate_frame(_make_frame(), timestamp_ms=0)  # 推理
    results = estimator.estimate_batch(
        [_make_frame(), _make_frame(), _make_frame()],
        [33, 66, 100],  # 前两帧传播，100 命中窗口再推理
    )

    assert adapter.detect_count == 1
    assert adapter.detect_batch_frames == [100]
    assert results[0].primary.propagated is True
    assert results[1].primary.propagated is True
    assert results[2].primary.propagated is False


# ════════════════════════════════════════════════════════════════════
# 统计与报告
# ════════════════════════════════════════════════════════════════════


def test_sampling_stats_invariant() -> None:
    """统计恒等式：total = detected + no_hand + failed + skipped。"""
    adapter = _CountingAdapter(detections=[_make_detection()])
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=False, bbox_fps=10.0,
    )

    for ts in _FRAMES_30FPS_7:
        estimator.estimate_frame(_make_frame(), ts)

    stats = estimator.frame_stats
    assert stats.total_frames == 7
    assert stats.detected == 7  # 传播帧也计入（保持 coverage 语义）
    assert stats.propagated_frames == 4
    assert (
        stats.total_frames
        == stats.detected + stats.no_hand + stats.failed
        + stats.skipped_invalid_input
    )
    # 推理耗时只累计真实推理帧
    assert stats.total_inference_ms > 0.0
    assert stats.avg_inference_ms > 0.0


def test_sampling_report_reflects_config() -> None:
    """build_run_report：配置与推理/传播计数如实上报。"""
    adapter = _CountingAdapter(detections=[_make_detection()])
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=False, bbox_fps=10.0,
    )
    for ts in _FRAMES_30FPS_7:
        estimator.estimate_frame(_make_frame(), ts)

    report = estimator.build_run_report()

    assert report.ego_bbox_every_frame is False
    assert report.coverage["decoded_frames"] == 7
    assert report.coverage["wilor_requests"] == 3  # 真实推理次数
    assert report.coverage["propagated_frames"] == 4
    assert report.coverage["detected_frames"] == 7


def test_every_frame_report_unchanged() -> None:
    """每帧模式 report 与历史一致（propagated=0, wilor_requests=total）。"""
    adapter = _CountingAdapter(detections=[_make_detection()])
    estimator = _make_estimator(
        adapter, ego_bbox_every_frame=True, bbox_fps=10.0,
    )
    for ts in _FRAMES_30FPS_7:
        estimator.estimate_frame(_make_frame(), ts)

    report = estimator.build_run_report()

    assert report.ego_bbox_every_frame is True
    assert report.coverage["wilor_requests"] == 7
    assert report.coverage["propagated_frames"] == 0
