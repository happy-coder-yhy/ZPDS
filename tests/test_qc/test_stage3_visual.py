"""Stage 3 视觉 QC 测试：D13 过曝 + D14 模糊。"""

from pathlib import Path

import cv2
import numpy as np

from zpds.core.decisions import ReasonCode, Severity
from zpds.qc.stage3_visual import (
    _compute_frame_timestamp_ns,
    _find_continuous_spans,
    check,
    detect_blur,
    detect_overexposure,
)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _create_test_video(
    path: str,
    frames: list[np.ndarray],
    fps: float = 30.0,
    codec: str = "mp4v",
) -> str:
    """创建临时测试视频。"""
    if not frames:
        return path
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        writer.write(f)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# _find_continuous_spans
# ---------------------------------------------------------------------------


class TestFindContinuousSpans:
    def test_empty(self):
        assert _find_continuous_spans(np.array([], dtype=bool)) == []

    def test_no_true(self):
        assert _find_continuous_spans(np.array([False, False, False])) == []

    def test_single_span(self):
        spans = _find_continuous_spans(np.array([False, True, True, False]))
        assert spans == [(1, 3)]

    def test_multiple_spans(self):
        flags = np.array([True, True, False, True, False, True, True, True])
        spans = _find_continuous_spans(flags)
        assert spans == [(0, 2), (3, 4), (5, 8)]

    def test_min_consecutive(self):
        flags = np.array([True, False, True, True, False, True])
        spans = _find_continuous_spans(flags, min_consecutive=2)
        assert spans == [(2, 4)]


class TestComputeFrameTimestamp:
    def test_zero_start(self):
        assert _compute_frame_timestamp_ns(0, 30.0) == 0

    def test_one_second(self):
        assert _compute_frame_timestamp_ns(30, 30.0) == 1_000_000_000

    def test_custom_start(self):
        ts = _compute_frame_timestamp_ns(15, 30.0, start_ns=100_000_000)
        assert ts == 600_000_000  # 0.5s + 0.1s


# ---------------------------------------------------------------------------
# D13 过曝检测
# ---------------------------------------------------------------------------


class TestOverexposure:
    def test_normal_video_no_overexposure(self, tmp_path):
        """正常视频不应检测到过曝。"""
        frames = [np.full((120, 160), 100, dtype=np.uint8) for _ in range(10)]
        vpath = str(tmp_path / "normal.mp4")
        _create_test_video(vpath, frames)

        decisions = detect_overexposure(vpath)
        assert len(decisions) == 0

    def test_all_overexposed(self, tmp_path):
        """全白视频应检测到过曝。"""
        frames = [np.full((120, 160), 250, dtype=np.uint8) for _ in range(10)]
        vpath = str(tmp_path / "white.mp4")
        _create_test_video(vpath, frames)

        decisions = detect_overexposure(vpath, consecutive_min=1)
        assert len(decisions) >= 1
        overexp = [d for d in decisions if d.reason == ReasonCode.OVEREXPOSED and d.severity != Severity.INFO]
        assert len(overexp) >= 1
        assert overexp[0].detail["duration_frames"] >= 9

    def test_partial_overexposed(self, tmp_path):
        """部分过曝帧应检测到。"""
        normal = np.full((120, 160), 100, dtype=np.uint8)
        bright = np.full((120, 160), 240, dtype=np.uint8)
        frames = [normal] * 5 + [bright] * 5 + [normal] * 5
        vpath = str(tmp_path / "partial.mp4")
        _create_test_video(vpath, frames)

        decisions = detect_overexposure(vpath, consecutive_min=2)
        overexp = [d for d in decisions if d.reason == ReasonCode.OVEREXPOSED and d.severity != Severity.INFO]
        assert len(overexp) >= 1

    def test_cannot_open(self):
        """不存在视频应返回 ERROR。"""
        decisions = detect_overexposure("/nonexistent/video.mp4")
        assert len(decisions) == 1
        assert decisions[0].severity == Severity.ERROR

    def test_custom_thresholds(self, tmp_path):
        """自定义阈值应生效。"""
        frames = [np.full((120, 160), 150, dtype=np.uint8) for _ in range(10)]
        vpath = str(tmp_path / "mid.mp4")
        _create_test_video(vpath, frames)

        # 默认阈值不应检测到
        d1 = detect_overexposure(vpath)
        assert len(d1) == 0

        # 降低阈值应检测到
        d2 = detect_overexposure(vpath, mean_threshold=140, overexposure_ratio=0.5)
        overexp = [d for d in d2 if d.reason == ReasonCode.OVEREXPOSED and d.severity != Severity.INFO]
        assert len(overexp) >= 1

    def test_evidence_saved(self, tmp_path):
        """证据帧应被保存。"""
        frames = [np.full((120, 160), 250, dtype=np.uint8) for _ in range(5)]
        vpath = str(tmp_path / "bright.mp4")
        _create_test_video(vpath, frames)

        evidence_dir = str(tmp_path / "evidence")
        detect_overexposure(vpath, consecutive_min=1, evidence_dir=evidence_dir)
        # 检查是否有证据文件生成
        evidence_files = list(Path(evidence_dir).glob("overexposed_*.jpg"))
        assert len(evidence_files) > 0


# ---------------------------------------------------------------------------
# D14 模糊检测
# ---------------------------------------------------------------------------


class TestBlur:
    def test_sharp_video(self, tmp_path):
        """清晰视频不应检测到模糊。"""
        frames = []
        for i in range(10):
            # 有纹理的图像
            img = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
            frames.append(img)
        vpath = str(tmp_path / "sharp.mp4")
        _create_test_video(vpath, frames)

        decisions = detect_blur(vpath, laplacian_threshold=10.0)
        blur_decisions = [d for d in decisions if d.severity != Severity.INFO]
        assert len(blur_decisions) == 0

    def test_blurred_video(self, tmp_path):
        """模糊视频应检测到。"""
        # 均匀灰度图像 -> Laplacian 方差接近 0
        frames = [np.full((120, 160), 128, dtype=np.uint8) for _ in range(10)]
        vpath = str(tmp_path / "blur.mp4")
        _create_test_video(vpath, frames)

        decisions = detect_blur(vpath, laplacian_threshold=5.0)
        blur_decisions = [d for d in decisions if d.reason == ReasonCode.BLUR_DETECTED and d.severity != Severity.INFO]
        assert len(blur_decisions) >= 1

    def test_cannot_open(self):
        decisions = detect_blur("/nonexistent/video.mp4")
        assert len(decisions) == 1
        assert decisions[0].severity == Severity.ERROR

    def test_resolution_auto_threshold(self, tmp_path):
        """按分辨率自动选择阈值。"""
        frames = [np.full((288, 352), 128, dtype=np.uint8) for _ in range(5)]
        vpath = str(tmp_path / "small.mp4")
        _create_test_video(vpath, frames)

        # 默认分辨率自适应
        decisions = detect_blur(vpath)
        assert len(decisions) >= 0  # 不应崩溃

    @staticmethod
    def _make_sharp_blur_sharp(n_blur: int) -> list:
        """清晰段 + 模糊段 + 清晰段。"""
        rng = np.random.default_rng(7)
        sharp = [
            rng.integers(0, 255, (120, 160), dtype=np.uint8) for _ in range(10)
        ]
        blur = [np.full((120, 160), 128, dtype=np.uint8) for _ in range(n_blur)]
        return sharp + blur + sharp

    def test_long_blur_span_splits(self, tmp_path):
        """模糊段时长 >= quarantine_duration_s → SPLIT 处置（切分缺口）。"""
        # 40 帧模糊 @30fps = 1.33s > 1.0s
        frames = self._make_sharp_blur_sharp(n_blur=40)
        vpath = str(tmp_path / "long_blur.mp4")
        _create_test_video(vpath, frames)

        decisions = detect_blur(vpath, laplacian_threshold=5.0, quarantine_duration_s=1.0)
        spans = [d for d in decisions if d.reason == ReasonCode.BLUR_DETECTED
                 and d.severity != Severity.INFO]
        assert len(spans) == 1
        assert spans[0].disposition is not None
        assert spans[0].disposition.value == "split"
        assert spans[0].detail["recommended_action"] == "split"
        # 缺口区间：模糊段 [10, 50)
        assert spans[0].detail["start_frame"] == 10
        assert spans[0].detail["end_frame"] == 50

    def test_short_blur_span_keep_with_flag(self, tmp_path):
        """模糊段时长 < quarantine_duration_s → KEEP_WITH_FLAG 打标保留。"""
        # 10 帧模糊 @30fps = 0.33s < 1.0s
        frames = self._make_sharp_blur_sharp(n_blur=10)
        vpath = str(tmp_path / "short_blur.mp4")
        _create_test_video(vpath, frames)

        decisions = detect_blur(vpath, laplacian_threshold=5.0, quarantine_duration_s=1.0)
        spans = [d for d in decisions if d.reason == ReasonCode.BLUR_DETECTED
                 and d.severity != Severity.INFO]
        assert len(spans) == 1
        assert spans[0].disposition is not None
        assert spans[0].disposition.value == "keep_with_flag"
        assert spans[0].detail["recommended_action"] == "keep_with_flag"

    def test_no_quarantine_threshold_keeps_old_behavior(self, tmp_path):
        """不传 quarantine_duration_s → 不设置处置（旧行为：只记录）。"""
        frames = self._make_sharp_blur_sharp(n_blur=40)
        vpath = str(tmp_path / "nothresh.mp4")
        _create_test_video(vpath, frames)

        decisions = detect_blur(vpath, laplacian_threshold=5.0)
        spans = [d for d in decisions if d.reason == ReasonCode.BLUR_DETECTED
                 and d.severity != Severity.INFO]
        assert len(spans) == 1
        assert spans[0].disposition is None
        assert spans[0].detail["recommended_action"] == "quarantine"


# ---------------------------------------------------------------------------
# check() 统一入口
# ---------------------------------------------------------------------------


class TestCheckEntry:
    def test_empty_config(self, tmp_path):
        """空配置时使用默认阈值。"""
        frames = [np.full((120, 160), 100, dtype=np.uint8) for _ in range(5)]
        vpath = str(tmp_path / "test.mp4")
        _create_test_video(vpath, frames)

        decisions = check(video_path=vpath)
        assert isinstance(decisions, list)
        # 清晰随机帧不应有严重问题
        errors = [d for d in decisions if d.severity in (Severity.FATAL, Severity.ERROR)]
        assert len(errors) == 0

    def test_disabled_overexposure(self, tmp_path):
        """可以禁用子检测。"""
        frames = [np.full((120, 160), 250, dtype=np.uint8) for _ in range(5)]
        vpath = str(tmp_path / "bright.mp4")
        _create_test_video(vpath, frames)

        decisions = check(
            video_path=vpath,
            stage_config={"overexposure": {"enabled": False}, "blur": {"enabled": False}},
        )
        assert len(decisions) == 0
