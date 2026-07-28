"""Stage 5 深度有效性 QC 测试（D15）。"""

import numpy as np
import pytest

from zpds.core.decisions import ReasonCode, Severity
from zpds.qc.stage5_depth import (
    check,
    check_depth_frame,
    check_depth_sequence,
    check_rgb_depth_pairing,
)

# ---------------------------------------------------------------------------
# check_depth_frame
# ---------------------------------------------------------------------------


class TestCheckDepthFrame:
    def test_valid_frame(self):
        """全有效深度帧。"""
        depth = np.full((480, 640), 1000, dtype=np.uint16)
        stats = check_depth_frame(depth)
        assert stats["zero_ratio"] == 0.0
        assert stats["invalid_ratio"] == 0.0
        assert stats["valid_ratio"] == 1.0
        assert not stats["all_zero"]
        assert stats["mean"] == 1000.0

    def test_all_zero(self):
        """全零深度帧。"""
        depth = np.zeros((480, 640), dtype=np.uint16)
        stats = check_depth_frame(depth)
        assert stats["zero_ratio"] == 1.0
        assert stats["all_zero"]
        assert stats["valid_ratio"] == 0.0

    def test_partial_zero(self):
        """部分零值帧。"""
        depth = np.ones((100, 100), dtype=np.uint16) * 500
        depth[:30, :] = 0  # 30% zero
        stats = check_depth_frame(depth)
        assert stats["zero_ratio"] == pytest.approx(0.30, abs=0.01)
        assert stats["valid_ratio"] == pytest.approx(0.70, abs=0.01)
        assert not stats["all_zero"]

    def test_empty_array(self):
        stats = check_depth_frame(np.array([], dtype=np.uint16))
        assert stats["zero_ratio"] == 1.0
        assert stats["all_zero"]

    def test_custom_invalid_value(self):
        """自定义无效值标记。"""
        depth = np.ones((50, 50), dtype=np.uint16) * 1000
        depth[:20, :] = 0         # 40% zero
        depth[20:35, :] = 9999    # 30% marked invalid
        stats = check_depth_frame(depth, invalid_value=9999)
        # zero + invalid = 40% + 30% = 70%
        assert stats["invalid_ratio"] == pytest.approx(0.70, abs=0.01)


# ---------------------------------------------------------------------------
# check_depth_sequence
# ---------------------------------------------------------------------------


class TestCheckDepthSequence:
    def test_empty_sequence(self):
        decisions = check_depth_sequence([])
        assert len(decisions) == 1
        assert decisions[0].message == "Depth sequence is empty"

    def test_valid_sequence(self):
        """全部有效的深度序列。"""
        frames = [np.full((240, 320), 2000, dtype=np.uint16) for _ in range(10)]
        decisions = check_depth_sequence(frames)
        fatal = [d for d in decisions if d.severity == Severity.FATAL]
        assert len(fatal) == 0

    def test_all_zero_sequence(self):
        """全零深度序列应触发 FATAL。"""
        frames = [np.zeros((240, 320), dtype=np.uint16) for _ in range(10)]
        decisions = check_depth_sequence(frames)
        fatal = [d for d in decisions if d.severity == Severity.FATAL]
        assert len(fatal) >= 1

    def test_high_invalid_ratio(self):
        """高无效比例应触发 WARN。"""
        frames = []
        for _ in range(10):
            f = np.full((100, 100), 500, dtype=np.uint16)
            f[:80, :] = 0  # 80% invalid
            frames.append(f)
        decisions = check_depth_sequence(frames, valid_ratio_min=0.50)
        warn = [d for d in decisions if d.severity == Severity.WARN and "valid pixel ratio" in d.message]
        assert len(warn) >= 1

    def test_frozen_depth(self):
        """连续完全相同帧应检测到冻结。"""
        f1 = np.random.randint(1, 1000, (60, 80), dtype=np.uint16)
        f2 = np.random.randint(1, 1000, (60, 80), dtype=np.uint16)
        frames = [f1, f1, f1, f1, f2, f2, f2]  # 4 frames frozen, then 3
        decisions = check_depth_sequence(frames, frozen_consecutive_min=3)
        frozen = [d for d in decisions if "frozen" in d.message]
        assert len(frozen) >= 1

    def test_dtype_info(self):
        """应输出 dtype 和范围信息。"""
        frames = [np.full((100, 100), 500, dtype=np.uint16) for _ in range(3)]
        decisions = check_depth_sequence(frames)
        info = [d for d in decisions if d.reason == ReasonCode.DEPTH_UNIT_UNKNOWN]
        assert len(info) >= 1
        assert "uint16" in info[0].detail["dtype"]


# ---------------------------------------------------------------------------
# RGB-Depth 配对率
# ---------------------------------------------------------------------------


class TestRGBDepthPairing:
    def test_perfect_pairing(self):
        """完全配对。"""
        rgb_ts = [0, 33_333_333, 66_666_667, 100_000_000]  # 30fps
        depth_ts = [0, 33_333_333, 66_666_667, 100_000_000]
        decisions = check_rgb_depth_pairing(rgb_ts, depth_ts)
        info = [d for d in decisions if d.severity == Severity.INFO]
        assert len(info) >= 1
        assert "100.00%" in info[0].message

    def test_poor_pairing(self):
        """配对率低应触发 WARN/ERROR。"""
        rgb_ts = list(range(0, 1_000_000_000, 33_333_333))
        depth_ts = [t + 200_000_000 for t in rgb_ts]  # 200ms offset
        decisions = check_rgb_depth_pairing(rgb_ts, depth_ts, max_offset_ns=50_000_000)
        issues = [d for d in decisions if d.severity in (Severity.WARN, Severity.ERROR)]
        assert len(issues) >= 1

    def test_empty_timestamps(self):
        """空时间戳列表。"""
        decisions = check_rgb_depth_pairing([], [1, 2, 3])
        assert len(decisions) == 1
        assert decisions[0].severity == Severity.WARN


# ---------------------------------------------------------------------------
# check() 统一入口
# ---------------------------------------------------------------------------


class TestCheckEntry:
    def test_with_frames(self):
        """直接传入帧列表。"""
        frames = [np.full((60, 80), 1000, dtype=np.uint16) for _ in range(5)]
        decisions = check(depth_frames=frames)
        assert isinstance(decisions, list)

    def test_with_pairing(self):
        """传入 RGB 和 Depth 时间戳进行配对检查。"""
        rgb_ts = [0, 33_333_333, 66_666_667]
        depth_ts = [0, 33_333_333, 66_666_667]
        decisions = check(rgb_timestamps_ns=rgb_ts, depth_timestamps_ns=depth_ts)
        assert len(decisions) >= 1

    def test_no_input(self):
        """无输入时应返回空列表。"""
        decisions = check()
        assert decisions == []

    def test_empty_depth_frames(self):
        """空帧列表 — 无配对数据时返回空。"""
        decisions = check(depth_frames=[])
        assert isinstance(decisions, list)
        # 空帧 + 无配对时间戳 → 空结果
        # validate it doesn't crash
