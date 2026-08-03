"""B2 遁甲 RGB-D 质量检测单元测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zpds_prepare.detectors.dunjia.rgbd_quality import (
    DepthFrameSample,
    DunjiaRGBDReport,
    RGBDepthAlignment,
    _find_unpaired_spans,
    _is_likely_invalid,
    _pair_rgb_depth,
    check_dunjia_rgbd,
)


# ---- Mock Session ----

@dataclass
class _MockDepthStream:
    stream_id: str = "ego_depth"
    frame_count: int = 312
    width: int = 1920
    height: int = 1080
    dtype: str = "uint16"
    unit: str = "unknown"
    timestamps_ns: list[int] = None
    frame_id: str = "depth_optical_frame"

    def __post_init__(self):
        if self.timestamps_ns is None:
            self.timestamps_ns = list(range(0, self.frame_count * 40_000_000, 40_000_000))


@dataclass
class _MockVideoStream:
    stream_id: str = "camera0"
    frame_count: int = 351
    width: int = 1600
    height: int = 1300
    timestamps_ns: list[int] = None

    def __post_init__(self):
        if self.timestamps_ns is None:
            self.timestamps_ns = list(range(0, self.frame_count * 40_000_000, 40_000_000))


class _MockSession:
    def __init__(
        self,
        session_id: str = "dunjia_test",
        source_path: str = "/fake/session.mcap",
        depth_stream: _MockDepthStream | None = None,
        rgb_stream: _MockVideoStream | None = None,
    ):
        self.session_id = session_id
        self.source_path = source_path
        self.depth_streams = {}
        if depth_stream is not None:
            self.depth_streams["ego_depth"] = depth_stream
        self.video_streams = {}
        if rgb_stream is not None:
            self.video_streams["camera0"] = rgb_stream
        self.imu_streams = {}


# ===================================================================
# 纯函数测试
# ===================================================================


class TestIsLikelyInvalid:
    def test_zero(self):
        assert _is_likely_invalid(0, 65535)

    def test_65535_uint16(self):
        assert _is_likely_invalid(65535, 65535)

    def test_valid_value(self):
        assert not _is_likely_invalid(1000, 65535)

    def test_65504(self):
        assert _is_likely_invalid(65504, 65535)


class TestFindUnpairedSpans:
    def test_all_paired(self):
        spans = _find_unpaired_spans(100, set(range(100)))
        assert spans == []

    def test_none_paired(self):
        spans = _find_unpaired_spans(5, set())
        assert spans == [(0, 4)]

    def test_mixed(self):
        # paired: indices 2, 3, 4
        paired = {2, 3, 4}
        spans = _find_unpaired_spans(10, paired)
        assert spans == [(0, 1), (5, 9)]

    def test_edge_start(self):
        paired = {5, 6, 7}
        spans = _find_unpaired_spans(10, paired)
        assert spans == [(0, 4), (8, 9)]

    def test_edge_end(self):
        paired = {0, 1, 2}
        spans = _find_unpaired_spans(10, paired)
        assert spans == [(3, 9)]


class TestPairRGBDepth:
    def test_perfect_alignment(self):
        rgb_ts = np.array([0, 40_000_000, 80_000_000], dtype=np.int64)
        depth_ts = np.array([0, 40_000_000, 80_000_000], dtype=np.int64)
        result = _pair_rgb_depth(rgb_ts, depth_ts, max_offset_ns=50_000_000)

        assert result.paired_count == 3
        assert result.paired_ratio == 1.0
        assert result.offset_ns_max == 0

    def test_with_offset(self):
        rgb_ts = np.array([0, 40_000_000, 80_000_000], dtype=np.int64)
        # 深度帧偏移 1ms
        depth_ts = np.array([1_000_000, 41_000_000, 81_000_000], dtype=np.int64)
        result = _pair_rgb_depth(rgb_ts, depth_ts, max_offset_ns=50_000_000)

        assert result.paired_count == 3
        assert abs(result.offset_ns_p50) == pytest.approx(1_000_000, abs=100)

    def test_beyond_threshold_not_paired(self):
        rgb_ts = np.array([0, 100_000_000], dtype=np.int64)
        depth_ts = np.array([200_000_000], dtype=np.int64)
        result = _pair_rgb_depth(rgb_ts, depth_ts, max_offset_ns=50_000_000)

        assert result.paired_count == 0
        assert result.paired_ratio == 0.0
        assert len(result.unpaired_rgb_spans) > 0

    def test_extra_rgb_frames(self):
        """RGB 帧中间帧通过最近邻配对到同一深度帧。"""
        rgb_ts = np.array([0, 40_000_000, 80_000_000], dtype=np.int64)
        depth_ts = np.array([0, 80_000_000], dtype=np.int64)
        result = _pair_rgb_depth(rgb_ts, depth_ts, max_offset_ns=50_000_000)

        # 三个 RGB 帧都在 50ms 内找到最近邻深度帧
        # rgb[0]→depth[0](0), rgb[1]→depth[0](40ms), rgb[2]→depth[1](0)
        assert result.paired_count == 3
        assert result.paired_ratio == 1.0  # capped

    def test_empty_inputs(self):
        result = _pair_rgb_depth(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            50_000_000,
        )
        assert result.paired_count == 0
        assert math.isnan(result.offset_ns_p50)

    def test_mapping_method_is_timestamp_based(self):
        """配对方法必须基于时间戳，不是帧号。"""
        # RGB 和 depth 帧号不对应但时间戳接近 → 应该成功配对
        rgb_ts = np.array([0, 20_000_000], dtype=np.int64)
        depth_ts = np.array([100_000_000, 21_000_000], dtype=np.int64)
        # depth[0] 时间戳 100_000_000 远超阈值
        # depth[1] 时间戳 21_000_000 接近 rgb[1]=20_000_000
        result = _pair_rgb_depth(rgb_ts, depth_ts, max_offset_ns=5_000_000)

        assert result.mapping_method == "nearest_neighbor_timestamp"
        # 帧号 1→1 会被配对，帧号 0→0 不会（时间戳差太大）
        assert result.paired_count >= 1


import math


# ===================================================================
# 集成测试（mock MCAP）
# ===================================================================


class TestCheckDunjiaRGBD:
    def test_no_depth_stream(self):
        session = _MockSession(depth_stream=None)
        result = check_dunjia_rgbd(session)
        assert result.overall_disposition == "reject"
        assert "不存在" in str(result.issues)

    def test_no_rgb_stream(self):
        session = _MockSession(
            depth_stream=_MockDepthStream(),
            rgb_stream=None,
        )
        result = check_dunjia_rgbd(session)
        assert result.overall_disposition == "reject"

    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._sample_depth_frames")
    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._compute_depth_quality")
    def test_basic_report_structure(
        self, mock_compute, mock_sample,
    ):
        """即使跳过全量扫描，报告结构应完整。"""
        session = _MockSession(
            depth_stream=_MockDepthStream(),
            rgb_stream=_MockVideoStream(),
        )
        result = check_dunjia_rgbd(session)

        assert result.session_id == "dunjia_test"
        assert result.depth_frame_count == 312
        assert result.rgb_frame_count == 351
        assert result.depth_unit == "unknown"
        assert result.alignment is not None
        assert result.alignment.rgb_frame_count == 351
        assert result.alignment.depth_frame_count == 312
        assert result.alignment.mapping_method == "nearest_neighbor_timestamp"

    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._sample_depth_frames")
    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._compute_depth_quality")
    def test_depth_unit_unknown_flag(
        self, mock_compute, mock_sample,
    ):
        session = _MockSession(
            depth_stream=_MockDepthStream(unit="unknown"),
            rgb_stream=_MockVideoStream(),
        )
        result = check_dunjia_rgbd(session)
        assert any("单位未知" in i for i in result.issues)

    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._sample_depth_frames")
    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._compute_depth_quality")
    def test_perfect_pairing(
        self, mock_compute, mock_sample,
    ):
        """时间戳完美对齐 → paired_ratio = 1.0。"""
        n = 100
        rgb_ts = np.arange(n, dtype=np.int64) * 40_000_000
        depth_ts = np.arange(n, dtype=np.int64) * 40_000_000

        session = _MockSession(
            depth_stream=_MockDepthStream(
                frame_count=n, timestamps_ns=list(depth_ts),
            ),
            rgb_stream=_MockVideoStream(
                frame_count=n, timestamps_ns=list(rgb_ts),
            ),
        )
        result = check_dunjia_rgbd(session)
        assert result.alignment.paired_ratio == 1.0
        assert result.alignment.offset_ns_max == 0

    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._sample_depth_frames")
    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._compute_depth_quality")
    def test_calibration_issues_detected(
        self, mock_compute, mock_sample,
    ):
        """分辨率不匹配应被检出。"""
        session = _MockSession(
            depth_stream=_MockDepthStream(width=1920, height=1080),
            rgb_stream=_MockVideoStream(width=1600, height=1300),
        )
        result = check_dunjia_rgbd(session)
        assert len(result.calibration_issues) > 0
        assert not result.calibration_consistent

    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._sample_depth_frames")
    @patch("zpds_prepare.detectors.dunjia.rgbd_quality._compute_depth_quality")
    def test_zero_frames_reject(
        self, mock_compute, mock_sample,
    ):
        session = _MockSession(
            depth_stream=_MockDepthStream(frame_count=0),
            rgb_stream=_MockVideoStream(),
        )
        result = check_dunjia_rgbd(session)
        assert result.depth_frame_count == 0
        assert result.overall_disposition == "reject"


# ===================================================================
# 数据类测试
# ===================================================================


class TestDataClasses:
    def test_depth_frame_sample(self):
        s = DepthFrameSample(
            frame_index=0,
            timestamp_ns=1000,
            width=1920,
            height=1080,
            dtype="uint16",
            min_val=0,
            max_val=5000,
            zero_ratio=0.1,
            invalid_ratio=0.01,
            mean_val=2000.0,
        )
        assert s.zero_ratio == 0.1
        assert not s.is_frozen

    def test_rgb_depth_alignment(self):
        a = RGBDepthAlignment(
            paired_count=300,
            rgb_frame_count=351,
            depth_frame_count=312,
            paired_ratio=0.96,
            offset_ns_p50=5000,
            offset_ns_p95=15000,
            offset_ns_max=45000,
        )
        assert a.paired_ratio > 0.9
        assert a.mapping_method == "nearest_neighbor_timestamp"

    def test_report_defaults(self):
        r = DunjiaRGBDReport(session_id="test", source_path="/test")
        assert r.depth_frame_count == 0
        assert r.alignment is None
        assert r.overall_disposition == "pass"
