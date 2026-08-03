"""B7 A2D 相机-机器人对齐单元测试。"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zpds_prepare.detectors.a2d.alignment import (
    A2DAlignmentReport,
    CameraRobotAlignmentRow,
    StreamAlignmentSummary,
    _check_monotonic,
    _median_interval,
    check_a2d_alignment,
    write_alignment_parquet,
)


# ---- Mock 类型 ----


@dataclass
class _MockIndexFrame:
    frame_index: int
    source_path: str = "/fake/camera/0/head_color.jpg"
    source_timestamp_ns: int | None = None
    timestamp_method: str = "aligned_joints_index"
    timestamp_error_ns: int | None = 0


@dataclass
class _MockVideoStream:
    stream_id: str
    frame_count: int
    index_frames: list[dict]
    timestamps_ns: list[int] = field(default_factory=list)
    width: int = 640
    height: int = 480
    fps: float = 30.0

    def __post_init__(self):
        if not self.timestamps_ns:
            self.timestamps_ns = [
                f.get("source_timestamp_ns", 0) or 0
                for f in self.index_frames
            ]


class _MockTimeSeriesStream:
    def __init__(self, timestamps_ns: list[int]):
        self.stream_id = "robot_state"
        self.timestamps_ns = timestamps_ns
        self.modality = "joint_state"
        self.role = "state"


class _MockSession:
    def __init__(
        self,
        video_streams: dict | None = None,
        time_series_streams: dict | None = None,
        source_path: str = "/fake/episode/8032",
    ):
        self.session_id = "a2d_8032"
        self.source_path = source_path
        self.video_streams = video_streams or {}
        self.depth_streams = {}
        self.imu_streams = {}
        self.annotation_streams = {}
        self.time_series_streams = time_series_streams or {}


# ---- 工具函数 ----

def _make_index_frames(n: int) -> list[dict]:
    return [
        {
            "frame_index": i,
            "source_path": f"/fake/camera/{i}/head_color.jpg",
            "source_timestamp_ns": i * 33_333_333,
            "timestamp_method": "aligned_joints_index",
            "timestamp_error_ns": 0,
        }
        for i in range(n)
    ]


def _make_robot_ts(n: int) -> list[int]:
    return list(range(0, n * 50_000_000, 50_000_000))


# ===================================================================
# 辅助函数测试
# ===================================================================


class TestCheckMonotonic:
    def test_monotonic(self):
        assert _check_monotonic(np.array([1, 2, 3, 4]))

    def test_non_monotonic(self):
        assert not _check_monotonic(np.array([1, 3, 2, 4]))

    def test_single_element(self):
        assert _check_monotonic(np.array([1]))

    def test_empty(self):
        assert _check_monotonic(np.array([]))


class TestMedianInterval:
    def test_uniform(self):
        result = _median_interval(np.array([0, 10, 20, 30]))
        assert result == 10.0

    def test_two_elements(self):
        result = _median_interval(np.array([0, 100]))
        assert result == 100.0


# ===================================================================
# 对齐逻辑测试
# ===================================================================


class TestCheckA2DAlignment:
    def test_no_robot_state(self):
        session = _MockSession(time_series_streams={})
        report = check_a2d_alignment(session)
        assert report.overall_disposition == "reject"
        assert report.robot_bc_ready is False

    def test_no_rgb_streams(self):
        session = _MockSession(
            time_series_streams={
                "robot_state": _MockTimeSeriesStream(_make_robot_ts(100)),
            },
        )
        report = check_a2d_alignment(session)
        assert report.overall_disposition == "reject"

    def test_perfect_alignment(self):
        """相机帧索引与 HDF5 行号一一对应 → 全覆盖。"""
        n = 166
        robot_ts = _make_robot_ts(200)  # HDF5 有 200 行, 相机只有 166 帧

        session = _MockSession(
            video_streams={
                "head_rgb": _MockVideoStream(
                    stream_id="head_rgb",
                    frame_count=n,
                    index_frames=_make_index_frames(n),
                ),
                "hand_left_rgb": _MockVideoStream(
                    stream_id="hand_left_rgb",
                    frame_count=n,
                    index_frames=_make_index_frames(n),
                ),
            },
            time_series_streams={
                "robot_state": _MockTimeSeriesStream(robot_ts),
            },
        )
        report = check_a2d_alignment(session)

        assert report.overall_disposition == "pass"
        assert report.robot_bc_ready is True
        assert len(report.alignment_rows) == n * 2  # 2 cameras × 166 frames

        # head 流
        head = report.streams["head_rgb"]
        assert head.total_camera_frames == n
        assert head.mapped_frames == n
        assert head.unmapped_frames == 0
        assert head.continuity_groups == 1

        # 映射方法验证
        first_row = report.alignment_rows[0]
        assert first_row.mapping_method == "aligned_joints_index"
        assert first_row.camera_stream_id == "head_rgb"
        assert first_row.source_frame_index == 0
        assert first_row.robot_row == 0  # 不是 "第 N 个相机帧" 的假设

    def test_partial_mapping(self):
        """相机帧超出 HDF5 行范围 → 部分 unmapped。"""
        n_h5 = 50
        n_cam = 100  # 相机帧比 HDF5 多

        session = _MockSession(
            video_streams={
                "head_rgb": _MockVideoStream(
                    stream_id="head_rgb",
                    frame_count=n_cam,
                    index_frames=_make_index_frames(n_cam),
                ),
            },
            time_series_streams={
                "robot_state": _MockTimeSeriesStream(_make_robot_ts(n_h5)),
            },
        )
        report = check_a2d_alignment(session)

        head = report.streams["head_rgb"]
        assert head.mapped_frames == n_h5
        assert head.unmapped_frames == n_cam - n_h5
        assert report.robot_bc_ready is False
        assert report.overall_disposition == "keep_with_flag"

    def test_non_monotonic_robot_ts(self):
        """HDF5 时间戳非递增 → flagged。"""
        bad_ts = [0, 100, 50, 200]
        session = _MockSession(
            video_streams={
                "head_rgb": _MockVideoStream(
                    stream_id="head_rgb",
                    frame_count=4,
                    index_frames=[
                        {"frame_index": i} for i in range(4)
                    ],
                ),
            },
            time_series_streams={
                "robot_state": _MockTimeSeriesStream(bad_ts),
            },
        )
        report = check_a2d_alignment(session)
        assert not report.hdf5_timestamp_valid
        assert any("非单调" in i for i in report.issues)

    def test_mapping_not_by_row_number(self):
        """对齐必须基于显式映射而非纯行号假设。"""
        # 相机帧索引与 HDF5 行号不同
        # 模拟: 相机有帧 0,2,4（缺 1,3），HDF5 有 5 行
        robot_ts = _make_robot_ts(5)
        sparse_indices = [
            {"frame_index": 0},
            {"frame_index": 2},
            {"frame_index": 4},
        ]

        session = _MockSession(
            video_streams={
                "hand_left_rgb": _MockVideoStream(
                    stream_id="hand_left_rgb",
                    frame_count=3,
                    index_frames=sparse_indices,
                ),
            },
            time_series_streams={
                "robot_state": _MockTimeSeriesStream(robot_ts),
            },
        )
        report = check_a2d_alignment(session)

        rows = report.alignment_rows
        assert len(rows) == 3

        # 帧 0 → HDF5 行 0
        assert rows[0].source_frame_index == 0
        assert rows[0].robot_row == 0

        # 帧 2 → HDF5 行 2（不是第 2 个相机帧 = HDF5 行 1）
        assert rows[1].source_frame_index == 2
        assert rows[1].robot_row == 2

        # 连续性: 帧 0,2,4 各自孤立 → 3 个 continuity groups
        hand = report.streams["hand_left_rgb"]
        assert hand.continuity_groups == 3

    def test_alignment_rows_have_required_fields(self):
        """对齐行包含所有必需字段。"""
        n = 10
        session = _MockSession(
            video_streams={
                "head_rgb": _MockVideoStream(
                    stream_id="head_rgb",
                    frame_count=n,
                    index_frames=_make_index_frames(n),
                ),
            },
            time_series_streams={
                "robot_state": _MockTimeSeriesStream(_make_robot_ts(20)),
            },
        )
        report = check_a2d_alignment(session)

        row = report.alignment_rows[0]
        required = {
            "camera_stream_id",
            "source_frame_index",
            "camera_timestamp_ns",
            "robot_row",
            "robot_timestamp_ns",
            "mapping_method",
            "error_ns",
            "uncertainty_ns",
            "continuity_group",
            "evidence_uri",
        }
        row_dict = {
            "camera_stream_id": row.camera_stream_id,
            "source_frame_index": row.source_frame_index,
            "camera_timestamp_ns": row.camera_timestamp_ns,
            "robot_row": row.robot_row,
            "robot_timestamp_ns": row.robot_timestamp_ns,
            "mapping_method": row.mapping_method,
            "error_ns": row.error_ns,
            "uncertainty_ns": row.uncertainty_ns,
            "continuity_group": row.continuity_group,
            "evidence_uri": row.evidence_uri,
        }
        assert set(row_dict.keys()) == required

    def test_all_three_cameras(self):
        """三路相机都有对齐行。"""
        n = 50
        session = _MockSession(
            video_streams={
                "head_rgb": _MockVideoStream("head_rgb", n, _make_index_frames(n)),
                "hand_left_rgb": _MockVideoStream("hand_left_rgb", n, _make_index_frames(n)),
                "hand_right_rgb": _MockVideoStream("hand_right_rgb", n, _make_index_frames(n)),
            },
            time_series_streams={
                "robot_state": _MockTimeSeriesStream(_make_robot_ts(100)),
            },
        )
        report = check_a2d_alignment(session)

        stream_ids = {r.camera_stream_id for r in report.alignment_rows}
        assert stream_ids == {"head_rgb", "hand_left_rgb", "hand_right_rgb"}


# ===================================================================
# Parquet 写出测试
# ===================================================================


class TestWriteAlignmentParquet:
    def test_write_and_read(self):
        rows = [
            CameraRobotAlignmentRow(
                camera_stream_id="head_rgb",
                source_frame_index=i,
                camera_timestamp_ns=i * 33_333_333,
                robot_row=i,
                robot_timestamp_ns=i * 50_000_000,
                mapping_method="aligned_joints_index",
                error_ns=0,
                uncertainty_ns=16_666_667,
                continuity_group=0,
            )
            for i in range(10)
        ]
        report = A2DAlignmentReport(
            episode_id="8032",
            source_path="/test",
            alignment_rows=rows,
        )

        with tempfile.TemporaryDirectory() as td:
            path = write_alignment_parquet(report, Path(td) / "camera_robot_alignment.parquet")
            assert path.is_file()

            import pandas as pd
            df = pd.read_parquet(path)
            assert len(df) == 10
            assert list(df.columns) == [
                "camera_stream_id",
                "source_frame_index",
                "camera_timestamp_ns",
                "robot_row",
                "robot_timestamp_ns",
                "mapping_method",
                "error_ns",
                "uncertainty_ns",
                "continuity_group",
                "evidence_uri",
            ]

    def test_empty_rows_raises(self):
        report = A2DAlignmentReport(
            episode_id="8032",
            source_path="/test",
        )
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError, match="空"):
                write_alignment_parquet(report, Path(td) / "test.parquet")


# ===================================================================
# 数据类测试
# ===================================================================


class TestDataClasses:
    def test_alignment_row(self):
        row = CameraRobotAlignmentRow(
            camera_stream_id="head_rgb",
            source_frame_index=0,
            camera_timestamp_ns=0,
            robot_row=0,
            robot_timestamp_ns=0,
            mapping_method="aligned_joints_index",
            error_ns=0,
            uncertainty_ns=16_666_667,
            continuity_group=0,
        )
        assert row.mapping_method == "aligned_joints_index"

    def test_stream_summary(self):
        s = StreamAlignmentSummary(
            stream_id="head_rgb",
            total_camera_frames=166,
            mapped_frames=166,
            unmapped_frames=0,
            continuity_groups=1,
            method_distribution={"aligned_joints_index": 166},
        )
        assert s.mapped_frames == 166
