"""B4 遁甲多相机覆盖与末端可见性单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zpds_prepare.detectors.dunjia.coverage import (
    CameraCoverageRecord,
    DunjiaCoverageReport,
    EndEffectorVisibility,
    _assess_end_effector,
    _compute_overlap,
    check_dunjia_coverage,
)


# ---- Mock types ----

class _MockVideoStream:
    def __init__(
        self, stream_id: str, frame_count: int = 100,
        timestamps_ns: list | None = None,
        width: int = 1600, height: int = 1300,
        video_path: str = "/fake/video.mp4",
    ):
        self.stream_id = stream_id
        self.frame_count = frame_count
        self.timestamps_ns = timestamps_ns or list(range(0, frame_count * 40_000_000, 40_000_000))
        self.width = width
        self.height = height
        self.video_path = video_path
        self.index_frames = []
        self.fps = 25.0


class _MockDepthStream:
    def __init__(self, timestamps_ns: list | None = None):
        self.stream_id = "ego_depth"
        self.timestamps_ns = timestamps_ns or list(range(0, 100 * 40_000_000, 40_000_000))
        self.frame_count = len(self.timestamps_ns)


class _MockIMUStream:
    def __init__(self, timestamps_ns: list | None = None):
        self.stream_id = "robot0_imu"
        ts = timestamps_ns or list(range(0, 1000 * 5_000_000, 5_000_000))
        n = len(ts)
        self.dataframe = pd.DataFrame({
            "timestamp_ns": ts,
            "ax": np.zeros(n),
            "ay": np.zeros(n),
            "az": np.full(n, 9.81),
            "gx": np.zeros(n),
            "gy": np.zeros(n),
            "gz": np.zeros(n),
        })
        self.sample_rate_hz = 196.0


class _MockSession:
    def __init__(
        self,
        video_streams: dict | None = None,
        depth_stream=None,
        imu_stream=None,
    ):
        self.session_id = "dunjia_test"
        self.source_path = "/fake/session.mcap"
        self.video_streams = video_streams or {}
        self.depth_streams = {}
        if depth_stream is not None:
            self.depth_streams["ego_depth"] = depth_stream
        self.imu_streams = {}
        if imu_stream is not None:
            self.imu_streams["robot0_imu"] = imu_stream


# ===================================================================
# 重叠计算测试
# ===================================================================


class TestComputeOverlap:
    def test_full_overlap(self):
        result = _compute_overlap((0, 10_000_000_000), (0, 10_000_000_000))
        assert result == 10.0

    def test_partial(self):
        result = _compute_overlap((0, 10_000_000_000), (5_000_000_000, 15_000_000_000))
        assert result == 5.0

    def test_no_overlap(self):
        result = _compute_overlap((0, 5_000_000_000), (10_000_000_000, 15_000_000_000))
        assert result == 0.0

    def test_none_input(self):
        assert _compute_overlap(None, (0, 100)) == 0.0
        assert _compute_overlap((0, 100), None) == 0.0


# ===================================================================
# 末端可见性测试
# ===================================================================


class TestAssessEndEffector:
    def test_no_primary(self):
        report = DunjiaCoverageReport(session_id="t", source_path="/t")
        _assess_end_effector(report)
        assert report.end_effector_visibility.status == "not_visible"

    def test_primary_zero_frames(self):
        report = DunjiaCoverageReport(session_id="t", source_path="/t")
        report.cameras["camera0"] = CameraCoverageRecord(
            camera_id="camera0", role="primary",
            role_source="test", frame_id="cam0",
            frame_count=0,
            start_timestamp_ns=None, end_timestamp_ns=None,
            duration_s=0.0, decode_status="unavailable",
        )
        _assess_end_effector(report)
        assert report.end_effector_visibility.status == "not_visible"

    def test_primary_available_unassessed(self):
        """主视角可用但无几何 → marked unassessed, not confirmed visible."""
        report = DunjiaCoverageReport(session_id="t", source_path="/t")
        report.cameras["camera0"] = CameraCoverageRecord(
            camera_id="camera0", role="primary",
            role_source="test", frame_id="cam0",
            frame_count=100,
            start_timestamp_ns=0, end_timestamp_ns=4_000_000_000,
            duration_s=4.0, decode_status="decodable",
        )
        _assess_end_effector(report)
        assert report.end_effector_visibility.status == "unassessed"
        assert report.end_effector_visibility.assessment_method == "camera_coverage_only"

    def test_primary_undecodable(self):
        report = DunjiaCoverageReport(session_id="t", source_path="/t")
        report.cameras["camera0"] = CameraCoverageRecord(
            camera_id="camera0", role="primary",
            role_source="test", frame_id="cam0",
            frame_count=100,
            start_timestamp_ns=0, end_timestamp_ns=4_000_000_000,
            duration_s=4.0, decode_status="undecodable",
        )
        _assess_end_effector(report)
        assert report.end_effector_visibility.status == "not_visible"

    def test_side_cameras_help_occlusion(self):
        """侧视角可用应在 notes 中体现遮挡互补。"""
        report = DunjiaCoverageReport(session_id="t", source_path="/t")
        report.cameras["camera0"] = CameraCoverageRecord(
            camera_id="camera0", role="primary",
            role_source="test", frame_id="cam0",
            frame_count=100, start_timestamp_ns=0, end_timestamp_ns=4e9,
            duration_s=4.0, decode_status="decodable",
        )
        report.cameras["camera1"] = CameraCoverageRecord(
            camera_id="camera1", role="side",
            role_source="test", frame_id="cam1",
            frame_count=90, start_timestamp_ns=0, end_timestamp_ns=3.6e9,
            duration_s=3.6, decode_status="decodable",
        )
        _assess_end_effector(report)
        assert "侧视角可用" in report.end_effector_visibility.notes


# ===================================================================
# 集成测试
# ===================================================================


class TestCheckDunjiaCoverage:
    def test_full_coverage(self):
        """三路相机 + 深度 + IMU 全有，时间范围一致。"""
        # 所有流使用同一时间范围避免重叠率 < 50%
        n_frames = 351
        cam_ts = list(range(0, n_frames * 40_000_000, 40_000_000))
        cam_max = cam_ts[-1]
        # IMU 时间戳覆盖整个相机时长（196Hz ≈ 5.1ms）
        imu_n = int(cam_max / 5_000_000) + 1
        imu_ts = list(range(0, imu_n * 5_000_000, 5_000_000))
        session = _MockSession(
            video_streams={
                "camera0": _MockVideoStream("camera0", n_frames, timestamps_ns=cam_ts),
                "camera1": _MockVideoStream("camera1", 357, timestamps_ns=cam_ts),
                "camera2": _MockVideoStream("camera2", 357, timestamps_ns=cam_ts),
            },
            depth_stream=_MockDepthStream(timestamps_ns=cam_ts),
            imu_stream=_MockIMUStream(timestamps_ns=imu_ts),
        )
        report = check_dunjia_coverage(session)

        assert report.overall_disposition == "pass"
        assert "camera0" in report.cameras
        assert "camera1" in report.cameras
        assert "camera2" in report.cameras

        c0 = report.cameras["camera0"]
        assert c0.role == "primary"
        assert c0.role_source == "dunjia_reader.CAMERA_IDS"
        assert c0.frame_count == 351

        # 深度/IMU 重叠
        assert c0.depth_overlap_s > 0
        assert c0.depth_overlap_ratio > 0
        assert c0.imu_overlap_s > 0

        # 侧相机角色
        assert report.cameras["camera1"].role == "side"
        assert report.cameras["camera2"].role == "side"

    def test_missing_camera1_not_fatal(self):
        """camera1 缺失不阻断。"""
        session = _MockSession(
            video_streams={
                "camera0": _MockVideoStream("camera0", 351),
                # camera1 缺失
                "camera2": _MockVideoStream("camera2", 357),
            },
            depth_stream=_MockDepthStream(),
            imu_stream=_MockIMUStream(),
        )
        report = check_dunjia_coverage(session)

        assert report.overall_disposition != "reject"
        assert report.cameras["camera1"].frame_count == 0
        assert any("camera1" in i for i in report.issues)

    def test_missing_primary_reject(self):
        """主视角缺失 → reject。"""
        session = _MockSession(
            video_streams={},  # 无 camera0
            depth_stream=_MockDepthStream(),
            imu_stream=_MockIMUStream(),
        )
        report = check_dunjia_coverage(session)
        assert report.overall_disposition == "reject"

    def test_no_depth_still_pass(self):
        """深度流缺失不影响相机覆盖判定。"""
        session = _MockSession(
            video_streams={
                "camera0": _MockVideoStream("camera0", 351),
                "camera1": _MockVideoStream("camera1", 357),
                "camera2": _MockVideoStream("camera2", 357),
            },
            depth_stream=None,  # 无深度
            imu_stream=_MockIMUStream(),
        )
        report = check_dunjia_coverage(session)

        c0 = report.cameras["camera0"]
        assert c0.depth_overlap_s == 0.0
        assert c0.depth_overlap_ratio == 0.0
        # 深度重叠不足应标记
        assert report.overall_disposition == "keep_with_flag"

    def test_camera_roles_from_metadata(self):
        """相机角色来自 CAMERA_IDS，不来自目录名。"""
        session = _MockSession(
            video_streams={
                "camera0": _MockVideoStream("camera0", 100),
            },
        )
        report = check_dunjia_coverage(session)

        for cam_name, cam in report.cameras.items():
            if cam.frame_count > 0:
                assert cam.role_source == "dunjia_reader.CAMERA_IDS", (
                    f"{cam_name} role_source 不是 CAMERA_IDS"
                )

    def test_end_effector_recorded(self):
        """末端可见性记录应存在。"""
        session = _MockSession(
            video_streams={
                "camera0": _MockVideoStream("camera0", 351),
            },
        )
        report = check_dunjia_coverage(session)

        ee = report.end_effector_visibility
        assert ee is not None
        assert ee.assessment_method in {
            "camera_coverage_only", "geometric_projection",
            "vlm_review", "manual_review", "unavailable",
        }


# ===================================================================
# 数据类测试
# ===================================================================


class TestDataClasses:
    def test_camera_coverage(self):
        c = CameraCoverageRecord(
            camera_id="camera0", role="primary",
            role_source="CAMERA_IDS", frame_id="cam0",
            frame_count=351,
            start_timestamp_ns=0, end_timestamp_ns=14_000_000_000,
            duration_s=14.0, decode_status="decodable",
            depth_overlap_s=12.0, depth_overlap_ratio=0.85,
            imu_overlap_s=14.0, imu_overlap_ratio=1.0,
        )
        assert c.role == "primary"
        assert c.depth_overlap_ratio == 0.85

    def test_end_effector_visibility(self):
        ee = EndEffectorVisibility(
            status="unassessed",
            assessment_method="camera_coverage_only",
            notes="需 VLM 复核",
        )
        assert ee.assessment_method != "geometric_projection"
        assert ee.visible_ratio is None
