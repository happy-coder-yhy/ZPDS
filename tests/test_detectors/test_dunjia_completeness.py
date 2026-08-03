"""B1 遁甲完整性检测单元测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from zpds_prepare.detectors.dunjia.completeness import (
    CameraRole,
    DunjiaCompletenessReport,
    StreamCompleteness,
    _check_calibration,
    _check_depth_stream,
    _check_imu_stream,
    _check_video_stream,
    _stream_type_for_id,
    check_dunjia_completeness,
)


# ---- 轻量 Mock Session ----

@dataclass
class _MockVideoStream:
    stream_id: str
    video_path: str
    frame_count: int
    width: int
    height: int


@dataclass
class _MockDepthStream:
    stream_id: str
    frame_count: int
    width: int
    height: int
    dtype: str
    unit: str = "unknown"


class _MockSession:
    def __init__(
        self,
        session_id: str = "dunjia_test",
        source_path: str = "/fake/session.mcap",
        video_streams: dict | None = None,
        depth_streams: dict | None = None,
        imu_streams: dict | None = None,
    ):
        self.session_id = session_id
        self.source_path = source_path
        self.video_streams = video_streams or {}
        self.depth_streams = depth_streams or {}
        self.imu_streams = imu_streams or {}
        self.annotation_streams = {}
        self.time_series_streams = {}


# ---- 辅助 ----

def _make_imu_df(samples: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = np.arange(samples, dtype=np.int64) * 5_000_000  # 5ms interval
    return pd.DataFrame({
        "timestamp_ns": ts,
        "ax": rng.normal(0, 0.1, samples),
        "ay": rng.normal(0, 0.1, samples),
        "az": rng.normal(9.81, 0.1, samples),
        "gx": rng.normal(0, 0.01, samples),
        "gy": rng.normal(0, 0.01, samples),
        "gz": rng.normal(0, 0.01, samples),
    })


class MockImuStream:
    def __init__(self, dataframe: pd.DataFrame):
        self.stream_id = "robot0_imu"
        self.dataframe = dataframe
        self.sample_rate_hz = 196.0


# ===================================================================
# 单元测试: 辅助函数
# ===================================================================


class TestStreamTypeForId:
    def test_camera_video(self):
        assert _stream_type_for_id("camera0") == "video"
        assert _stream_type_for_id("camera1") == "video"

    def test_depth(self):
        assert _stream_type_for_id("depth") == "depth"

    def test_imu(self):
        assert _stream_type_for_id("robot0_imu") == "imu"

    def test_calibration(self):
        assert _stream_type_for_id("calibration") == "calibration"

    def test_unknown(self):
        assert _stream_type_for_id("something_else") == "unknown"


class TestCheckVideoStream:
    def test_missing_stream(self):
        result = _check_video_stream("camera0", None, required=True, source_path="")
        assert not result.present
        assert result.disposition == "reject"
        assert len(result.issues) > 0

    def test_missing_optional_stream(self):
        result = _check_video_stream("camera1", None, required=False, source_path="")
        assert not result.present
        assert result.disposition == "keep_with_flag"

    def test_zero_frames(self):
        vs = _MockVideoStream("camera0", "/tmp/v.mp4", 0, 1600, 1300)
        result = _check_video_stream("camera0", vs, required=True, source_path="")
        assert result.present
        assert result.disposition == "reject"

    def test_missing_video_path(self):
        vs = _MockVideoStream("camera0", "", 100, 1600, 1300)
        result = _check_video_stream("camera0", vs, required=True, source_path="")
        assert result.decodable is False
        assert result.disposition == "reject"

    @patch("cv2.VideoCapture")
    def test_video_opens(self, mock_cap):
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_cap.return_value = mock_instance

        vs = _MockVideoStream("camera0", "/tmp/v.mp4", 100, 1600, 1300)
        with patch("pathlib.Path.is_file", return_value=True):
            result = _check_video_stream("camera0", vs, required=True, source_path="")
        assert result.decodable is True
        assert result.disposition == "pass"

    @patch("cv2.VideoCapture")
    def test_video_cannot_open(self, mock_cap):
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap.return_value = mock_instance

        vs = _MockVideoStream("camera0", "/tmp/v.mp4", 100, 1600, 1300)
        with patch("pathlib.Path.is_file", return_value=True):
            result = _check_video_stream("camera0", vs, required=True, source_path="")
        assert result.decodable is False
        assert result.disposition == "reject"


class TestCheckDepthStream:
    def test_missing_stream(self):
        result = _check_depth_stream(None, required=True, source_path="")
        assert not result.present
        assert result.disposition == "reject"

    def test_missing_optional(self):
        result = _check_depth_stream(None, required=False, source_path="")
        assert not result.present
        assert result.disposition == "keep_with_flag"

    def test_zero_frames(self):
        ds = _MockDepthStream("ego_depth", 0, 1920, 1080, "uint16")
        result = _check_depth_stream(ds, required=True, source_path="")
        assert result.disposition == "reject"
        assert len(result.issues) > 0

    def test_valid(self):
        ds = _MockDepthStream("ego_depth", 312, 1920, 1080, "uint16")
        result = _check_depth_stream(ds, required=True, source_path="")
        # No source file → skips decode validation, but stream itself valid
        assert result.present
        assert result.frame_count == 312
        assert result.width == 1920
        assert result.height == 1080


class TestCheckImuStream:
    def test_missing_stream(self):
        result = _check_imu_stream(None, source_path="")
        assert not result.present
        assert result.disposition == "reject"

    def test_zero_samples(self):
        imu = MockImuStream(pd.DataFrame(columns=["timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"]))
        result = _check_imu_stream(imu, source_path="")
        assert result.present
        assert result.disposition == "reject"

    def test_valid(self):
        imu = MockImuStream(_make_imu_df(100))
        result = _check_imu_stream(imu, source_path="")
        assert result.present
        assert result.sample_count == 100
        assert result.disposition == "pass"

    def test_missing_columns(self):
        df = pd.DataFrame({"timestamp_ns": [1, 2, 3]})
        imu = MockImuStream(df)
        result = _check_imu_stream(imu, source_path="")
        assert result.present
        assert result.disposition == "keep_with_flag"
        assert len(result.issues) > 0


class TestCheckCalibration:
    @patch("zpds_prepare.detectors.dunjia.completeness._count_topic_messages")
    def test_all_present(self, mock_count):
        mock_count.return_value = 1
        result = _check_calibration("/fake/session.mcap", mcap_ok=True)
        assert result.present is True
        assert result.disposition == "pass"
        assert len(result.issues) == 0

    @patch("zpds_prepare.detectors.dunjia.completeness._count_topic_messages")
    def test_some_missing(self, mock_count):
        def side_effect(_path, topic):
            if "camera2" in topic:
                return 0
            return 1

        mock_count.side_effect = side_effect
        result = _check_calibration("/fake/session.mcap", mcap_ok=True)
        assert result.present is True
        assert result.disposition == "keep_with_flag"
        assert "camera2" in str(result.issues)

    def test_no_source_path(self):
        result = _check_calibration("", mcap_ok=False)
        assert result.present is False
        assert result.disposition == "keep_with_flag"


# ===================================================================
# 集成测试: check_dunjia_completeness
# ===================================================================


class TestCheckDunjiaCompleteness:
    """测试完整完整性检查流程。"""

    @patch("zpds_prepare.detectors.dunjia.completeness._sha256_file")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_mcap")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_depth_first_frame")
    @patch("zpds_prepare.detectors.dunjia.completeness._count_topic_messages")
    @patch("cv2.VideoCapture")
    def test_full_session_pass(
        self, mock_cap, mock_count, mock_depth, mock_mcap, mock_sha256,
    ):
        """完整 Session，所有流正常。"""
        mock_mcap.return_value = (True, "")
        mock_sha256.return_value = "abc123"

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_cap.return_value = mock_instance

        mock_count.return_value = 1  # all calibs present
        mock_depth.return_value = (True, 1920, 1080, "uint16", "")

        session = _MockSession(
            session_id="dunjia_test",
            source_path="/fake/session.mcap",
            video_streams={
                "camera0": _MockVideoStream("camera0", "/fake/cam0.mp4", 351, 1600, 1300),
                "camera1": _MockVideoStream("camera1", "/fake/cam1.mp4", 357, 352, 288),
                "camera2": _MockVideoStream("camera2", "/fake/cam2.mp4", 357, 352, 288),
            },
            depth_streams={
                "ego_depth": _MockDepthStream("ego_depth", 312, 1920, 1080, "uint16"),
            },
            imu_streams={
                "robot0_imu": MockImuStream(_make_imu_df(2899)),
            },
        )

        with patch("pathlib.Path.is_file", return_value=True):
            report = check_dunjia_completeness(session)

        assert report.session_id == "dunjia_test"
        assert report.overall_disposition == "pass"
        assert report.all_required_present is True
        assert report.required_present == report.required_total

        # 检查 camera0
        c0 = report.streams["camera0"]
        assert c0.present
        assert c0.frame_count == 351
        assert c0.disposition == "pass"

        # 检查深度
        depth = report.streams["depth"]
        assert depth.present
        assert depth.frame_count == 312
        assert depth.dtype == "uint16"

        # 检查 IMU
        imu = report.streams["robot0_imu"]
        assert imu.present
        assert imu.sample_count == 2899

        # 相机角色
        assert len(report.camera_roles) == 4  # camera0/1/2 + depth
        assert report.camera_roles["camera0"].role == "primary"
        assert report.camera_roles["camera1"].role == "side"
        assert report.camera_roles["camera2"].role == "side"
        assert report.camera_roles["depth"].role == "depth"

        # robot_bc_ready 声明
        assert report.robot_bc_ready == "not_applicable"
        assert "无机器人 state/action" in report.robot_bc_ready_reason

    @patch("zpds_prepare.detectors.dunjia.completeness._sha256_file")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_mcap")
    def test_mcap_corrupt(self, mock_mcap, mock_sha256):
        """MCAP 损坏 → 所有流 reject。"""
        mock_mcap.return_value = (False, "文件损坏")
        mock_sha256.return_value = ""

        session = _MockSession(source_path="/fake/bad.mcap")
        report = check_dunjia_completeness(session)

        assert report.overall_disposition == "reject"
        for stream in report.streams.values():
            assert stream.disposition == "reject"

    @patch("zpds_prepare.detectors.dunjia.completeness._sha256_file")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_mcap")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_depth_first_frame")
    @patch("zpds_prepare.detectors.dunjia.completeness._count_topic_messages")
    @patch("cv2.VideoCapture")
    def test_missing_camera1_not_fatal(
        self, mock_cap, mock_count, mock_depth, mock_mcap, mock_sha256,
    ):
        """camera1 缺失 → 仅标记，不阻断整体。"""
        mock_mcap.return_value = (True, "")
        mock_sha256.return_value = "abc"
        mock_count.return_value = 1
        mock_depth.return_value = (True, 1920, 1080, "uint16", "")

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_cap.return_value = mock_instance

        session = _MockSession(
            source_path="/fake/session.mcap",
            video_streams={
                "camera0": _MockVideoStream("camera0", "/fake/cam0.mp4", 351, 1600, 1300),
                # camera1 缺失
                "camera2": _MockVideoStream("camera2", "/fake/cam2.mp4", 357, 352, 288),
            },
            depth_streams={
                "ego_depth": _MockDepthStream("ego_depth", 312, 1920, 1080, "uint16"),
            },
            imu_streams={
                "robot0_imu": MockImuStream(_make_imu_df(100)),
            },
        )

        with patch("pathlib.Path.is_file", return_value=True):
            report = check_dunjia_completeness(session)

        assert report.overall_disposition == "keep_with_flag"
        assert report.streams["camera0"].disposition == "pass"
        assert report.streams["camera1"].disposition == "keep_with_flag"
        assert report.streams["camera2"].disposition == "pass"

    @patch("zpds_prepare.detectors.dunjia.completeness._sha256_file")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_mcap")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_depth_first_frame")
    @patch("zpds_prepare.detectors.dunjia.completeness._count_topic_messages")
    @patch("cv2.VideoCapture")
    def test_missing_camera0_fatal(
        self, mock_cap, mock_count, mock_depth, mock_mcap, mock_sha256,
    ):
        """camera0 缺失 → 必须 reject。"""
        mock_mcap.return_value = (True, "")
        mock_sha256.return_value = "abc"
        mock_count.return_value = 1
        mock_depth.return_value = (True, 1920, 1080, "uint16", "")

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_cap.return_value = mock_instance

        session = _MockSession(
            source_path="/fake/session.mcap",
            video_streams={},  # 没有 camera0
            depth_streams={
                "ego_depth": _MockDepthStream("ego_depth", 312, 1920, 1080, "uint16"),
            },
            imu_streams={
                "robot0_imu": MockImuStream(_make_imu_df(100)),
            },
        )

        with patch("pathlib.Path.is_file", return_value=True):
            report = check_dunjia_completeness(session)

        assert report.streams["camera0"].disposition == "reject"
        assert report.overall_disposition == "reject"

    @patch("zpds_prepare.detectors.dunjia.completeness._sha256_file")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_mcap")
    @patch("zpds_prepare.detectors.dunjia.completeness._validate_depth_first_frame")
    @patch("zpds_prepare.detectors.dunjia.completeness._count_topic_messages")
    @patch("cv2.VideoCapture")
    def test_camera_role_source_is_metadata(
        self, mock_cap, mock_count, mock_depth, mock_mcap, mock_sha256,
    ):
        """相机角色必须来自源元数据，不能是目录名或画面内容。"""
        mock_mcap.return_value = (True, "")
        mock_sha256.return_value = "abc"
        mock_count.return_value = 1
        mock_depth.return_value = (True, 1920, 1080, "uint16", "")

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_cap.return_value = mock_instance

        session = _MockSession(
            source_path="/fake/session.mcap",
            video_streams={
                "camera0": _MockVideoStream("camera0", "/fake/cam0.mp4", 100, 1600, 1300),
            },
            imu_streams={
                "robot0_imu": MockImuStream(_make_imu_df(100)),
            },
        )

        with patch("pathlib.Path.is_file", return_value=True):
            report = check_dunjia_completeness(session)

        for cam_name, role in report.camera_roles.items():
            assert role.role_source == "dunjia_reader.CAMERA_IDS", (
                f"{cam_name} 角色来源错误: {role.role_source}，"
                f"必须来自源元数据，不能是目录名或画面推测"
            )


# ===================================================================
# 数据类基础测试
# ===================================================================


class TestDataClasses:
    def test_stream_completeness_defaults(self):
        sc = StreamCompleteness(stream_id="test", stream_type="video", required=True, present=False)
        assert sc.frame_count == 0
        assert sc.disposition == "pass"
        assert sc.issues == []

    def test_camera_role(self):
        cr = CameraRole(
            camera_id="camera0",
            role="primary",
            role_source="dunjia_reader.CAMERA_IDS",
            frame_id="headcam_center_optical_frame",
        )
        assert cr.role == "primary"
        assert "CAMERA_IDS" in cr.role_source

    def test_report_property(self):
        report = DunjiaCompletenessReport(
            session_id="test",
            source_path="/test",
            source_sha256="abc",
            required_total=3,
            required_present=3,
        )
        assert report.all_required_present is True

        report2 = DunjiaCompletenessReport(
            session_id="test2",
            source_path="/test2",
            source_sha256="def",
            required_total=3,
            required_present=2,
        )
        assert report2.all_required_present is False
