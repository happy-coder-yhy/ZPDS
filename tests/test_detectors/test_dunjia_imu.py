"""B3 遁甲 IMU 质量检测单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zpds_prepare.detectors.dunjia.imu_quality import (
    DunjiaIMUReport,
    IMUGapSpan,
    IMUSpikeEvent,
    IMUStaticWindow,
    _check_saturation,
    _check_timestamps,
    _detect_freezes,
    _detect_spikes,
    _estimate_static_bias,
    check_dunjia_imu,
)


# ---- Mock ----

class _MockIMUStream:
    def __init__(self, df: pd.DataFrame, sample_rate_hz: float = 196.0):
        self.stream_id = "robot0_imu"
        self.dataframe = df
        self.sample_rate_hz = sample_rate_hz


class _MockSession:
    def __init__(self, imu_stream=None):
        self.session_id = "dunjia_test"
        self.source_path = "/fake/session.mcap"
        self.imu_streams = {}
        if imu_stream is not None:
            self.imu_streams["robot0_imu"] = imu_stream
        self.video_streams = {}
        self.depth_streams = {}


# ---- 工具 ----

def _make_normal_imu(samples: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = np.arange(samples, dtype=np.int64) * 5_100_000  # ~196Hz
    return pd.DataFrame({
        "timestamp_ns": ts,
        "ax": rng.normal(0, 0.1, samples),
        "ay": rng.normal(0, 0.1, samples),
        "az": rng.normal(9.81, 0.1, samples),
        "gx": rng.normal(0, 0.01, samples),
        "gy": rng.normal(0, 0.01, samples),
        "gz": rng.normal(0, 0.01, samples),
    })


# ===================================================================
# 时间戳检查
# ===================================================================


class TestCheckTimestamps:
    def test_perfect(self):
        ts = np.arange(1000, dtype=np.int64) * 5_000_000
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _check_timestamps(ts, report, 3.0)

        assert report.timestamp_valid
        assert not report.has_duplicates
        assert report.gap_count == 0

    def test_duplicates(self):
        ts = np.array([0, 5_000_000, 5_000_000, 10_000_000], dtype=np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _check_timestamps(ts, report, 3.0)

        assert report.has_duplicates
        assert report.duplicate_count == 1

    def test_regression(self):
        ts = np.array([0, 10_000_000, 5_000_000, 15_000_000], dtype=np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _check_timestamps(ts, report, 3.0)

        assert report.has_regression
        assert not report.timestamp_valid

    def test_gap(self):
        ts = np.array([0, 5_000_000, 50_000_000, 55_000_000], dtype=np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _check_timestamps(ts, report, 3.0)

        assert report.gap_count == 1
        assert report.gaps[0].gap_ns == 45_000_000

    def test_no_gap_if_below_threshold(self):
        ts = np.arange(100, dtype=np.int64) * 5_000_000
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _check_timestamps(ts, report, 3.0)

        assert report.gap_count == 0  # 均匀间隔，无 gap


# ===================================================================
# 尖峰检测
# ===================================================================


class TestDetectSpikes:
    def test_no_spikes_normal(self):
        df = _make_normal_imu(500)
        ts = df["timestamp_ns"].values.astype(np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _detect_spikes(df, ts, report, 6.0)

        # 正常数据无尖峰
        assert report.spike_count == 0

    def test_spike_detected(self):
        df = _make_normal_imu(500)
        # 注入一个明显尖峰
        df.loc[100, "ax"] = 100.0  # 远大于正常范围
        df.loc[100, "gz"] = 50.0
        ts = df["timestamp_ns"].values.astype(np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _detect_spikes(df, ts, report, 6.0)

        assert report.spike_count >= 2  # ax + gz


# ===================================================================
# 冻结检测
# ===================================================================


class TestDetectFreezes:
    def test_no_freeze(self):
        df = _make_normal_imu(100)
        ts = df["timestamp_ns"].values.astype(np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _detect_freezes(df, ts, report, 5)

        assert report.freeze_span_count == 0

    def test_freeze_detected(self):
        df = _make_normal_imu(100)
        # 冻结 ax 列最后 10 行
        df.loc[90:, "ax"] = 1.0
        df.loc[90:, "ay"] = 2.0
        df.loc[90:, "az"] = 3.0
        ts = df["timestamp_ns"].values.astype(np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _detect_freezes(df, ts, report, 5)

        assert report.freeze_span_count >= 1


# ===================================================================
# 静止零偏
# ===================================================================


class TestStaticBias:
    def test_no_static(self):
        df = _make_normal_imu(100)
        df["az"] = df["az"] + 5.0  # 偏离重力
        ts = df["timestamp_ns"].values.astype(np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t", median_interval_ns=5_000_000)
        _estimate_static_bias(df, ts, report, 0.5)

        assert report.static_window_count == 0

    def test_static_detected(self):
        df = _make_normal_imu(100)
        # 前 50 样本设为静止
        df.loc[:50, "ax"] = 0.0
        df.loc[:50, "ay"] = 0.0
        df.loc[:50, "az"] = 9.81
        df.loc[:50, "gx"] = 0.0
        df.loc[:50, "gy"] = 0.0
        df.loc[:50, "gz"] = 0.0
        ts = df["timestamp_ns"].values.astype(np.int64)
        report = DunjiaIMUReport(session_id="t", source_path="/t", median_interval_ns=5_000_000)
        _estimate_static_bias(df, ts, report, 0.2)

        assert report.static_window_count >= 1
        w = report.static_windows[0]
        assert abs(w.accel_bias[2] - 9.81) < 0.1
        assert abs(w.gyro_bias[0]) < 0.01


# ===================================================================
# 饱和检查
# ===================================================================


class TestCheckSaturation:
    def test_unavailable_by_default(self):
        df = _make_normal_imu(100)
        report = DunjiaIMUReport(session_id="t", source_path="/t")
        _check_saturation(df, report)
        assert report.saturation_status == "unavailable"
        assert report.saturation_accel_count == 0

    def test_saturation_with_range(self):
        df = _make_normal_imu(100)
        # 注入饱和样本
        df.loc[0, "ax"] = 160.0  # 超量程
        report = DunjiaIMUReport(
            session_id="t", source_path="/t",
            accel_range_mps2=156.96,
            gyro_range_rps=34.9,
        )
        _check_saturation(df, report)
        assert report.saturation_status == "checked"
        assert report.saturation_accel_count > 0


# ===================================================================
# 集成测试
# ===================================================================


class TestCheckDunjiaIMU:
    def test_no_imu_stream(self):
        session = _MockSession()
        result = check_dunjia_imu(session)
        assert result.overall_disposition == "reject"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"])
        session = _MockSession(_MockIMUStream(df))
        result = check_dunjia_imu(session)
        assert result.overall_disposition == "reject"

    def test_missing_columns(self):
        df = pd.DataFrame({"timestamp_ns": [1, 2]})
        session = _MockSession(_MockIMUStream(df))
        result = check_dunjia_imu(session)
        assert result.overall_disposition == "reject"

    def test_normal_imu_pass(self):
        df = _make_normal_imu(500)
        session = _MockSession(_MockIMUStream(df))
        result = check_dunjia_imu(session)

        assert result.sample_count == 500
        assert result.timestamp_valid
        assert result.gap_count == 0
        assert result.overall_disposition == "pass"

    def test_saturation_unavailable_when_no_range(self):
        """量程未知时饱和状态为 unavailable，不判定饱和。"""
        df = _make_normal_imu(500)
        session = _MockSession(_MockIMUStream(df))
        result = check_dunjia_imu(session)

        assert result.saturation_status == "unavailable"

    def test_with_gaps(self):
        """含 gap 的正常数据 → keep_with_flag。"""
        df = _make_normal_imu(500)
        # 在第 100 行插入一个 gap
        ts = df["timestamp_ns"].values.copy()
        ts[100:] += 500_000_000  # 0.5s gap
        df["timestamp_ns"] = ts
        session = _MockSession(_MockIMUStream(df))
        result = check_dunjia_imu(session)

        assert result.gap_count > 0  # gap 被检测到


# ===================================================================
# 数据类
# ===================================================================


class TestDataClasses:
    def test_gap_span(self):
        g = IMUGapSpan(
            start_sample=10, end_sample=11,
            start_timestamp_ns=50_000_000, end_timestamp_ns=150_000_000,
            gap_ns=100_000_000, gap_s=0.1,
            expected_interval_ns=5_000_000, factor=20.0,
        )
        assert g.gap_s == 0.1
        assert g.factor == 20.0

    def test_spike_event(self):
        s = IMUSpikeEvent(
            sample_index=42, timestamp_ns=210_000_000,
            field="ax", value=50.0,
            median=0.0, mad=0.1, deviation_factor=500.0,
        )
        assert s.deviation_factor == 500.0

    def test_static_window(self):
        w = IMUStaticWindow(
            start_sample=0, end_sample=100,
            duration_s=0.5,
            accel_bias=(0.01, 0.02, 9.80),
            gyro_bias=(0.001, -0.001, 0.0),
            accel_mag_mean=9.81,
            gyro_mag_mean=0.002,
        )
        assert abs(w.accel_bias[2] - 9.80) < 0.01
