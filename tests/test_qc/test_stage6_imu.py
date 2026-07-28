"""Stage 6 IMU 异常 QC 测试（D16）。"""

import numpy as np

from zpds.core.decisions import ReasonCode, Severity
from zpds.qc.stage6_imu import (
    check,
    detect_imu_freeze,
    detect_imu_interval_anomalies,
    detect_imu_saturation,
    detect_imu_spikes,
    estimate_static_bias,
)

# ---------------------------------------------------------------------------
# 间隔异常
# ---------------------------------------------------------------------------


class TestIntervalAnomalies:
    def test_normal_interval(self):
        """正常等间隔采样。"""
        ts = list(range(0, 10_000_000, 5_000_000))  # 200 Hz, 10 samples
        decisions = detect_imu_interval_anomalies(ts, expected_interval_s=0.005)
        gaps = [d for d in decisions if d.severity in (Severity.WARN, Severity.ERROR) and d.reason == ReasonCode.IMU_GAP]
        assert len(gaps) == 0

    def test_large_gap(self):
        """大 gap 应检测到。"""
        ts = [0, 5_000_000, 10_000_000, 500_000_000, 505_000_000]  # 490ms gap
        decisions = detect_imu_interval_anomalies(
            ts, expected_interval_s=0.005, gap_factor=3.0
        )
        gaps = [d for d in decisions if d.severity in (Severity.WARN, Severity.ERROR)]
        assert len(gaps) >= 1

    def test_duplicate_timestamps(self):
        """重复时间戳应检测到。"""
        ts = [0, 0, 5_000_000, 5_000_000, 10_000_000]
        decisions = detect_imu_interval_anomalies(ts)
        dup = [d for d in decisions if "duplicate" in d.message.lower()]
        assert len(dup) >= 1

    def test_empty_input(self):
        """空输入。"""
        decisions = detect_imu_interval_anomalies([])
        assert len(decisions) == 0

    def test_single_sample(self):
        """单样本不应报错。"""
        decisions = detect_imu_interval_anomalies([100])
        assert len(decisions) == 0

    def test_info_level_output(self):
        """应有 INFO 级别的汇总输出。"""
        ts = list(range(0, 50_000_000, 5_000_000))
        decisions = detect_imu_interval_anomalies(ts)
        info = [d for d in decisions if d.severity == Severity.INFO]
        assert len(info) >= 1


# ---------------------------------------------------------------------------
# 尖峰检测
# ---------------------------------------------------------------------------


class TestSpikes:
    def test_no_spikes(self):
        """无明显尖峰。"""
        n = 200
        ts = list(range(0, n * 5_000_000, 5_000_000))
        # 正态分布噪声
        rng = np.random.RandomState(42)
        values = rng.normal(0, 0.1, (n, 6))
        values[:, :3] += 9.81  # accel
        decisions = detect_imu_spikes(ts, values)
        spikes = [d for d in decisions if "spike" in d.message.lower()]
        assert len(spikes) == 0

    def test_with_spikes(self):
        """有明显尖峰。"""
        n = 200
        ts = list(range(0, n * 5_000_000, 5_000_000))
        rng = np.random.RandomState(42)
        values = rng.normal(0, 0.1, (n, 6))
        values[:, :3] += 9.81
        # 插入尖峰
        values[50, 0] = 50.0   # 5g spike
        values[100, 3] = 10.0  # large gyro spike
        decisions = detect_imu_spikes(ts, values)
        spikes = [d for d in decisions if "spike" in d.message.lower()]
        assert len(spikes) >= 1

    def test_nan_values(self):
        """NaN 值不应导致崩溃。"""
        n = 50
        ts = list(range(n))
        values = np.random.randn(n, 6).astype(np.float64)
        values[10, :] = np.nan
        values[20, 0] = np.inf
        decisions = detect_imu_spikes(ts, values)
        assert isinstance(decisions, list)  # should not crash

    def test_empty_values(self):
        decisions = detect_imu_spikes([], np.array([]))
        assert decisions == []


# ---------------------------------------------------------------------------
# 冻结检测
# ---------------------------------------------------------------------------


class TestFreeze:
    def test_no_freeze(self):
        """不断变化的数据。"""
        n = 50
        ts = list(range(n))
        values = np.random.randn(n, 3).astype(np.float64)
        decisions = detect_imu_freeze(ts, values)
        freeze = [d for d in decisions if "frozen" in d.message]
        assert len(freeze) == 0

    def test_frozen_segment(self):
        """有冻结段。"""
        n = 50
        ts = list(range(n))
        values = np.random.randn(n, 3).astype(np.float64)
        # 连续 15 帧相同
        values[10:25, :] = values[10, :]
        decisions = detect_imu_freeze(ts, values, consecutive_min=10)
        freeze = [d for d in decisions if "frozen" in d.message]
        assert len(freeze) >= 1
        assert freeze[0].detail["frozen_sample_count"] >= 15

    def test_short_input(self):
        """少于最少触发数的输入。"""
        ts = [0, 1, 2]
        values = np.ones((3, 3))
        decisions = detect_imu_freeze(ts, values, consecutive_min=10)
        assert decisions == []


# ---------------------------------------------------------------------------
# 静止零偏
# ---------------------------------------------------------------------------


class TestStaticBias:
    def test_stationary_data(self):
        """静止数据应能估计零偏。"""
        n = 200
        ts = list(range(0, n * 5_000_000, 5_000_000))
        rng = np.random.RandomState(42)
        values = np.zeros((n, 6))
        # 静止：加速度 ≈ [0, 0, 9.81]，陀螺 ≈ [0, 0, 0]
        values[:, 2] = 9.81
        values += rng.normal(0, 0.01, (n, 6))
        decisions = estimate_static_bias(ts, values)
        assert len(decisions) >= 1
        assert "static bias" in decisions[0].message.lower()

    def test_moving_data(self):
        """运动数据应有少量或零静止样本。"""
        n = 100
        ts = list(range(n))
        rng = np.random.RandomState(42)
        values = rng.normal(5, 5, (n, 6))
        decisions = estimate_static_bias(ts, values)
        assert isinstance(decisions, list)  # should not crash

    def test_insufficient_axes(self):
        """少于 6 轴不应做零偏估计。"""
        ts = [0, 1, 2]
        values = np.random.randn(3, 3)
        decisions = estimate_static_bias(ts, values)
        assert decisions == []


# ---------------------------------------------------------------------------
# 饱和检查
# ---------------------------------------------------------------------------


class TestSaturation:
    def test_no_saturation(self):
        """正常范围数据。"""
        n = 100
        values = np.random.randn(n, 6).astype(np.float64) * 5
        decisions = detect_imu_saturation(
            values, accel_range_mps2=156.96, gyro_range_rps=34.9
        )
        info = [d for d in decisions if d.severity == Severity.INFO]
        assert len(info) >= 1
        assert "no saturation" in info[0].message

    def test_accel_saturation(self):
        """加速度饱和。"""
        n = 50
        values = np.ones((n, 6)) * 156.96  # at range limit
        decisions = detect_imu_saturation(
            values, accel_range_mps2=156.96, gyro_range_rps=34.9
        )
        sat = [d for d in decisions if d.reason == ReasonCode.IMU_SATURATION and d.severity == Severity.WARN]
        assert len(sat) >= 1

    def test_gyro_saturation(self):
        """陀螺饱和。"""
        n = 50
        values = np.ones((n, 6)) * 34.9
        decisions = detect_imu_saturation(
            values, accel_range_mps2=156.96, gyro_range_rps=34.9
        )
        sat = [d for d in decisions if d.reason == ReasonCode.IMU_SATURATION and d.severity == Severity.WARN]
        assert len(sat) >= 1

    def test_no_range_provided(self):
        """无明确量程时不判定饱和。"""
        values = np.ones((10, 6)) * 200
        decisions = detect_imu_saturation(values)
        assert decisions == []


# ---------------------------------------------------------------------------
# check() 统一入口
# ---------------------------------------------------------------------------


class TestCheckEntry:
    def test_no_data(self):
        decisions = check()
        assert len(decisions) == 1
        assert decisions[0].severity == Severity.WARN

    def test_normal_data(self):
        """正常 IMU 数据。"""
        n = 100
        ts = list(range(0, n * 5_000_000, 5_000_000))
        rng = np.random.RandomState(42)
        values = rng.normal(0, 0.1, (n, 6))
        values[:, :3] += 9.81
        decisions = check(timestamps_ns=ts, values=values)
        assert isinstance(decisions, list)
        # 正常数据不应有 FATAL/ERROR
        severe = [d for d in decisions if d.severity in (Severity.FATAL, Severity.ERROR)]
        assert len(severe) == 0

    def test_with_saturation_range(self):
        """提供量程进行饱和检查。"""
        n = 50
        ts = list(range(n))
        values = np.random.randn(n, 6).astype(np.float64) * 5
        decisions = check(
            timestamps_ns=ts,
            values=values,
            accel_range_mps2=156.96,
            gyro_range_rps=34.9,
        )
        sat_info = [d for d in decisions if d.reason == ReasonCode.IMU_SATURATION]
        assert len(sat_info) >= 1

    def test_disabled_checks(self):
        """禁用子检测。"""
        ts = list(range(10))
        values = np.ones((10, 6)) * 200
        decisions = check(
            timestamps_ns=ts,
            values=values,
            stage_config={
                "spike": {"enabled": False},
                "freeze": {"enabled": False},
                "static_bias": {"enabled": False},
            },
        )
        assert isinstance(decisions, list)
