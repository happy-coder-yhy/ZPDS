"""B8 A2D state/action/夹爪/安全质量单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from zpds_prepare.detectors.a2d.robot_quality import (
    GripperResponse,
    StateActionLag,
    TimeSeriesQuality,
    _check_gripper_response,
    _check_timeseries,
    _estimate_state_action_lag,
    check_a2d_robot_quality,
)


# ---- Mock types ----

class _MockTSStream:
    def __init__(
        self,
        stream_id: str,
        timestamps_ns: list[int],
        rows: np.ndarray,
        metadata: dict | None = None,
    ):
        self.stream_id = stream_id
        self.timestamps_ns = timestamps_ns
        self.rows = rows
        self.metadata = metadata or {}


class _MockSession:
    def __init__(self, time_series_streams: dict | None = None):
        self.session_id = "a2d_8032"
        self.source_path = "/fake/episode"
        self.video_streams = {}
        self.depth_streams = {}
        self.imu_streams = {}
        self.annotation_streams = {}
        self.time_series_streams = time_series_streams or {}


def _make_ts_stream(
    stream_id: str, samples: int = 100, fields: int = 18,
) -> _MockTSStream:
    ts = list(range(0, samples * 50_000_000, 50_000_000))
    rng = np.random.default_rng(42)
    rows = rng.normal(0, 0.1, (samples, fields))
    return _MockTSStream(stream_id, ts, rows)


# ===================================================================
# 时序流质量
# ===================================================================


class TestCheckTimeseries:
    def test_normal(self):
        stream = _make_ts_stream("robot_state", 100, 18)
        q = _check_timeseries(stream, freeze_min_duration_s=2.0, gap_factor=3.0)

        assert q.sample_count == 100
        assert q.field_count == 18
        assert q.nan_count == 0
        assert q.inf_count == 0
        assert q.finite_ratio == 1.0
        assert q.timestamp_valid

    def test_with_nan(self):
        ts = list(range(0, 100 * 50_000_000, 50_000_000))
        rows = np.zeros((100, 1))
        rows[10, 0] = np.nan
        rows[20, 0] = np.nan
        stream = _MockTSStream("robot_state", ts, rows)
        q = _check_timeseries(stream, 2.0, 3.0)

        assert q.nan_count == 2
        assert q.finite_ratio < 1.0

    def test_with_gap(self):
        ts = list(range(0, 50 * 50_000_000, 50_000_000))
        ts[30:] = [t + 500_000_000 for t in ts[30:]]  # 0.5s gap
        rows = np.zeros((50, 1))
        stream = _MockTSStream("test", ts, rows)
        q = _check_timeseries(stream, 2.0, 3.0)

        assert q.gap_count == 1

    def test_timestamp_regression(self):
        ts = [0, 100, 50, 200]  # 回退
        rows = np.zeros((4, 1))
        stream = _MockTSStream("test", ts, rows)
        q = _check_timeseries(stream, 2.0, 3.0)

        assert not q.timestamp_valid
        assert q.has_regression


# ===================================================================
# state-action lag
# ===================================================================


class TestEstimateLag:
    def test_no_streams(self):
        session = _MockSession()
        lag = _estimate_state_action_lag(session, 50)
        assert not lag.estimated

    def test_with_data(self):
        """有数据的互相关估计。"""
        samples = 200
        ts = list(range(0, samples * 50_000_000, 50_000_000))
        state_rows = np.sin(np.linspace(0, 4 * np.pi, samples)).reshape(-1, 1)
        action_rows = np.roll(state_rows, 3)  # action 领先 state 3 步

        session = _MockSession({
            "robot_state": _MockTSStream("robot_state", ts, state_rows),
            "robot_action": _MockTSStream("robot_action", ts, action_rows),
        })
        lag = _estimate_state_action_lag(session, 50)

        assert lag.estimated
        assert lag.method == "cross_correlation"


# ===================================================================
# 夹爪响应
# ===================================================================


class TestCheckGripperResponse:
    def test_no_streams(self):
        session = _MockSession()
        resp = _check_gripper_response(session)
        assert "无 gripper_state" in resp.notes

    def test_no_movement(self):
        """无命令无响应 → no_op。"""
        ts = list(range(0, 100 * 50_000_000, 50_000_000))
        state_rows = np.ones((100, 1))  # 无变化
        action_rows = np.ones((100, 1))  # 无命令

        session = _MockSession({
            "gripper_state": _MockTSStream("gripper_state", ts, state_rows),
            "gripper_action": _MockTSStream("gripper_action", ts, action_rows),
        })
        resp = _check_gripper_response(session)

        assert resp.command_count == 0
        assert resp.stall_count == 0
        assert resp.no_op_count > 0

    def test_with_movement(self):
        """有命令有响应。"""
        samples = 100
        ts = list(range(0, samples * 50_000_000, 50_000_000))
        state_rows = np.linspace(0, 1, samples).reshape(-1, 1)
        action_rows = np.linspace(0, 1, samples).reshape(-1, 1)

        session = _MockSession({
            "gripper_state": _MockTSStream("gripper_state", ts, state_rows),
            "gripper_action": _MockTSStream("gripper_action", ts, action_rows),
        })
        resp = _check_gripper_response(session)

        assert resp.command_count > 0
        assert resp.response_count > 0


# ===================================================================
# 集成测试
# ===================================================================


class TestCheckA2DRobotQuality:
    def test_no_robot_state(self):
        session = _MockSession()
        report = check_a2d_robot_quality(session)

        assert report.overall_disposition == "reject"
        assert report.robot_bc_ready is False

    def test_full_quality(self):
        """完整的 state+action+gripper → pass。"""
        samples = 200
        ts = list(range(0, samples * 50_000_000, 50_000_000))
        session = _MockSession({
            "robot_state": _MockTSStream("robot_state", ts, np.random.randn(samples, 18)),
            "robot_action": _MockTSStream("robot_action", ts, np.random.randn(samples, 18)),
            "gripper_state": _MockTSStream("gripper_state", ts, np.linspace(0, 1, samples).reshape(-1, 1)),
            "gripper_action": _MockTSStream("gripper_action", ts, np.ones((samples, 1))),
        })

        report = check_a2d_robot_quality(session)

        assert report.robot_state_quality is not None
        assert report.robot_action_quality is not None
        assert report.state_action_lag.estimated
        assert report.gripper_response is not None

    def test_action_nan_over_50_percent(self):
        """action NaN > 50% → flagged。"""
        samples = 100
        ts = list(range(0, samples * 50_000_000, 50_000_000))
        state_rows = np.random.randn(samples, 18)
        action_rows = np.full((samples, 18), np.nan)

        session = _MockSession({
            "robot_state": _MockTSStream("robot_state", ts, state_rows),
            "robot_action": _MockTSStream("robot_action", ts, action_rows),
        })
        report = check_a2d_robot_quality(session)

        assert any("NaN" in i for i in report.issues)
        assert report.robot_bc_ready is False


# ===================================================================
# 数据类
# ===================================================================


class TestDataClasses:
    def test_time_series_quality(self):
        q = TimeSeriesQuality(
            stream_id="robot_state",
            sample_count=100,
            field_count=18,
            joint_count=18,
            nan_count=0, inf_count=0, finite_ratio=1.0,
        )
        assert q.sample_count == 100

    def test_state_action_lag(self):
        lag = StateActionLag(
            estimated=True, method="cross_correlation",
            lag_ns_p50=15_000_000, lag_samples=3,
        )
        assert lag.lag_samples == 3

    def test_gripper_response(self):
        g = GripperResponse(
            command_count=50, response_count=48, stall_count=2,
            no_op_count=50,
        )
        assert g.stall_count == 2
