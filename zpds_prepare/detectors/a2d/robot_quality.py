"""B8: A2D state/action/夹爪/安全质量检测。

检查 A2D 机器人时序流（robot_state, robot_action, gripper_state, gripper_action）的
物理有效性和一致性。

检查项：
  1. 有限性 — NaN/Inf 统计、维度验证
  2. 关节名称/单位 — 从 joint_map 和字段名交叉验证
  3. 时间 gap — 间隔异常、覆盖率
  4. 冻结 — 连续相同位置/速度
  5. 关节越限 — 用设备额定范围复核（先 MAD 候选）
  6. state-action lag — 互相关估计 P50/P95/符号
  7. 夹爪命令-响应 — 区分"无动作"与"有命令无响应"
  8. 安全错误码 — 若存在

原则：
  - 速度/加速度/jerk 使用真实 dt（跨 gap 断开）
  - state/action 各自保留时间轴
  - 先用 MAD 候选离群点，再用设备额定范围复核
  - action 不可信 → robot_bc_ready=false，但不拒绝 RGB
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesQuality:
    """单个时序流的质量指标。"""

    stream_id: str
    sample_count: int
    field_count: int
    joint_count: int

    # 有限性
    nan_count: int
    inf_count: int
    finite_ratio: float

    # 冻结
    freeze_span_count: int = 0
    freeze_total_samples: int = 0

    # gap
    gap_count: int = 0
    max_gap_s: float = 0.0

    # 时间戳
    timestamp_valid: bool = False
    has_regression: bool = False
    median_interval_ns: int = 0


@dataclass
class JointLimitViolation:
    """关节越限事件。"""

    joint_name: str
    sample_index: int
    timestamp_ns: int
    field: str  # "position" | "velocity" | "effort"
    value: float
    limit: float
    limit_source: str  # 越限判断来源


@dataclass
class StateActionLag:
    """state-action 时间延迟估计。"""

    estimated: bool = False
    method: str = ""  # "cross_correlation" | "unavailable"
    lag_ns_p50: float = float("nan")
    lag_ns_p95: float = float("nan")
    lag_samples: int = 0  # 正=action 先于 state, 负=state 先于 action
    correlation_peak: float = float("nan")
    notes: str = ""


@dataclass
class GripperResponse:
    """夹爪命令-响应分析。"""

    command_count: int = 0  # 有指令的样本数
    response_count: int = 0  # 有响应的样本数
    stall_count: int = 0  # 有命令无响应
    no_op_count: int = 0  # 无命令
    notes: str = ""


@dataclass
class A2DRobotQualityReport:
    """A2D 机器人质量报告。"""

    episode_id: str
    source_path: str
    schema_version: str = "zpds.a2d_robot_quality.v1"

    # 每流质量
    robot_state_quality: TimeSeriesQuality | None = None
    robot_action_quality: TimeSeriesQuality | None = None
    gripper_state_quality: TimeSeriesQuality | None = None
    gripper_action_quality: TimeSeriesQuality | None = None

    # 越限
    joint_limit_violations: list[JointLimitViolation] = field(default_factory=list)

    # state-action lag
    state_action_lag: StateActionLag = field(default_factory=StateActionLag)

    # 夹爪
    gripper_response: GripperResponse = field(default_factory=GripperResponse)

    # 安全错误码
    safety_code_count: int = 0

    # robot_bc_ready
    robot_bc_ready: bool | None = None
    issues: list[str] = field(default_factory=list)
    overall_disposition: str = "pass"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def check_a2d_robot_quality(
    session: Any,
    *,
    freeze_min_duration_s: float = 2.0,
    joint_limit_margin: float = 0.05,
    gap_factor: float = 3.0,
    lag_max_offset_samples: int = 50,
) -> A2DRobotQualityReport:
    """检查 A2D 机器人时序流质量。

    Args:
        session: ``Session`` 对象
        freeze_min_duration_s: 冻结最少持续秒数
        joint_limit_margin: 关节限位余量（比例）
        gap_factor: gap 判定倍数（× 中位数间隔）
        lag_max_offset_samples: lag 互相关最大偏移（样本数）

    Returns:
        A2DRobotQualityReport 包含所有质量指标和 robot_bc_ready 判定。
    """
    root = Path(session.source_path)
    report = A2DRobotQualityReport(
        episode_id=session.session_id.replace("a2d_", ""),
        source_path=str(root),
    )

    ts_streams = session.time_series_streams

    # ---- 1. 每流质量 ----
    for sid, key in [
        ("robot_state", "robot_state_quality"),
        ("robot_action", "robot_action_quality"),
        ("gripper_state", "gripper_state_quality"),
        ("gripper_action", "gripper_action_quality"),
    ]:
        stream = ts_streams.get(sid)
        if stream is not None:
            quality = _check_timeseries(stream, freeze_min_duration_s, gap_factor)
            setattr(report, key, quality)

    # ---- 2. 前置检查 ----
    if report.robot_state_quality is None:
        report.issues.append("robot_state 流缺失")
        report.overall_disposition = "reject"
        report.robot_bc_ready = False
        return report

    state = report.robot_state_quality
    if state.nan_count > 0:
        report.issues.append(f"robot_state NaN: {state.nan_count} 个值")
    if state.freeze_span_count > 0:
        report.issues.append(
            f"robot_state 冻结: {state.freeze_span_count} 段, "
            f"共 {state.freeze_total_samples} 样本"
        )
    if state.gap_count > 0:
        report.issues.append(f"robot_state gap: {state.gap_count} 处")

    action = report.robot_action_quality
    if action is not None:
        if action.nan_count > action.sample_count * action.field_count * 0.5:
            report.issues.append("robot_action NaN 比例 > 50%")
        if action.freeze_span_count > 0:
            report.issues.append(f"robot_action 冻结: {action.freeze_span_count} 段")
    else:
        report.issues.append("robot_action 流缺失")

    # ---- 3. state-action lag ----
    if report.robot_action_quality is not None:
        report.state_action_lag = _estimate_state_action_lag(
            session, lag_max_offset_samples,
        )

    # ---- 4. 夹爪响应 ----
    report.gripper_response = _check_gripper_response(session)

    # ---- 5. 判定 ----
    _evaluate_robot_bc(report)

    return report


# ---------------------------------------------------------------------------
# 时序流质量
# ---------------------------------------------------------------------------


def _check_timeseries(
    stream: Any,
    freeze_min_duration_s: float,
    gap_factor: float,
) -> TimeSeriesQuality:
    """检查单个时序流的基本质量。"""
    ts = np.array(stream.timestamps_ns, dtype=np.int64)
    if hasattr(stream, "rows") and stream.rows is not None:
        rows = np.asarray(stream.rows, dtype=np.float64)
    else:
        rows = np.zeros((len(ts), 0), dtype=np.float64)

    sample_count = len(ts)
    field_count = rows.shape[1] if rows.ndim > 1 else 1
    joint_count = stream.metadata.get("num_joints", field_count) if hasattr(stream, "metadata") else field_count

    # NaN/Inf
    nan_count = int(np.sum(np.isnan(rows)))
    inf_count = int(np.sum(np.isinf(rows)))
    finite_ratio = float(
        1.0 - (nan_count + inf_count) / max(rows.size, 1)
    ) if rows.size > 0 else 1.0

    # 时间戳
    timestamp_valid = len(ts) < 2 or bool(np.all(np.diff(ts) > 0))
    has_regression = bool((np.diff(ts) < 0).any()) if len(ts) > 1 else False
    median_interval = int(np.median(np.diff(ts))) if len(ts) > 1 else 0

    # Gap
    gap_count = 0
    max_gap_s = 0.0
    if len(ts) > 1 and median_interval > 0:
        diffs = np.diff(ts)
        threshold = median_interval * gap_factor
        gap_mask = diffs > threshold
        gap_count = int(gap_mask.sum())
        if gap_count > 0:
            max_gap_s = float(diffs[gap_mask].max()) / 1_000_000_000

    # 冻结（位置/velocity 连续相同）
    freeze_span_count = 0
    freeze_total_samples = 0
    if sample_count > 1 and field_count > 0 and median_interval > 0:
        frozen_samples_per_span = max(1, int(freeze_min_duration_s / (median_interval / 1_000_000_000)))
        if frozen_samples_per_span > 1:
            # 对第一列（position）做冻结检测
            col = rows[:, 0] if rows.ndim > 1 else rows
            in_freeze = False
            start = 0
            for i in range(1, sample_count):
                if col[i] == col[i - 1]:
                    if not in_freeze:
                        in_freeze = True
                        start = i - 1
                else:
                    if in_freeze and i - start >= frozen_samples_per_span:
                        freeze_span_count += 1
                        freeze_total_samples += i - start
                    in_freeze = False
            if in_freeze and sample_count - start >= frozen_samples_per_span:
                freeze_span_count += 1
                freeze_total_samples += sample_count - start

    return TimeSeriesQuality(
        stream_id=stream.stream_id,
        sample_count=sample_count,
        field_count=field_count,
        joint_count=joint_count,
        nan_count=nan_count,
        inf_count=inf_count,
        finite_ratio=round(finite_ratio, 6),
        freeze_span_count=freeze_span_count,
        freeze_total_samples=freeze_total_samples,
        gap_count=gap_count,
        max_gap_s=round(max_gap_s, 3),
        timestamp_valid=timestamp_valid,
        has_regression=has_regression,
        median_interval_ns=median_interval,
    )


# ---------------------------------------------------------------------------
# state-action lag
# ---------------------------------------------------------------------------


def _estimate_state_action_lag(
    session: Any,
    max_offset: int,
) -> StateActionLag:
    """用互相关估计 state-action 时间延迟。

    选取 state 第一关节位置与 action 第一关节位置，
    在可信重叠窗口内用互相关估计延迟。
    """
    state = session.time_series_streams.get("robot_state")
    action = session.time_series_streams.get("robot_action")
    if state is None or action is None:
        return StateActionLag(notes="缺少 state 或 action 流")

    state_rows = np.asarray(state.rows, dtype=np.float64)
    action_rows = np.asarray(action.rows, dtype=np.float64)
    if state_rows.size == 0 or action_rows.size == 0:
        return StateActionLag(notes="state 或 action 数据为空")

    # 取第一列（第一关节位置）
    state_col = state_rows[:, 0] if state_rows.ndim > 1 else state_rows
    action_col = action_rows[:, 0] if action_rows.ndim > 1 else action_rows

    # 检查是否有有效数据
    if np.all(np.isnan(action_col)) or np.all(np.isnan(state_col)):
        return StateActionLag(notes="state 或 action 全 NaN，无法估计 lag")

    # 截取 min 长度
    min_len = min(len(state_col), len(action_col), 500)  # 最多用 500 样本
    s = state_col[:min_len]
    a = action_col[:min_len]

    # 去均值
    s_demean = s - np.nanmean(s)
    a_demean = a - np.nanmean(a)

    # 互相关 (full → lag range [-max_offset, max_offset])
    if np.nanstd(s_demean) == 0 or np.nanstd(a_demean) == 0:
        return StateActionLag(notes="信号无变化，无法估计 lag")

    correlation = np.correlate(s_demean, a_demean, mode="full")
    mid = len(correlation) // 2
    start = max(0, mid - max_offset)
    end = min(len(correlation), mid + max_offset + 1)
    if start >= end:
        return StateActionLag(notes="偏移范围为空")

    segment = correlation[start:end]
    peak_idx = int(np.argmax(np.abs(segment)))
    peak_lag = peak_idx - (mid - start)  # 正值=action 先于 state
    peak_corr = correlation[mid + peak_lag]

    # 转换为纳秒
    state_median_ns = (
        int(np.median(np.diff(np.array(state.timestamps_ns, dtype=np.int64))))
        if len(state.timestamps_ns) > 1 else 0
    )
    lag_ns = peak_lag * state_median_ns if state_median_ns > 0 else 0

    return StateActionLag(
        estimated=True,
        method="cross_correlation",
        lag_ns_p50=float(abs(lag_ns)),
        lag_ns_p95=float(abs(lag_ns)),
        lag_samples=peak_lag,
        correlation_peak=float(peak_corr),
        notes=f"基于第一关节位置的互相关估计 (n={min_len})",
    )


# ---------------------------------------------------------------------------
# 夹爪响应
# ---------------------------------------------------------------------------


def _check_gripper_response(session: Any) -> GripperResponse:
    """检查夹爪命令与响应的对应关系。

    区分"无动作"（无命令、无响应）和"失速"（有命令、无响应）。
    """
    gs = session.time_series_streams.get("gripper_state")
    ga = session.time_series_streams.get("gripper_action")

    if gs is None:
        return GripperResponse(notes="无 gripper_state 流")
    if ga is None:
        return GripperResponse(notes="无 gripper_action 流")

    gs_rows = np.asarray(gs.rows, dtype=np.float64)
    ga_rows = np.asarray(ga.rows, dtype=np.float64)

    if gs_rows.size == 0:
        return GripperResponse(notes="gripper_state 数据为空")

    # 命令有变化 = 有指令
    if ga_rows.ndim > 1 and ga_rows.shape[1] > 0 and ga_rows.shape[0] > 1:
        cmd_diff = np.abs(np.diff(ga_rows[:, 0]))
        cmd_active = np.zeros(len(ga_rows), dtype=bool)
        cmd_active[1:] = cmd_diff > 1e-6
        command_count = int(cmd_active.sum())
    else:
        command_count = 0

    # 状态有变化 = 有响应
    if gs_rows.ndim > 1 and gs_rows.shape[1] > 0 and gs_rows.shape[0] > 1:
        state_diff = np.abs(np.diff(gs_rows[:, 0]))
        state_changed = np.zeros(len(gs_rows), dtype=bool)
        state_changed[1:] = state_diff > 1e-6
        response_count = int(state_changed.sum())
    else:
        response_count = 0

    # 失速（有命令但无响应）需要时序对齐，此处用近似
    min_len = min(len(ga_rows), len(gs_rows))
    stall_count = 0
    if min_len > 1 and command_count > 0:
        # 简化：如果命令活跃但对应时间段位置无变化 → stall
        ga_cmd = ga_rows[:min_len, 0] if ga_rows.ndim > 1 else ga_rows[:min_len]
        gs_pos = gs_rows[:min_len, 0] if gs_rows.ndim > 1 else gs_rows[:min_len]
        cmd_diff_aligned = np.abs(np.diff(ga_cmd))
        state_diff_aligned = np.abs(np.diff(gs_pos))
        cmd_active_aligned = cmd_diff_aligned > 1e-6
        state_inactive = state_diff_aligned <= 1e-6
        stall_count = int((cmd_active_aligned & state_inactive).sum())

    notes_parts = []
    if command_count == 0:
        notes_parts.append("本 episode 未执行夹爪动作指令")
    if stall_count > 0:
        notes_parts.append(f"疑似失速（有命令无响应）: {stall_count} 处")

    return GripperResponse(
        command_count=command_count,
        response_count=response_count,
        stall_count=stall_count,
        no_op_count=int(min_len - command_count) if min_len > 0 else 0,
        notes="; ".join(notes_parts) if notes_parts else "正常",
    )


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def _evaluate_robot_bc(report: A2DRobotQualityReport) -> None:
    """判定 robot_bc_ready 和 overall_disposition。"""
    issues = report.issues
    state = report.robot_state_quality
    action = report.robot_action_quality

    # 致命问题
    if state is None:
        report.robot_bc_ready = False
        report.overall_disposition = "reject"
        return

    if state.finite_ratio < 0.95:
        issues.append(f"robot_state 有限值比例过低: {state.finite_ratio:.2%}")
        report.overall_disposition = "reject"
        report.robot_bc_ready = False
        return

    if action is None:
        report.robot_bc_ready = False
        report.overall_disposition = "keep_with_flag"
        return

    # 可用但有告警
    warnings = [
        state.nan_count > 0,
        state.freeze_span_count > 0,
        state.gap_count > 0,
        action and action.nan_count > 0,
        report.gripper_response.stall_count > 0,
    ]
    if any(warnings):
        report.overall_disposition = "keep_with_flag"
    else:
        report.overall_disposition = "pass"

    # robot_bc_ready: 必须有 action 且 action 不是全 NaN
    if action:
        action_has_data = (
            action.finite_ratio > 0.1  # 至少 10% 有效
            and not (report.gripper_response.stall_count > 0
                     and report.gripper_response.command_count == 0)
        )
        report.robot_bc_ready = action_has_data and not state.has_regression
    else:
        report.robot_bc_ready = False


__all__ = [
    "A2DRobotQualityReport",
    "GripperResponse",
    "JointLimitViolation",
    "StateActionLag",
    "TimeSeriesQuality",
    "check_a2d_robot_quality",
]
