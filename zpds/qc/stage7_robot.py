"""Stage 7: 机器人信号质量检查。

检查 TimeSeriesStream（robot_state/action/gripper_state/action）的：
  - 时间戳结构（单调性 / 重复 / 回拨 / 间隔）
  - 关节位置（有限值 / 冻结 / 温度）
  - 动作指令（NaN / 维度 / 指令变化 / 时间覆盖）
  - 时序缺口（短 → flag / 长 → split）

适用性：仅当 profile 声明 ``end_effector`` 为 ``applicable`` 时运行。
"""

from __future__ import annotations

from zpds.core.decisions import Decision, Disposition, ReasonCode, Severity
from zpds.qc.cascade import register_stage

# ---------------------------------------------------------------------------
# QualityIssue → Decision 映射
# ---------------------------------------------------------------------------

_ISSUE_TO_REASON: dict[str, ReasonCode] = {
    # 关节信号
    "joint_position_non_finite": ReasonCode.JOINT_LIMIT_VIOLATION,
    "temperature_out_of_range": ReasonCode.JOINT_LIMIT_VIOLATION,
    "temperature_excessive_nan": ReasonCode.JOINT_LIMIT_VIOLATION,
    "joint_velocity_extreme": ReasonCode.JOINT_LIMIT_VIOLATION,
    "joint_signal_frozen": ReasonCode.JOINT_LIMIT_VIOLATION,
    "joint_dimension_mismatch": ReasonCode.JOINT_LIMIT_VIOLATION,
    # 动作指令
    "action_dimension_mismatch": ReasonCode.COMMAND_TIMEOUT,
    "action_all_nan": ReasonCode.COMMAND_TIMEOUT,
    "action_high_nan": ReasonCode.COMMAND_TIMEOUT,
    "action_no_command_change": ReasonCode.COMMAND_TIMEOUT,
    "action_time_no_overlap": ReasonCode.COMMAND_TIMEOUT,
    "action_time_partial_coverage": ReasonCode.COMMAND_TIMEOUT,
    # 时序缺口
    "timeseries_long_gap": ReasonCode.TIMESTAMP_GAP,
    "timeseries_short_gap": ReasonCode.TIMESTAMP_GAP,
    # 时间戳结构
    "timestamp_not_monotonic": ReasonCode.TIMESTAMP_REGRESSION,
    "timestamp_duplicate": ReasonCode.TIMESTAMP_REGRESSION,
    "timestamp_rollback": ReasonCode.TIMESTAMP_REGRESSION,
    "timeseries_interval_deviation": ReasonCode.CLOCK_MISALIGN,
    # 数据完整性
    "nan_or_inf": ReasonCode.JOINT_LIMIT_VIOLATION,
    "timeseries_length_mismatch": ReasonCode.JOINT_LIMIT_VIOLATION,
}

_SEVERITY_MAP: dict[str, Severity] = {
    "critical": Severity.FATAL,
    "error": Severity.ERROR,
    "warning": Severity.WARN,
    "info": Severity.INFO,
}

_DISPOSITION_MAP: dict[str, Disposition] = {
    "trim": Disposition.TRIM,
    "split": Disposition.SPLIT,
    "keep_with_flag": Disposition.KEEP_WITH_FLAG,
    "quarantine": Disposition.QUARANTINE,
    "keep": Disposition.KEEP,
}


def _quality_issue_to_decision(issue, *, stage: int = 7) -> Decision:
    """将旧式 QualityIssue 转换为级联 Decision。"""
    reason = _ISSUE_TO_REASON.get(
        issue.issue_type, ReasonCode.SOURCE_QUALITY_FLAG,
    )
    severity = _SEVERITY_MAP.get(issue.severity, Severity.WARN)
    disposition = _DISPOSITION_MAP.get(issue.decision, Disposition.KEEP_WITH_FLAG)

    return Decision(
        stage=stage,
        reason=reason,
        severity=severity,
        message=(
            f"[{issue.stream_id}] {issue.issue_type}: "
            f"{issue.details.get('check', '')}"
        ),
        timestamp_ns=issue.start_ns,
        end_timestamp_ns=issue.end_ns,
        disposition=disposition,
        detail={
            "source_issue_type": issue.issue_type,
            "stream_id": issue.stream_id,
            **issue.details,
        },
    )


# ---------------------------------------------------------------------------
# Stage 7 统一入口
# ---------------------------------------------------------------------------


def check(
    ts_streams: dict | None = None,
    *,
    stage_config: dict | None = None,
) -> list[Decision]:
    """Stage 7 统一检查入口：机器人信号质量。

    对每个 TimeSeriesStream 运行结构 / 关节 / 动作 / 缺口检测器，
    聚合所有 QualityIssue 并转为 Decision。

    Parameters
    ----------
    ts_streams : Optional[dict]
        {stream_id: TimeSeriesStream} 映射。为 None 或空时返回空列表。
    stage_config : Optional[dict]
        阈值覆盖（当前透传给各检测器）。

    Returns
    -------
    list[Decision]
    """
    cfg = stage_config or {}
    if not cfg.get("enabled", True):
        return []

    if not ts_streams:
        return []

    from zpds_prepare.detectors.robot.action_quality import (
        detect_action_quality,
    )
    from zpds_prepare.detectors.robot.gap_detection import (
        detect_timeseries_gaps,
    )
    from zpds_prepare.detectors.robot.joint_quality import (
        detect_joint_quality,
    )
    from zpds_prepare.detectors.robot.timeseries_structure import (
        detect_timeseries_structure,
    )

    decisions: list[Decision] = []

    # 按 role 分类，便于 action→state 配对
    state_streams: dict[str, object] = {}
    action_streams: dict[str, object] = {}
    for sid, ts in ts_streams.items():
        role = getattr(ts, "role", "")
        if role == "state":
            state_streams[sid] = ts
        elif role == "action":
            action_streams[sid] = ts

    checked_ids: set[str] = set()

    for stream_id, ts in ts_streams.items():
        issues: list = []

        # 1. 时间戳结构（所有流）
        issues += detect_timeseries_structure(ts, config=cfg)

        # 2. 关节质量（仅 state 流）
        role = getattr(ts, "role", "")
        if role == "state":
            issues += detect_joint_quality(ts, config=cfg)

        # 3. 动作质量（仅 action 流），配对 state 流
        if role == "action":
            # 尝试找配对的 state 流
            matched_state = None
            for sid, ss in state_streams.items():
                if sid not in checked_ids:
                    matched_state = ss
                    break
            issues += detect_action_quality(
                ts, state_stream=matched_state, config=cfg,
            )

        # 4. 时序缺口（所有流）
        issues += detect_timeseries_gaps(ts, config=cfg)

        for issue in issues:
            decisions.append(_quality_issue_to_decision(issue))

        checked_ids.add(stream_id)

    if not decisions:
        decisions.append(
            Decision(
                stage=7,
                reason=ReasonCode.SOURCE_QUALITY_FLAG,
                severity=Severity.INFO,
                message=(
                    f"Robot signal check passed: "
                    f"{len(ts_streams)} stream(s) checked, 0 issues"
                ),
                disposition=Disposition.KEEP,
                detail={"stream_count": len(ts_streams)},
            ),
        )

    return decisions


# ---------------------------------------------------------------------------
# QCCascade 注册入口
# ---------------------------------------------------------------------------


@register_stage(7)
def _check_stage7(context: dict) -> list[Decision]:
    """Stage 7 QCCascade 入口：从 context dict 提取参数并调用 check()。

    Stage 7 是跨流 / 跨 session 的机器人信号检查，只需执行一次。
    使用 ``_stage7_done`` 上下文标记避免多 stream 重复执行。
    """
    if context.get("_stage7_done"):
        return []
    context["_stage7_done"] = True

    stage_config = context.get("stage_config", {})
    if not stage_config.get("enabled", True):
        return []

    # 适用性守卫：仅 robot / end_effector profile 运行
    profile = context.get("profile")
    if profile:
        from zpds.profiles.registry import get

        registered = get(str(profile))
        if registered is not None:
            modalities = registered.modalities
            if modalities.get("end_effector") != "applicable":
                return [
                    Decision(
                        stage=7,
                        reason=ReasonCode.CHECK_NOT_APPLICABLE,
                        severity=Severity.INFO,
                        message=(
                            "Robot signal QC skipped: "
                            "source does not observe a robot end effector"
                        ),
                        disposition=Disposition.KEEP,
                        detail={"applicability": "not_applicable"},
                    ),
                ]

    ts_streams = context.get("time_series_streams")
    if not ts_streams:
        return [
            Decision(
                stage=7,
                reason=ReasonCode.CHECK_NOT_APPLICABLE,
                severity=Severity.INFO,
                message=(
                    "Robot signal QC skipped: "
                    "no time series streams in session"
                ),
                disposition=Disposition.KEEP,
                detail={"applicability": "not_applicable"},
            ),
        ]

    return check(ts_streams=ts_streams, stage_config=stage_config)


__all__ = ["check", "_check_stage7"]
