"""
Action 质量检查。

针对 robot_action / gripper_action 流：
  - 动作数组维度
  - NaN / Inf 分布
  - 是否长时间完全无指令变化
  - 动作时间是否覆盖状态时间
"""

from __future__ import annotations

import numpy as np

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.readers.session_model import TimeSeriesStream


def detect_action_quality(
    ts_stream: TimeSeriesStream,
    state_stream: TimeSeriesStream | None = None,
    config: dict | None = None,
) -> list[QualityIssue]:
    """检查关节动作流的质量。

    Args:
        ts_stream: robot_action 或 gripper_action TimeSeriesStream。
        state_stream: 对应的 state 流（用于时间覆盖检查），可选。
        config: 可选配置。

    Returns:
        QualityIssue 列表。
    """
    if config is None:
        config = {}

    issues: list[QualityIssue] = []
    stream_id = ts_stream.stream_id
    timestamps = np.array(ts_stream.timestamps_ns, dtype=np.int64)
    if hasattr(ts_stream.rows, "select_dtypes"):
        numeric_frame = ts_stream.rows.select_dtypes(include=[np.number])
        rows = (
            numeric_frame.to_numpy(dtype=np.float64)
            if not numeric_frame.empty
            else np.zeros((len(ts_stream.rows), 0), dtype=np.float64)
        )
    else:
        rows = np.asarray(ts_stream.rows, dtype=np.float64)
    fields = ts_stream.fields
    num_samples = len(timestamps)
    num_fields = ts_stream.num_fields

    if num_samples == 0 or rows.size == 0:
        return issues

    # ---- 1. 动作数组维度 ----
    expected_joints = ts_stream.metadata.get("num_joints")
    if expected_joints is not None and num_fields > 0:
        # 每个 joint 有 N 个字段，总字段数应为 expected_joints * N
        # 用 field name 中的位置字段数推断
        pos_fields = [f for f in fields if "positions" in f["name"]]
        if pos_fields and len(pos_fields) != expected_joints:
            issues.append(QualityIssue(
                issue_type="action_dimension_mismatch",
                stream_id=stream_id,
                start_ns=int(timestamps[0]),
                end_ns=int(timestamps[-1]),
                severity="error",
                decision="quarantine",
                details={
                    "expected_joint_count": expected_joints,
                    "actual_position_field_count": len(pos_fields),
                    "total_fields": num_fields,
                    "check": "action_dimension",
                },
            ))

    # ---- 2. NaN / Inf 分布 ----
    if rows.size > 0:
        nan_mask = ~np.isfinite(rows)
        nan_ratio = float(nan_mask.sum() / rows.size)

        # 完全全 NaN
        if nan_ratio > 0.99:
            issues.append(QualityIssue(
                issue_type="action_all_nan",
                stream_id=stream_id,
                start_ns=int(timestamps[0]),
                end_ns=int(timestamps[-1]),
                severity="warning",
                decision="keep_with_flag",
                details={
                    "nan_ratio": round(nan_ratio, 4),
                    "note": "本 Episode 未执行动作指令，全部为 NaN",
                    "check": "action_nan",
                },
            ))
        elif nan_ratio > 0.5:
            # 按列统计 NaN
            col_nan_ratios = {
                fields[i]["name"]: round(float(nan_mask[:, i].mean()), 4)
                for i in range(min(num_fields, len(fields)))
                if i < nan_mask.shape[1]
            }
            issues.append(QualityIssue(
                issue_type="action_high_nan",
                stream_id=stream_id,
                start_ns=int(timestamps[0]),
                end_ns=int(timestamps[-1]),
                severity="warning",
                decision="keep_with_flag",
                details={
                    "overall_nan_ratio": round(nan_ratio, 4),
                    "per_field_nan_ratios": col_nan_ratios,
                    "note": "action 流 NaN 通常是插值空白，非数据损坏",
                    "check": "action_nan",
                },
            ))

    # ---- 3. 长时间无指令变化 ----
    position_indices = [i for i, f in enumerate(fields) if "positions" in f["name"]]
    if position_indices and num_samples >= 2:
        pos_cols = rows[:, position_indices]
        finite_mask = np.all(np.isfinite(pos_cols), axis=1)
        if not finite_mask.any():
            # 指令变化检测需要有限值——全部 NaN，跳过
            pass
        else:
            finite_pos = pos_cols[finite_mask]
            if len(finite_pos) >= 2:
                # 检查是否存在连续不变的指令
                changes = np.any(np.abs(np.diff(finite_pos, axis=0)) > 0.0, axis=1)
                no_change_count = int((~changes).sum())
                if no_change_count > 0 and no_change_count == len(changes):
                    # 整个 action 流无指令变化
                    issues.append(QualityIssue(
                        issue_type="action_no_command_change",
                        stream_id=stream_id,
                        start_ns=int(timestamps[0]),
                        end_ns=int(timestamps[-1]),
                        severity="info",
                        decision="keep_with_flag",
                        details={
                            "note": "整个 action 流无指令变化——可能本段仅回放状态",
                            "check": "action_command_change",
                        },
                    ))

    # ---- 4. 动作时间覆盖状态时间 ----
    if state_stream is not None:
        state_ts = np.array(state_stream.timestamps_ns, dtype=np.int64)
        if len(state_ts) > 0 and num_samples > 0:
            action_start = timestamps[0]
            action_end = timestamps[-1]
            state_start = state_ts[0]
            state_end = state_ts[-1]

            # 动作是否完全在状态范围外
            if action_end < state_start or action_start > state_end:
                issues.append(QualityIssue(
                    issue_type="action_time_no_overlap",
                    stream_id=stream_id,
                    start_ns=int(action_start),
                    end_ns=int(action_end),
                    severity="warning",
                    decision="keep_with_flag",
                    details={
                        "action_range_ns": [int(action_start), int(action_end)],
                        "state_range_ns": [int(state_start), int(state_end)],
                        "check": "action_time_coverage",
                    },
                ))
            else:
                # 动作覆盖率
                overlap_start = max(action_start, state_start)
                overlap_end = min(action_end, state_end)
                state_span = state_end - state_start
                overlap_span = overlap_end - overlap_start
                coverage = overlap_span / state_span if state_span > 0 else 0.0

                if coverage < 0.5:
                    issues.append(QualityIssue(
                        issue_type="action_time_partial_coverage",
                        stream_id=stream_id,
                        start_ns=int(action_start),
                        end_ns=int(action_end),
                        severity="info",
                        decision="keep_with_flag",
                        details={
                            "coverage_ratio": round(coverage, 2),
                            "state_duration_s": round(state_span / 1e9, 2),
                            "overlap_duration_s": round(overlap_span / 1e9, 2),
                            "check": "action_time_coverage",
                        },
                    ))

    return issues


__all__ = ["detect_action_quality"]
