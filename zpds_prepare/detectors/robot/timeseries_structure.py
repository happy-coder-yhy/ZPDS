"""
机器人时序结构检查。

检查 TimeSeriesStream 的基本结构完整性：
  - timestamp 单调性 / 重复 / 回拨
  - NaN / Inf
  - 各字段第一维一致
  - 关节数组维度一致
  - joint_names 数量匹配
"""

from __future__ import annotations

import numpy as np

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.readers.session_model import TimeSeriesStream


def detect_timeseries_structure(
    ts_stream: TimeSeriesStream,
    config: dict | None = None,
) -> list[QualityIssue]:
    """检查单个 TimeSeriesStream 的时序结构。

    Args:
        ts_stream: TimeSeriesStream 对象。
        config: 可选配置（暂未使用，为扩展预留）。

    Returns:
        QualityIssue 列表。
    """
    if config is None:
        config = {}

    issues: list[QualityIssue] = []
    stream_id = ts_stream.stream_id
    timestamps = np.array(ts_stream.timestamps_ns, dtype=np.int64)
    if hasattr(ts_stream.rows, "select_dtypes"):
        # UMI 等时序行含 log_time/publish_time 等非数值列，
        # 数值检查只取数值列，行数保持原样。
        numeric_frame = ts_stream.rows.select_dtypes(include=[np.number])
        rows = (
            numeric_frame.to_numpy(dtype=np.float64)
            if not numeric_frame.empty
            else np.zeros((len(ts_stream.rows), 0), dtype=np.float64)
        )
    else:
        rows = np.asarray(ts_stream.rows, dtype=np.float64)
    expected_rate = ts_stream.expected_rate_hz
    expected_interval_ns = int(1e9 / expected_rate) if expected_rate else None

    num_samples = len(timestamps)
    num_fields = ts_stream.num_fields

    # ---- 1. timestamp 单调性 ----
    if num_samples >= 2:
        diffs = np.diff(timestamps)
        non_monotonic = diffs <= 0
        if non_monotonic.any():
            bad_idx = int(np.where(non_monotonic)[0][0])
            issues.append(QualityIssue(
                issue_type="timestamp_not_monotonic",
                stream_id=stream_id,
                start_ns=int(timestamps[max(0, bad_idx - 1)]),
                end_ns=int(timestamps[min(num_samples - 1, bad_idx + 1)]),
                severity="error",
                decision="keep_with_flag",
                details={
                    "violation_count": int(non_monotonic.sum()),
                    "first_violation_index": bad_idx,
                    "example_diff_ns": int(diffs[bad_idx]),
                    "check": "timestamp_monotonic",
                },
            ))

        # 重复时间戳
        duplicates = diffs == 0
        if duplicates.any():
            dup_idx = int(np.where(duplicates)[0][0])
            issues.append(QualityIssue(
                issue_type="timestamp_duplicate",
                stream_id=stream_id,
                start_ns=int(timestamps[dup_idx]),
                end_ns=int(timestamps[dup_idx + 1]),
                severity="warning",
                decision="keep_with_flag",
                details={
                    "duplicate_count": int(duplicates.sum()),
                    "first_duplicate_index": dup_idx,
                    "check": "timestamp_duplicate",
                },
            ))

        # 回拨 (负间隔)
        rollbacks = diffs < 0
        if rollbacks.any():
            rb_idx = int(np.where(rollbacks)[0][0])
            issues.append(QualityIssue(
                issue_type="timestamp_rollback",
                stream_id=stream_id,
                start_ns=int(timestamps[rb_idx]),
                end_ns=int(timestamps[rb_idx + 1]),
                severity="critical",
                decision="split",
                details={
                    "rollback_count": int(rollbacks.sum()),
                    "first_rollback_index": rb_idx,
                    "rollback_ns": int(diffs[rb_idx]),
                    "check": "timestamp_rollback",
                },
            ))

        # 间隔异常 (vs 期望采样率)
        if expected_interval_ns is not None and not non_monotonic.any():
            normal_diffs = diffs[diffs > 0]
            if len(normal_diffs) > 0:
                median_interval = float(np.median(normal_diffs))
                ratio = median_interval / expected_interval_ns if expected_interval_ns > 0 else 1.0
                if ratio < 0.5 or ratio > 2.0:
                    issues.append(QualityIssue(
                        issue_type="timeseries_interval_deviation",
                        stream_id=stream_id,
                        start_ns=int(timestamps[0]),
                        end_ns=int(timestamps[-1]),
                        severity="warning",
                        decision="keep_with_flag",
                        details={
                            "expected_interval_ns": expected_interval_ns,
                            "actual_median_interval_ns": int(median_interval),
                            "ratio": round(ratio, 2),
                            "expected_rate_hz": expected_rate,
                            "check": "timeseries_interval",
                        },
                    ))

    # ---- 2. NaN / Inf ----
    if num_fields > 0 and rows.size > 0:
        nan_count = int(np.isnan(rows).sum())
        inf_count = int(np.isinf(rows).sum())
        total = rows.size

        if nan_count > 0 or inf_count > 0:
            nan_ratio = nan_count / total if total > 0 else 0.0
            # action 流 NaN 是插值空白（非数据损坏），降低严重度
            is_action = ("action" in stream_id or "command" in ts_stream.modality)
            if is_action:
                severity = "warning" if nan_ratio > 0.5 else "info"
            else:
                severity = "error" if nan_ratio > 0.5 else ("warning" if nan_ratio > 0 else "info")
            issues.append(QualityIssue(
                issue_type="nan_or_inf",
                stream_id=stream_id,
                start_ns=int(timestamps[0]),
                end_ns=int(timestamps[-1]),
                severity=severity,
                decision="keep_with_flag",
                details={
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "total_elements": total,
                    "nan_ratio": round(nan_ratio, 4),
                    "check": "nan_inf",
                    "note": (
                        "action 流中的 NaN 通常是未执行动作的插值空白，"
                        "属于正常现象"
                    ) if "action" in stream_id else "",
                },
            ))

    # ---- 3. 第一维一致性 ----
    if num_samples != len(rows) and num_fields > 0:
        issues.append(QualityIssue(
            issue_type="timeseries_length_mismatch",
            stream_id=stream_id,
            start_ns=int(timestamps[0]) if num_samples > 0 else 0,
            end_ns=int(timestamps[-1]) if num_samples > 0 else 0,
            severity="error",
            decision="quarantine",
            details={
                "timestamp_count": num_samples,
                "data_row_count": len(rows),
                "check": "length_consistency",
            },
        ))

    # ---- 4. 关节维度 ----
    expected_joints = ts_stream.metadata.get("num_joints")
    joint_names = ts_stream.metadata.get("joint_names", [])
    if expected_joints is not None and joint_names:
        actual_names = len(joint_names)
        if actual_names != expected_joints:
            issues.append(QualityIssue(
                issue_type="joint_dimension_mismatch",
                stream_id=stream_id,
                start_ns=int(timestamps[0]) if num_samples > 0 else 0,
                end_ns=int(timestamps[-1]) if num_samples > 0 else 0,
                severity="error",
                decision="quarantine",
                details={
                    "expected_joint_count": expected_joints,
                    "actual_name_count": actual_names,
                    "check": "joint_dimension",
                },
            ))

    return issues


__all__ = ["detect_timeseries_structure"]
