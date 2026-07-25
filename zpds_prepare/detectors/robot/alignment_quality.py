"""
相机—机器人对齐检查。

针对 camera_robot_alignment.parquet 的校验：
  - 最大/平均/P95 alignment_error_ns
  - 有多少相机帧找不到有效机器人状态
  - 是否跨越机器人时间缺口进行最近邻映射
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zpds_prepare.decisions.issue_model import QualityIssue


def detect_alignment_quality(
    alignment_df: pd.DataFrame,
    config: dict | None = None,
) -> list[QualityIssue]:
    """检查相机帧与机器人时间的对齐质量。

    Args:
        alignment_df: camera_robot_alignment.parquet 的 DataFrame。
        config: 配置字典，支持:
            - max_error_warn_ns: alignment_error 超过此值发 warning (默认 50ms)
            - max_error_error_ns: alignment_error 超过此值发 error (默认 100ms)
            - p95_warn_ns: P95 超过此值发 warning (默认 30ms)

    Returns:
        QualityIssue 列表。
    """
    if config is None:
        config = {}

    max_error_warn_ns = int(config.get("max_error_warn_ns", 50_000_000))
    max_error_error_ns = int(config.get("max_error_error_ns", 100_000_000))
    p95_warn_ns = int(config.get("p95_warn_ns", 30_000_000))
    robot_gap_threshold_ns = int(config.get("robot_gap_threshold_ns", 100_000_000))

    issues: list[QualityIssue] = []

    if alignment_df.empty:
        return issues

    errors = alignment_df["alignment_error_ns"].dropna()
    if len(errors) == 0:
        return issues

    # ---- 整体统计 ----
    mean_err = float(errors.mean())
    max_err = float(errors.max())
    p95_err = float(np.percentile(errors, 95))
    p99_err = float(np.percentile(errors, 99))

    # ---- 1. 最大对齐误差 ----
    if max_err > max_error_error_ns:
        # 找最大误差的帧
        worst_idx = int(errors.idxmax())
        worst_row = alignment_df.iloc[worst_idx]
        issues.append(QualityIssue(
            issue_type="camera_robot_alignment_error",
            stream_id=str(worst_row.get("camera_stream_id", "unknown")),
            start_ns=int(worst_row.get("camera_timestamp_ns", 0)),
            end_ns=int(worst_row.get("camera_timestamp_ns", 0)),
            severity="error",
            decision="keep_with_flag",
            details={
                "max_error_ns": int(max_err),
                "mean_error_ns": int(mean_err),
                "p95_error_ns": int(p95_err),
                "p99_error_ns": int(p99_err),
                "threshold_error_ns": max_error_error_ns,
                "worst_camera_frame": int(worst_row["source_frame_index"]),
                "check": "alignment_max_error",
            },
        ))
    elif max_err > max_error_warn_ns:
        worst_idx = int(errors.idxmax())
        worst_row = alignment_df.iloc[worst_idx]
        issues.append(QualityIssue(
            issue_type="camera_robot_alignment_error",
            stream_id=str(worst_row.get("camera_stream_id", "unknown")),
            start_ns=int(worst_row.get("camera_timestamp_ns", 0)),
            end_ns=int(worst_row.get("camera_timestamp_ns", 0)),
            severity="warning" if p95_err > p95_warn_ns else "info",
            decision="keep_with_flag",
            details={
                "max_error_ns": int(max_err),
                "mean_error_ns": int(mean_err),
                "p95_error_ns": int(p95_err),
                "threshold_warn_ns": max_error_warn_ns,
                "check": "alignment_max_error",
            },
        ))

    # ---- 2. 相机帧无机器人状态 ----
    unavailable = alignment_df[alignment_df["mapping_method"] == "unavailable"]
    if len(unavailable) > 0:
        first = unavailable.iloc[0]
        issues.append(QualityIssue(
            issue_type="camera_frame_without_robot_state",
            stream_id=str(first.get("camera_stream_id", "unknown")),
            start_ns=int(first.get("camera_timestamp_ns", 0)) if pd.notna(first.get("camera_timestamp_ns")) else 0,
            end_ns=int(unavailable.iloc[-1].get("camera_timestamp_ns", 0)) if pd.notna(unavailable.iloc[-1].get("camera_timestamp_ns")) else 0,
            severity="error",
            decision="keep_with_flag",
            details={
                "unavailable_frame_count": len(unavailable),
                "total_frames": len(alignment_df),
                "check": "alignment_unavailable",
            },
        ))

    # ---- 3. 推断时间戳标记 ----
    inferred = alignment_df[alignment_df["mapping_method"].str.contains("synthetic|inferred", na=False)]
    if len(inferred) > 0:
        issues.append(QualityIssue(
            issue_type="camera_timestamp_inferred",
            stream_id="all",
            start_ns=0,
            end_ns=0,
            severity="warning",
            decision="keep_with_flag",
            details={
                "inferred_frame_count": len(inferred),
                "total_frames": len(alignment_df),
                "inferred_ratio": round(len(inferred) / len(alignment_df), 4),
                "note": "使用推断时间而非直接映射的相机帧",
                "check": "alignment_inferred",
            },
        ))

    # ---- 4. 跨缺口映射 ----
    # 检查相邻相机帧的 robot_row_index 是否出现大跳跃
    # （说明中间跨越了部分机器人数据）
    robot_rows = alignment_df["robot_row_index"].dropna()
    if len(robot_rows) >= 2:
        robot_gaps = np.diff(robot_rows.values)
        large_gaps = robot_gaps > 1
        if large_gaps.any():
            gap_count = int(large_gaps.sum())
            max_gap = int(robot_gaps[large_gaps].max()) if gap_count > 0 else 0
            issues.append(QualityIssue(
                issue_type="camera_robot_alignment_gap",
                stream_id="all",
                start_ns=0,
                end_ns=0,
                severity="warning",
                decision="keep_with_flag",
                details={
                    "large_gap_count": gap_count,
                    "max_robot_row_gap": max_gap,
                    "note": (
                        f"相机帧的 robot_row_index 存在 {gap_count} 次跳跃 >1，"
                        f"说明相机采样稀疏，两帧之间跨越了多行机器人数据"
                    ),
                    "check": "alignment_robot_gap",
                },
            ))

    return issues


__all__ = ["detect_alignment_quality"]
