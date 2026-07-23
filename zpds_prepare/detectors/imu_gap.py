"""
IMU 时间戳缺口检测器。

先去重再检测 IMU 时间戳间隔异常，
区分"小缺口（保留标记）"和"长缺口（建议切分或隔离）"。
"""

import numpy as np
import pandas as pd

from zpds_prepare.decisions.issue_model import QualityIssue


def decide_imu_gap_action(gap_ns: int, split_gap_ns: int) -> str:
    """根据 IMU 缺口大小决定处理方式。

    Args:
        gap_ns: 缺口大小 (纳秒)
        split_gap_ns: 超过此值建议切分 (纳秒)

    Returns:
        "split" | "quarantine" | "keep_with_flag"
    """
    if gap_ns >= split_gap_ns:
        return "split"
    return "keep_with_flag"


def detect_imu_gaps(
    imu: pd.DataFrame,
    expected_interval_ns: int,
    gap_factor: float = 3.0,
    split_gap_ns: int = 1_000_000_000,
    stream_id: str = "ego_imu",
) -> list[QualityIssue]:
    """检测 IMU 时间戳间隔异常。

    IMU 同一时间戳可能有多行（加速度/陀螺仪交错写入），
    先去重再检测间隔。

    Args:
        imu: IMU DataFrame（含 timestamp_ns 列）
        expected_interval_ns: 预期采样间隔（纳秒），如 50Hz → 20,000,000
        gap_factor: 超过 expected * gap_factor 视为缺口
        split_gap_ns: 缺口超过此值建议切分 Segment
        stream_id: 数据流标识

    Returns:
        QualityIssue 列表
    """
    unique_times = np.sort(imu["timestamp_ns"].unique())

    if len(unique_times) <= 1:
        return []

    # 用中位数计算实际正常间隔
    normal_interval_ns = float(np.median(np.diff(unique_times)))
    threshold_ns = max(
        int(expected_interval_ns * gap_factor),
        int(normal_interval_ns * gap_factor),
    )

    issues = []

    for index in range(1, len(unique_times)):
        previous_ns = int(unique_times[index - 1])
        current_ns = int(unique_times[index])
        gap_ns = current_ns - previous_ns

        if gap_ns > threshold_ns:
            estimated_missing = max(0, round(gap_ns / expected_interval_ns) - 1)
            decision = decide_imu_gap_action(gap_ns, split_gap_ns)

            issues.append(QualityIssue(
                issue_type="imu_gap",
                stream_id=stream_id,
                start_ns=previous_ns,
                end_ns=current_ns,
                severity="error" if decision == "split" else "warning",
                decision=decision,
                details={
                    "gap_ns": gap_ns,
                    "gap_s": round(gap_ns / 1_000_000_000, 3),
                    "expected_interval_ns": expected_interval_ns,
                    "normal_interval_ns": int(normal_interval_ns),
                    "gap_factor": round(gap_ns / normal_interval_ns, 2),
                    "threshold_ns": threshold_ns,
                    "split_gap_ns": split_gap_ns,
                    "estimated_missing_samples": estimated_missing,
                    "sample_index": index,
                },
            ))

    return issues
