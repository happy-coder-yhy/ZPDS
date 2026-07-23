"""
视频时间戳缺口检测器。

检测 index.jsonl 中帧间时间戳间隔异常，
区分"小缺口（保留标记）"和"长缺口（建议切分）"。
"""

import numpy as np

from zpds_prepare.decisions.issue_model import QualityIssue


def decide_gap_action(gap_ns: int, split_gap_ns: int) -> str:
    """根据缺口大小决定处理方式。

    Args:
        gap_ns: 缺口大小 (纳秒)
        split_gap_ns: 超过此值建议切分 (纳秒)

    Returns:
        "split" | "keep_with_flag"
    """
    if gap_ns >= split_gap_ns:
        return "split"
    return "keep_with_flag"


def detect_timestamp_gaps(
    timestamps_ns: list[int],
    expected_interval_ns: int,
    gap_factor: float = 2.0,
    split_gap_ns: int = 500_000_000,
    stream_id: str = "ego_rgb",
) -> list[QualityIssue]:
    """检测视频时间戳间隔异常。

    Args:
        timestamps_ns: 帧时间戳列表（已排序）
        expected_interval_ns: 预期帧间隔（纳秒），如 30fps → ~33,333,333
        gap_factor: 超过 expected * gap_factor 视为缺口
        split_gap_ns: 缺口超过此值建议切分 Segment
        stream_id: 数据流标识

    Returns:
        QualityIssue 列表
    """
    threshold_ns = int(expected_interval_ns * gap_factor)
    issues = []

    for index in range(1, len(timestamps_ns)):
        previous_ns = timestamps_ns[index - 1]
        current_ns = timestamps_ns[index]
        gap_ns = current_ns - previous_ns

        if gap_ns > threshold_ns:
            estimated_missing = max(0, round(gap_ns / expected_interval_ns) - 1)
            decision = decide_gap_action(gap_ns, split_gap_ns)

            issues.append(QualityIssue(
                issue_type="timestamp_gap",
                stream_id=stream_id,
                start_ns=previous_ns,
                end_ns=current_ns,
                severity="error" if decision == "split" else "warning",
                decision=decision,
                details={
                    "gap_ns": gap_ns,
                    "gap_ms": round(gap_ns / 1_000_000, 2),
                    "expected_interval_ns": expected_interval_ns,
                    "gap_factor": round(gap_ns / expected_interval_ns, 2),
                    "threshold_ns": threshold_ns,
                    "split_gap_ns": split_gap_ns,
                    "estimated_missing_frames": estimated_missing,
                    "frame_index": index,
                },
            ))

    return issues
