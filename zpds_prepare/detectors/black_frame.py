"""
黑屏检测器。

从 MKV 逐帧扫描，把连续黑帧合并成区间，
再根据区间位置决定 trim / split / keep_with_flag。
"""

import cv2
import numpy as np
from pathlib import Path

from zpds_prepare.decisions.issue_model import QualityIssue


# ================================================================
# 底层：连续 True 区间合并
# ================================================================

def find_continuous_true_spans(
    flags: list[bool],
    timestamps_ns: list[int],
    min_duration_ns: int,
) -> list[tuple[int, int, int]]:
    """把连续的 True 合并成区间。

    Args:
        flags: 逐帧黑屏标记
        timestamps_ns: 逐帧设备时间戳（与 flags 等长）
        min_duration_ns: 区间持续时间 ≥ 此值才保留（纳秒）

    Returns:
        [(start_ns, end_ns, frame_count), ...]
    """
    if len(flags) != len(timestamps_ns):
        raise ValueError("flags 与 timestamps_ns 长度不一致")

    spans = []
    start_index = None

    for index, is_true in enumerate(flags):
        if is_true and start_index is None:
            start_index = index

        is_last = index == len(flags) - 1

        if start_index is not None and (not is_true or is_last):
            # 确定结束索引：当前为 False → 前一个为结束；当前为最末且 True → 当前为结束
            end_index = index if is_true and is_last else index - 1

            start_ns = timestamps_ns[start_index]
            end_ns = timestamps_ns[end_index]

            # 加上最后一帧的大致持续时间
            if end_index > 0:
                frame_interval_ns = timestamps_ns[end_index] - timestamps_ns[end_index - 1]
            else:
                frame_interval_ns = 0
            end_ns += frame_interval_ns

            duration_ns = end_ns - start_ns

            if duration_ns >= min_duration_ns:
                spans.append((start_ns, end_ns, end_index - start_index + 1))

            start_index = None

    return spans


# ================================================================
# 决策：黑屏区间如何处置
# ================================================================

def decide_black_span(
    start_ns: int,
    end_ns: int,
    session_start_ns: int,
    session_end_ns: int,
    edge_tolerance_ns: int = 1_000_000_000,
) -> str:
    """根据黑屏区间位置决定 trim / split / keep_with_flag。

    - 开头连续黑屏 → trim
    - 结尾连续黑屏 → trim
    - 中间长黑屏   → split
    - 中间短黑屏   → keep_with_flag

    Args:
        start_ns: 黑屏起始 (设备时钟)
        end_ns: 黑屏结束 (设备时钟)
        session_start_ns: 会话起始时间
        session_end_ns: 会话结束时间
        edge_tolerance_ns: 距离首尾多远算「边缘」(默认 1 秒)
    """
    near_start = (start_ns - session_start_ns) <= edge_tolerance_ns
    near_end = (session_end_ns - end_ns) <= edge_tolerance_ns

    if near_start or near_end:
        return "trim"

    return "split"


# ================================================================
# 主检测函数
# ================================================================

def detect_black_frames(
    video_path: str,
    timestamps_ns: list[int],
    mean_intensity_threshold: float = 5.0,
    min_duration_ns: int = 500_000_000,
    edge_tolerance_ns: int = 1_000_000_000,
) -> list[QualityIssue]:
    """检测黑屏帧并合并为区间。

    Args:
        video_path: MKV 文件路径
        timestamps_ns: index.jsonl 帧时间戳列表
        mean_intensity_threshold: 灰度均值低于此值视为黑屏 (0-255)
        min_duration_ns: 连续黑屏最短持续时长 (纳秒)，默认 0.5s
        edge_tolerance_ns: 距首尾容差 (纳秒)，默认 1s

    Returns:
        QualityIssue 列表
    """
    if not Path(video_path).exists():
        return []

    cap = cv2.VideoCapture(video_path)
    black_flags = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_flags.append(gray.mean() < mean_intensity_threshold)
    cap.release()

    # 截齐长度（防止视频和 index 帧数不一致）
    n = min(len(black_flags), len(timestamps_ns))
    black_flags = black_flags[:n]
    ts_slice = timestamps_ns[:n]

    if not ts_slice:
        return []

    session_start_ns = ts_slice[0]
    session_end_ns = ts_slice[-1]

    # 合并连续区间
    spans = find_continuous_true_spans(black_flags, ts_slice, min_duration_ns)

    # 决定每个区间
    issues = []
    for start_ns, end_ns, frame_count in spans:
        decision = decide_black_span(
            start_ns, end_ns,
            session_start_ns, session_end_ns,
            edge_tolerance_ns,
        )

        issues.append(QualityIssue(
            issue_type="continuous_black_frames",
            stream_id="ego_rgb",
            start_ns=start_ns,
            end_ns=end_ns,
            severity="warning" if decision == "trim" else "error",
            decision=decision,
            details={
                "mean_intensity_threshold": mean_intensity_threshold,
                "min_duration_ns": min_duration_ns,
                "frame_count": frame_count,
                "edge_tolerance_ns": edge_tolerance_ns,
            },
        ))

    return issues
