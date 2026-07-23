"""
坏帧检测器：扫描 MKV 中解码失败 / None 的帧。
"""

import cv2
from pathlib import Path

from zpds_prepare.decisions.issue_model import QualityIssue


def detect_bad_frames(
    video_path: str,
    timestamps_ns: list[int],
    stream_id: str = "ego_rgb",
) -> list[QualityIssue]:
    """检测 MKV 中解码失败的帧。

    如果坏帧形成连续区间，以区间的 start/end 时间戳表示；
    如果坏帧零星分布，以整个 session 范围表示（方便标记）。

    Args:
        video_path: MKV 文件路径
        timestamps_ns: index.jsonl 帧时间戳列表
        stream_id: 数据流标识

    Returns:
        QualityIssue 列表（无坏帧时为空）
    """
    if not Path(video_path).exists():
        return []

    cap = cv2.VideoCapture(video_path)
    bad_indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            bad_indices.append(frame_idx)
        frame_idx += 1
    cap.release()

    if not bad_indices:
        return []

    total_frames = frame_idx
    # 将坏帧索引映射到时间戳
    n = min(len(timestamps_ns), total_frames)

    # 合并连续坏帧区间（复用黑屏检测的区间合并思路）
    spans = _merge_consecutive(bad_indices, timestamps_ns[:n])

    issues = []
    for start_ns, end_ns, count in spans:
        issues.append(QualityIssue(
            issue_type="bad_frame",
            stream_id=stream_id,
            start_ns=start_ns,
            end_ns=end_ns,
            severity="error",
            decision="keep_with_flag",
            details={
                "bad_frame_count": count,
                "total_frames_scanned": total_frames,
                "bad_ratio": round(count / max(total_frames, 1), 4),
            },
        ))

    return issues


def _merge_consecutive(
    indices: list[int],
    timestamps_ns: list[int],
) -> list[tuple[int, int, int]]:
    """将连续索引合并为 (start_ns, end_ns, count)。"""
    if not indices or not timestamps_ns:
        return []

    spans = []
    start_idx = indices[0]
    prev_idx = indices[0]

    for i in range(1, len(indices)):
        current = indices[i]
        if current != prev_idx + 1:
            # 区间结束
            spans.append(_make_span(start_idx, prev_idx, timestamps_ns))
            start_idx = current
        prev_idx = current

    # 最后一个区间
    spans.append(_make_span(start_idx, prev_idx, timestamps_ns))
    return spans


def _make_span(
    start_idx: int,
    end_idx: int,
    timestamps_ns: list[int],
) -> tuple[int, int, int]:
    """将起止帧号转为 (start_ns, end_ns, frame_count)。"""
    start_ns = (
        timestamps_ns[start_idx]
        if start_idx < len(timestamps_ns) else 0
    )
    end_ns = (
        timestamps_ns[end_idx]
        if end_idx < len(timestamps_ns) else 0
    )
    # 加上最后一帧的近似持续
    if end_idx > 0 and end_idx < len(timestamps_ns):
        end_ns += timestamps_ns[end_idx] - timestamps_ns[end_idx - 1]
    count = end_idx - start_idx + 1
    return start_ns, end_ns, count
