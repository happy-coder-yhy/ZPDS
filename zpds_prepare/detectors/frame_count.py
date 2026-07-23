"""
帧数一致性检查：比较 index.jsonl 与 meta.json 声明的帧数。
"""

from zpds_prepare.decisions.issue_model import QualityIssue


def detect_frame_count_mismatch(
    index_frame_count: int,
    meta_frame_count: int,
    timestamps_ns: list[int],
    stream_id: str = "ego_rgb",
) -> list[QualityIssue]:
    """比较 index.jsonl 帧数与 meta.json 声明帧数是否一致。

    Args:
        index_frame_count: index.jsonl 中 type=frame 的数量
        meta_frame_count: meta.json 中 recording_stats.total_frames
        timestamps_ns: 帧时间戳列表（用于定位异常时间范围）
        stream_id: 数据流标识

    Returns:
        QualityIssue 列表（一致时为空）
    """
    if index_frame_count == meta_frame_count:
        return []

    diff = abs(index_frame_count - meta_frame_count)
    start_ns = timestamps_ns[0] if timestamps_ns else 0
    end_ns = timestamps_ns[-1] if timestamps_ns else 0

    return [QualityIssue(
        issue_type="frame_count_mismatch",
        stream_id=stream_id,
        start_ns=start_ns,
        end_ns=end_ns,
        severity="warning" if diff <= 5 else "error",
        decision="keep_with_flag",
        details={
            "index_frame_count": index_frame_count,
            "meta_frame_count": meta_frame_count,
            "difference": diff,
        },
    )]
