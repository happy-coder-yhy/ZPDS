"""
帧数一致性检查：比较多来源声明的帧数。

支持两种调用模式:
  1. 传统 (Guida/Dunjia/UMI): index_frame_count vs meta_frame_count
  2. EPIC: declared_count / decoded_count / timestamp_count / annotation_max_frame

调用方可只传入已知的指标，缺失项传 None 即可跳过对应比较。
"""

from zpds_prepare.decisions.issue_model import QualityIssue


def detect_frame_count_mismatch(
    *,
    stream_id: str = "ego_rgb",
    timestamps_ns: list[int] | None = None,
    # 传统参数
    index_frame_count: int | None = None,
    meta_frame_count: int | None = None,
    # EPIC 参数
    declared_count: int | None = None,
    decoded_count: int | None = None,
    timestamp_count: int | None = None,
    annotation_max_frame: int | None = None,
) -> list[QualityIssue]:
    """比较多来源帧数声明是否一致。

    会检查所有传入的非 None 值之间的差异。
    差异 ≤5 → warning, >5 → error。

    Args:
        stream_id: 数据流标识
        timestamps_ns: 帧时间戳列表（用于定位异常时间范围）
        index_frame_count: index.jsonl 中帧数
        meta_frame_count: meta.json 声明帧数
        declared_count: FFprobe 声明帧数
        decoded_count: 实际可解码帧数
        timestamp_count: PTS 时间戳数量
        annotation_max_frame: 标注中最大 frame_index + 1

    Returns:
        QualityIssue 列表（一致时为空）
    """
    ts_list = timestamps_ns or []
    start_ns = ts_list[0] if ts_list else 0
    end_ns = ts_list[-1] if ts_list else 0

    # 收集所有已知计数
    sources: list[tuple[str, int]] = []
    for label, val in [
        ("index_frame_count", index_frame_count),
        ("meta_frame_count", meta_frame_count),
        ("declared_count", declared_count),
        ("decoded_count", decoded_count),
        ("timestamp_count", timestamp_count),
    ]:
        if val is not None:
            sources.append((label, val))

    issues: list[QualityIssue] = []

    # 两两比较
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            label_a, val_a = sources[i]
            label_b, val_b = sources[j]
            if val_a == val_b:
                continue
            diff = abs(val_a - val_b)
            issues.append(QualityIssue(
                issue_type="frame_count_mismatch",
                stream_id=stream_id,
                start_ns=start_ns,
                end_ns=end_ns,
                severity="warning" if diff <= 5 else "error",
                decision="keep_with_flag",
                details={
                    "source_a": label_a,
                    "value_a": val_a,
                    "source_b": label_b,
                    "value_b": val_b,
                    "difference": diff,
                },
            ))

    # 标注越界检查
    if annotation_max_frame is not None and decoded_count is not None:
        if annotation_max_frame > decoded_count:
            issues.append(QualityIssue(
                issue_type="annotation_frame_out_of_range",
                stream_id=stream_id,
                start_ns=start_ns,
                end_ns=end_ns,
                severity="error",
                decision="quarantine",
                details={
                    "annotation_max_frame": annotation_max_frame,
                    "decoded_frame_count": decoded_count,
                    "excess": annotation_max_frame - decoded_count,
                },
            ))

    return issues
