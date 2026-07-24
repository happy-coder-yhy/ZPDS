"""
生成 {stream_id}_sample_map.parquet：输出帧 ↔ 源帧映射表。
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_map(
    index_frames: list[dict],
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
) -> pd.DataFrame:
    """为 CFR 输出帧生成到源帧的最近邻映射表。

    Args:
        index_frames: index.jsonl 中 type=frame 的列表 (含 seq, timestamp_ns, segment)
        source_start_ns: 源时间戳起始
        source_end_ns: 源时间戳结束
        target_fps: 目标恒定帧率

    Returns:
        DataFrame，列：
        - output_frame_index
        - output_timestamp_ns
        - source_seq
        - source_timestamp_ns
        - source_file
        - source_frame_index
        - mapping_method
        - time_error_ns
    """
    # 筛选 Span 内的源帧
    span_frames = [
        f for f in index_frames
        if source_start_ns <= f["timestamp_ns"] <= source_end_ns
    ]
    span_timestamps = np.array([f["timestamp_ns"] for f in span_frames], dtype=np.int64)

    if len(span_frames) == 0:
        raise ValueError("Span 内没有帧")

    frame_interval_ns = int(1_000_000_000 / target_fps)
    segment_duration_ns = source_end_ns - source_start_ns

    rows = []
    output_frame_index = 0
    output_time_ns = 0

    while output_time_ns < segment_duration_ns:
        target_source_time = source_start_ns + output_time_ns

        # 最近邻
        nearest_idx = int(np.argmin(np.abs(span_timestamps - target_source_time)))
        source_row = span_frames[nearest_idx]
        source_ts = int(source_row["timestamp_ns"])

        rows.append({
            "output_frame_index": output_frame_index,
            "output_timestamp_ns": output_time_ns,
            "source_seq": int(source_row["seq"]),
            "source_timestamp_ns": source_ts,
            "source_file": f"color_{source_row['segment']:06d}.mkv",
            "source_frame_index": int(source_row["seq"]),  # seq 即 MKV 内帧号
            "mapping_method": "nearest",
            "time_error_ns": int(source_ts - target_source_time),
        })

        output_frame_index += 1
        output_time_ns += frame_interval_ns

    return pd.DataFrame(rows)


def write_sample_map(sample_map: pd.DataFrame, output_dir: str,
                     stream_id: str = "ego_rgb") -> str:
    """写出 sample_map 为 Parquet 文件。

    Args:
        sample_map: 映射表 DataFrame
        output_dir: Prepared Segment 根目录
        stream_id: 视频流标识，文件名生成为 {stream_id}_sample_map.parquet

    Returns:
        输出文件路径
    """
    maps_dir = Path(output_dir) / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    output_path = maps_dir / f"{stream_id}_sample_map.parquet"
    sample_map.to_parquet(str(output_path), index=False)
    return str(output_path)


def generate_sample_map_from_timestamps(
    timestamps_ns: list[int],
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
) -> pd.DataFrame:
    """为 CFR 输出帧生成最近邻映射表 — 不需要 index.jsonl 中的 segment 字段。

    适用于遁甲等没有 Guida 式 index.jsonl 的数据源。

    Args:
        timestamps_ns: 源帧时间戳列表（已排序）
        source_start_ns: 源时间戳起始
        source_end_ns: 源时间戳结束
        target_fps: 目标恒定帧率

    Returns:
        DataFrame，列：
        - output_frame_index
        - output_timestamp_ns
        - source_frame_index
        - source_timestamp_ns
        - mapping_method
        - time_error_ns
    """
    span_ts = np.array([
        ts for ts in timestamps_ns
        if source_start_ns <= ts <= source_end_ns
    ], dtype=np.int64)

    if len(span_ts) == 0:
        raise ValueError("Span 内没有帧")

    frame_interval_ns = int(1_000_000_000 / target_fps)
    segment_duration_ns = source_end_ns - source_start_ns

    rows = []
    output_frame_index = 0
    output_time_ns = 0

    while output_time_ns < segment_duration_ns:
        target_source_time = source_start_ns + output_time_ns
        nearest_idx = int(np.argmin(np.abs(span_ts - target_source_time)))
        source_ts = int(span_ts[nearest_idx])

        rows.append({
            "output_frame_index": output_frame_index,
            "output_timestamp_ns": output_time_ns,
            "source_frame_index": nearest_idx,
            "source_timestamp_ns": source_ts,
            "mapping_method": "nearest",
            "time_error_ns": int(source_ts - target_source_time),
        })

        output_frame_index += 1
        output_time_ns += frame_interval_ns

    return pd.DataFrame(rows)
