"""
TimeSeries 规范化：裁剪到 Segment 范围，转换为相对时间，写出 Parquet。

处理 A2D 的 robot_state / robot_action / gripper_state / gripper_action 流。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zpds_prepare.readers.session_model import TimeSeriesStream


def normalize_time_series(
    stream: TimeSeriesStream,
    source_start_ns: int,
    source_end_ns: int,
) -> pd.DataFrame:
    """规范化 TimeSeriesStream 为 Parquet-ready DataFrame。

    处理:
        1. 按 Segment 时间裁剪（source_start_ns..source_end_ns）
        2. 保留 source_timestamp_ns
        3. 转换为 Segment 相对 timestamp_ns
        4. 字段名保持不变（已在 Reader 中标准化为 {joint}_{field}）

    Args:
        stream: TimeSeriesStream 对象。
        source_start_ns: Segment 源起始时间。
        source_end_ns: Segment 源结束时间。

    Returns:
        规范化后的 DataFrame，列:
          - timestamp_ns (相对)
          - source_timestamp_ns (原始)
          - {joint_name}_{field_name}... (各数据列)
    """
    timestamps = np.array(stream.timestamps_ns, dtype=np.int64)
    rows = np.asarray(stream.rows, dtype=np.float64)
    fields = stream.fields

    if len(timestamps) == 0 or rows.size == 0:
        raise ValueError(f"TimeSeriesStream {stream.stream_id} 为空")

    # ---- 1. 裁剪 ----
    mask = (timestamps >= source_start_ns) & (timestamps <= source_end_ns)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(
            f"TimeSeriesStream {stream.stream_id} "
            f"在 [{source_start_ns}, {source_end_ns}] 范围内无数据"
        )

    clipped_ts = timestamps[indices]
    clipped_rows = rows[indices, :]

    # ---- 2. 构建 DataFrame ----
    data: dict[str, Any] = {
        "source_timestamp_ns": clipped_ts,
        "timestamp_ns": clipped_ts - source_start_ns,
    }

    for col_idx, field in enumerate(fields):
        if col_idx < clipped_rows.shape[1]:
            col_name = field["name"]
            data[col_name] = clipped_rows[:, col_idx].astype(np.float32)

    df = pd.DataFrame(data)

    # 列顺序: timestamp_ns, source_timestamp_ns, field columns...
    field_cols = [f["name"] for f in fields if f["name"] in df.columns]
    ordered_cols = ["timestamp_ns", "source_timestamp_ns"] + field_cols
    df = df[ordered_cols]

    return df


def write_time_series(
    df: pd.DataFrame,
    output_dir: str,
    stream_id: str,
) -> str:
    """写出规范化 TimeSeries 为 Parquet。

    Args:
        df: normalize_time_series() 返回的 DataFrame。
        output_dir: Prepared Segment 根目录。
        stream_id: 流标识，文件名生成为 {stream_id}.parquet。

    Returns:
        输出文件路径。
    """
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"{stream_id}.parquet"
    df.to_parquet(str(output_path), index=False)
    return str(output_path)


__all__ = ["normalize_time_series", "write_time_series"]
