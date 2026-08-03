"""手部相关质量指标。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def compute_hand_presence_ratio(
    tracks: Iterable[Any], total_frames: int | None = None
) -> float:
    """计算至少检测到一只手的帧数占总解码帧数的比例。

    ``tracks`` 可以是带 ``output_frame_index`` 属性/键的观测，也可以是与
    视频逐帧对齐的布尔序列。观测列表必须显式传入 ``total_frames``，避免
    把尾部无手帧静默漏出分母。
    """
    items = list(tracks)
    if items and all(isinstance(item, bool) for item in items):
        denominator = len(items) if total_frames is None else total_frames
        if denominator < len(items):
            raise ValueError("total_frames 不能小于布尔序列长度")
        return sum(items) / denominator if denominator else 0.0
    if total_frames is None:
        raise ValueError("观测列表计算 hand_presence_ratio 时必须提供 total_frames")
    if total_frames < 0:
        raise ValueError("total_frames 不能为负数")
    frame_indices: set[int] = set()
    for item in items:
        if isinstance(item, dict):
            value = item.get("output_frame_index")
        else:
            value = getattr(item, "output_frame_index", None)
        if value is None:
            raise TypeError("手部观测必须包含 output_frame_index")
        index = int(value)
        if not 0 <= index < total_frames:
            raise ValueError(f"output_frame_index 超出视频范围: {index}")
        frame_indices.add(index)
    return len(frame_indices) / total_frames if total_frames else 0.0
