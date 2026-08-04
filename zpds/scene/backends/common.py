"""Stage A 检测器共享的帧校验、时间换算和候选聚合工具。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np


def validate_frames(frames: Sequence[np.ndarray], *, fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须是大于 0 的有限数值")
    expected_shape: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frames[{index}] 必须是 numpy.ndarray")
        if frame.size == 0 or frame.ndim not in {2, 3}:
            raise ValueError(f"frames[{index}] 必须是非空灰度或 BGR 图像")
        if frame.ndim == 3 and frame.shape[2] not in {1, 3, 4}:
            raise ValueError(f"frames[{index}] 的通道数必须是 1、3 或 4")
        shape = (frame.shape[0], frame.shape[1])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("所有输入帧必须具有相同宽高")


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    if frame.shape[2] == 1:
        return frame[:, :, 0].astype(np.uint8, copy=False)
    conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(frame, conversion)


def to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def timestamp_ns(frame_index: int, *, fps: float, start_timestamp_ns: int) -> int:
    if isinstance(start_timestamp_ns, bool) or start_timestamp_ns < 0:
        raise ValueError("start_timestamp_ns 必须是非负整数")
    return start_timestamp_ns + round(frame_index * 1_000_000_000 / fps)


def select_peak_indices(
    indices: Sequence[int],
    scores: Sequence[float],
    *,
    max_gap: int = 1,
) -> list[int]:
    """将相邻候选折叠为得分最高的单一帧。"""

    if not indices:
        return []
    sorted_indices = sorted(set(indices))
    groups: list[list[int]] = [[sorted_indices[0]]]
    for index in sorted_indices[1:]:
        if index - groups[-1][-1] <= max_gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [max(group, key=lambda item: (scores[item], -item)) for group in groups]


def finite_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


__all__ = [
    "finite_unit",
    "select_peak_indices",
    "timestamp_ns",
    "to_bgr",
    "to_gray",
    "validate_frames",
]
