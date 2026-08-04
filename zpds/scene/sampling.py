"""场景代表帧索引与帧提取工具。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from zpds.scene.schemas import SceneProposal


def representative_frame_indices(
    scene: SceneProposal,
    *,
    fps: float,
    frame_count: int,
    segment_start_ns: int = 0,
) -> tuple[int, int, int]:
    """返回 scene 首、中、尾三帧；极短 scene 允许索引重复。"""

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须是大于 0 的有限数值")
    if isinstance(frame_count, bool) or frame_count <= 0:
        raise ValueError("frame_count 必须是正整数")
    if isinstance(segment_start_ns, bool) or segment_start_ns < 0:
        raise ValueError("segment_start_ns 必须是非负整数")
    segment_end_ns = segment_start_ns + round(frame_count * 1_000_000_000 / fps)
    if scene.start_ns < segment_start_ns or scene.end_ns > segment_end_ns:
        raise ValueError("scene 超出视频时间范围")

    first = math.floor((scene.start_ns - segment_start_ns) * fps / 1_000_000_000)
    last_exclusive = math.ceil(
        (scene.end_ns - segment_start_ns) * fps / 1_000_000_000
    )
    first = max(0, min(frame_count - 1, first))
    last = max(first, min(frame_count - 1, last_exclusive - 1))
    middle = (first + last) // 2
    return first, middle, last


def extract_representative_frames(
    frames: Sequence[np.ndarray],
    scene: SceneProposal,
    *,
    fps: float,
    segment_start_ns: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = representative_frame_indices(
        scene,
        fps=fps,
        frame_count=len(frames),
        segment_start_ns=segment_start_ns,
    )
    return tuple(frames[index] for index in indices)  # type: ignore[return-value]


__all__ = ["extract_representative_frames", "representative_frame_indices"]
