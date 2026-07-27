"""
A2D 图像序列 → CFR MP4 转码器。

将触发式相机捕获的 JPEG 序列转换为恒定帧率 H.264 MP4 视频。
使用 cv2.VideoWriter（底层 FFmpeg），通过 imdecode 绕过中文路径问题。

用法:
    from segment.a2d_video_transcoder import transcode_image_sequence

    result = transcode_image_sequence(
        index_frames=session.video_streams["head_rgb"].index_frames,
        output_mp4="output/data/head_rgb.mp4",
        source_start_ns=span_start,
        source_end_ns=span_end,
        target_fps=30.0,
        width=640, height=480,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def transcode_image_sequence(
    index_frames: list[dict[str, Any]],
    output_mp4: str,
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> dict:
    """将图像序列转码为 CFR H.264 MP4。

    流程:
        1. 筛选 source_start_ns..source_end_ns 范围内的源帧
        2. 按 target_fps 生成 CFR 输出时间轴
        3. 每个输出帧通过最近邻映射选取源帧
        4. 使用 cv2.VideoWriter 编码为 MP4

    注意:
        源 JPEG 通过 Python open() + cv2.imdecode() 读取，避免
        cv2.imread() 的中文路径问题。输出路径不含中文。

    Args:
        index_frames: A2D VideoStream.index_frames 列表。
        source_start_ns: Segment 源起始时间。
        source_end_ns: Segment 源结束时间。
        output_mp4: 输出 MP4 文件路径（不含中文）。
        target_fps: 目标恒定帧率。
        width: 帧宽。
        height: 帧高。

    Returns:
        {
            "output_frames": int,
            "output_fps": float,
            "width": int,
            "height": int,
            "codec": str,
            "output_path": str,
        }
    """
    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. 筛选 Span 内的帧 ----
    span_frames = [
        f for f in index_frames
        if f.get("source_timestamp_ns") is not None
        and source_start_ns <= f["source_timestamp_ns"] <= source_end_ns
    ]
    span_timestamps = np.array(
        [f["source_timestamp_ns"] for f in span_frames], dtype=np.int64
    )

    if len(span_frames) == 0:
        raise ValueError("Span 内没有有效帧")

    # ---- 2. 计算 CFR 输出帧数 ----
    frame_interval_ns = int(1_000_000_000 / target_fps)
    segment_duration_ns = source_end_ns - source_start_ns
    output_count = max(1, int(segment_duration_ns / frame_interval_ns))

    # ---- 3. 预加载所有源帧 ----
    # Python open() + cv2.imdecode() 绕过中文路径问题
    frame_cache: dict[int, np.ndarray] = {}
    loaded_count = 0

    for idx, sf in enumerate(span_frames):
        source_path = sf["source_path"]
        try:
            with open(source_path, "rb") as fh:
                jpeg_bytes = fh.read()
            img = cv2.imdecode(
                np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if img is not None:
                if img.shape[1] != width or img.shape[0] != height:
                    img = cv2.resize(img, (width, height))
                frame_cache[idx] = img
                loaded_count += 1
        except Exception:
            pass

    if not frame_cache:
        raise ValueError("无法加载任何源帧")

    # ---- 4. 创建 VideoWriter ----
    # 按优先级尝试编码器: mp4v（内置）→ avc1 → H264（需 OpenH264 DLL）
    codec = None
    writer = None
    for codec in ("mp4v", "avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
        if writer.isOpened():
            break

    if writer is None or not writer.isOpened():
        raise RuntimeError(f"无法创建 VideoWriter: {output_mp4}")

    # ---- 5. 逐个输出帧 → 最近邻 → 写入 ----
    total_output = 0
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    for out_idx in range(output_count):
        target_time = source_start_ns + out_idx * frame_interval_ns
        nearest_idx = int(np.argmin(np.abs(span_timestamps - target_time)))

        frame = _get_nearest_cached(frame_cache, nearest_idx)
        if frame is None:
            frame = black_frame

        writer.write(frame)
        total_output += 1

    writer.release()

    return {
        "output_frames": total_output,
        "output_fps": target_fps,
        "width": width,
        "height": height,
        "codec": "h264" if codec == "avc1" else codec,
        "output_path": str(output_path),
    }


def _get_nearest_cached(
    cache: dict[int, np.ndarray],
    target_idx: int,
) -> np.ndarray | None:
    """从缓存中取 target_idx 或其最近可用的帧。"""
    if target_idx in cache:
        return cache[target_idx]

    cached_indices = sorted(cache.keys())
    if not cached_indices:
        return None

    nearest = min(cached_indices, key=lambda i: abs(i - target_idx))
    return cache.get(nearest)


def generate_image_sample_map(
    index_frames: list[dict[str, Any]],
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
) -> "pd.DataFrame":
    """为图像序列 CFR 输出生成最近邻 sample_map.parquet。

    Args:
        index_frames: A2D VideoStream.index_frames 列表。
        source_start_ns: Segment 源起始时间。
        source_end_ns: Segment 源结束时间。
        target_fps: 目标恒定帧率。

    Returns:
        DataFrame，列:
          - output_frame_index
          - output_timestamp_ns
          - source_frame_index (原始 HDF5 row index)
          - source_timestamp_ns
          - source_path
          - mapping_method
          - time_error_ns
    """
    import pandas as pd

    # 筛选有效帧
    span_frames = [
        f for f in index_frames
        if f.get("source_timestamp_ns") is not None
        and source_start_ns <= f["source_timestamp_ns"] <= source_end_ns
    ]
    span_timestamps = np.array(
        [f["source_timestamp_ns"] for f in span_frames], dtype=np.int64
    )

    if len(span_frames) == 0:
        raise ValueError("Span 内没有有效帧")

    frame_interval_ns = int(1_000_000_000 / target_fps)
    segment_duration_ns = source_end_ns - source_start_ns
    output_count = max(1, int(segment_duration_ns / frame_interval_ns))

    rows = []
    for out_idx in range(output_count):
        target_time = source_start_ns + out_idx * frame_interval_ns
        nearest_idx = int(np.argmin(np.abs(span_timestamps - target_time)))
        source_row = span_frames[nearest_idx]
        source_ts = int(source_row["source_timestamp_ns"])

        rows.append({
            "output_frame_index": out_idx,
            "output_timestamp_ns": out_idx * frame_interval_ns,
            "source_frame_index": int(source_row.get("frame_index", nearest_idx)),
            "source_timestamp_ns": source_ts,
            "source_path": source_row.get("source_path", ""),
            "mapping_method": "nearest",
            "time_error_ns": int(source_ts - target_time),
        })

    return pd.DataFrame(rows)


def write_image_sample_map(
    sample_map: "pd.DataFrame",
    output_dir: str,
    stream_id: str,
) -> str:
    """写出图像序列 sample_map 为 Parquet。

    Args:
        sample_map: generate_image_sample_map() 返回的 DataFrame。
        output_dir: Prepared Segment 根目录。
        stream_id: 流标识。

    Returns:
        输出文件路径。
    """
    maps_dir = Path(output_dir) / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    output_path = maps_dir / f"{stream_id}_sample_map.parquet"
    sample_map.to_parquet(str(output_path), index=False)
    return str(output_path)


__all__ = [
    "transcode_image_sequence",
    "generate_image_sample_map",
    "write_image_sample_map",
]
