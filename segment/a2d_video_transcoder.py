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

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def transcode_image_sequence(
    index_frames: list[dict[str, Any]],
    output_mp4: str,
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
    width: int = 640,
    height: int = 480,
    frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
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
        frame_transform: 可选逐帧确定性变换，在缩放到输出分辨率后应用。

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

    # ---- 3. 创建 VideoWriter ----
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

    # ---- 4. 逐个输出帧 → 最近邻 → 写入 ----
    total_output = 0
    loaded_source_frame = False
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    # 固定容量缓存：避免将长 Episode 的全部 JPEG 解码进内存。
    frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    for out_idx in range(output_count):
        target_time = source_start_ns + out_idx * frame_interval_ns
        nearest_idx = int(np.argmin(np.abs(span_timestamps - target_time)))

        frame = _get_nearest_cached(
            frame_cache,
            span_frames,
            nearest_idx,
            width,
            height,
        )
        if frame is None:
            frame = black_frame
        else:
            loaded_source_frame = True
            if frame_transform is not None:
                frame = frame_transform(frame)

        writer.write(frame)
        total_output += 1

    writer.release()
    if not loaded_source_frame:
        output_path.unlink(missing_ok=True)
        raise ValueError("无法加载任何源帧")

    return {
        "output_frames": total_output,
        "output_fps": target_fps,
        "width": width,
        "height": height,
        "codec": "h264" if codec == "avc1" else codec,
        "output_path": str(output_path),
    }


def _get_nearest_cached(
    cache: OrderedDict[int, np.ndarray],
    frames: list[dict[str, Any]],
    target_idx: int,
    width: int,
    height: int,
    max_cache_size: int = 32,
) -> np.ndarray | None:
    """按序号就近读取帧，并保留固定容量的已解码缓存。"""
    if target_idx in cache:
        cache.move_to_end(target_idx)
        return cache[target_idx]

    for offset in range(len(frames)):
        candidate_indices = [target_idx - offset]
        if offset:
            candidate_indices.append(target_idx + offset)
        for frame_idx in candidate_indices:
            if frame_idx < 0 or frame_idx >= len(frames):
                continue
            if frame_idx in cache:
                cache.move_to_end(frame_idx)
                return cache[frame_idx]
            frame = _load_image(
                frames[frame_idx].get("source_path", ""), width, height
            )
            if frame is None:
                continue
            cache[frame_idx] = frame
            cache.move_to_end(frame_idx)
            if len(cache) > max_cache_size:
                cache.popitem(last=False)
            return frame
    return None


def _load_image(source_path: str, width: int, height: int) -> np.ndarray | None:
    """读取单张源 JPEG，并在必要时缩放到输出尺寸。"""
    try:
        with open(source_path, "rb") as fh:
            jpeg_bytes = fh.read()
    except OSError:
        return None
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[1] != width or image.shape[0] != height:
        return cv2.resize(image, (width, height))
    return image


def generate_image_sample_map(
    index_frames: list[dict[str, Any]],
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
) -> pd.DataFrame:
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
    sample_map: pd.DataFrame,
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
    "generate_image_sample_map",
    "transcode_image_sequence",
    "write_image_sample_map",
]
