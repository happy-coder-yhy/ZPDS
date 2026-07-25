"""
A2D 深度图像序列处理 — 裁剪并拷贝 PNG，保持 uint16 精度。

不转码为 MP4（会损失 16-bit 精度）。
V2 阶段使用裁剪拷贝方案（后续可升级为 zarr）。

用法:
    from segment.a2d_depth_copy import copy_depth_sequence

    result = copy_depth_sequence(
        index_frames=depth_stream.index_frames,
        output_dir="seg_000001/data/depth/head_depth/",
        source_start_ns=...,
        source_end_ns=...,
    )
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def copy_depth_sequence(
    index_frames: list[dict[str, Any]],
    output_dir: str,
    source_start_ns: int,
    source_end_ns: int,
) -> dict:
    """裁剪并拷贝深度 PNG 序列到 Segment 目录。

    仅拷贝时间范围内的帧，保持原始 uint16 PNG 格式。

    Args:
        index_frames: A2D 深度 VideoStream.index_frames 列表。
        output_dir: 输出目录（如 seg_000001/data/depth/head_depth/）。
        source_start_ns: Segment 源起始时间。
        source_end_ns: Segment 源结束时间。

    Returns:
        {
            "copied_frames": int,
            "width": int,
            "height": int,
            "dtype": str,
            "output_dir": str,
        }
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 筛选 Span 内的帧 ----
    span_frames = [
        f for f in index_frames
        if f.get("source_timestamp_ns") is not None
        and source_start_ns <= f["source_timestamp_ns"] <= source_end_ns
    ]

    if not span_frames:
        raise ValueError("Span 内没有深度帧")

    # ---- 2. 探测第一帧属性 ----
    first_path = span_frames[0]["source_path"]
    with open(first_path, "rb") as fh:
        png_bytes = fh.read()
    sample = cv2.imdecode(
        np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED
    )
    if sample is None:
        raise ValueError(f"无法解码深度帧: {first_path}")

    img_h, img_w = sample.shape[:2]
    img_dtype = str(sample.dtype)

    # ---- 3. 拷贝帧 ----
    copied = 0
    for seq, sf in enumerate(span_frames):
        src = sf["source_path"]
        # 保持原始文件名（包含 frame_index 信息）
        frame_idx = sf.get("frame_index", seq)
        dst = out_dir / f"{frame_idx:06d}.png"

        try:
            shutil.copy2(src, dst)
            copied += 1
        except OSError:
            pass

    return {
        "copied_frames": copied,
        "total_in_span": len(span_frames),
        "width": img_w,
        "height": img_h,
        "dtype": img_dtype,
        "output_dir": str(out_dir),
    }


def generate_depth_sample_map(
    index_frames: list[dict[str, Any]],
    source_start_ns: int,
    source_end_ns: int,
) -> pd.DataFrame:
    """为深度序列生成 sparse sample_map。

    与 RGB sample_map 不同，深度不打帧率——一帧即一条记录。
    每条记录包含源帧索引、时间戳、源路径。

    Returns:
        DataFrame，列:
          - output_seq: 输出序号
          - source_frame_index
          - source_timestamp_ns
          - source_path
          - mapping_method (= "direct_copy")
    """
    span_frames = [
        f for f in index_frames
        if f.get("source_timestamp_ns") is not None
        and source_start_ns <= f["source_timestamp_ns"] <= source_end_ns
    ]

    rows = []
    for seq, sf in enumerate(span_frames):
        rows.append({
            "output_seq": seq,
            "source_frame_index": int(sf.get("frame_index", seq)),
            "source_timestamp_ns": int(sf["source_timestamp_ns"]),
            "source_path": sf.get("source_path", ""),
            "mapping_method": "direct_copy",
        })

    return pd.DataFrame(rows)


def write_depth_sample_map(
    sample_map: pd.DataFrame,
    output_dir: str,
    stream_id: str,
) -> str:
    """写出深度 sample_map 为 Parquet。"""
    maps_dir = Path(output_dir) / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    output_path = maps_dir / f"{stream_id}_sample_map.parquet"
    sample_map.to_parquet(str(output_path), index=False)
    return str(output_path)


def probe_depth_properties(
    index_frames: list[dict[str, Any]],
    source_start_ns: int,
    source_end_ns: int,
    sample_size: int = 50,
) -> dict:
    """探测深度图像属性：dtype, 零值比例, 无效值比例。

    Returns:
        {
            "dtype": "uint16",
            "width": int, "height": int,
            "zero_ratio": float,
            "max_value": int,
            "invalid_ratio": float,
            "samples_checked": int,
        }
    """
    span_frames = [
        f for f in index_frames
        if f.get("source_timestamp_ns") is not None
        and source_start_ns <= f["source_timestamp_ns"] <= source_end_ns
    ]

    if not span_frames:
        return {"error": "Span 内无深度帧"}

    # 抽样检查
    step = max(1, len(span_frames) // sample_size)
    sample_frames = span_frames[::step][:sample_size]

    widths = set()
    heights = set()
    zero_counts = 0
    total_pixels = 0
    max_val = 0
    dtypes = set()

    for sf in sample_frames:
        src = sf["source_path"]
        try:
            with open(src, "rb") as fh:
                png_bytes = fh.read()
            img = cv2.imdecode(
                np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED
            )
            if img is None:
                continue
            dtypes.add(str(img.dtype))
            widths.add(img.shape[1])
            heights.add(img.shape[0])
            total_pixels += img.size
            zero_counts += int((img == 0).sum())
            m = int(img.max()) if img.size > 0 else 0
            if m > max_val:
                max_val = m
        except Exception:
            pass

    return {
        "dtype": sorted(dtypes)[0] if len(dtypes) == 1 else str(dtypes),
        "width": sorted(widths)[0] if len(widths) == 1 else list(widths),
        "height": sorted(heights)[0] if len(heights) == 1 else list(heights),
        "zero_ratio": round(zero_counts / total_pixels, 6) if total_pixels > 0 else None,
        "max_value": max_val,
        "invalid_ratio": 0.0,  # PNG 解码无 NaN/Inf
        "samples_checked": len(sample_frames),
    }


__all__ = [
    "copy_depth_sequence",
    "generate_depth_sample_map",
    "write_depth_sample_map",
    "probe_depth_properties",
]
