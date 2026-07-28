"""深度流的无损 Prepared 写出。

深度按原始频率保存为 uint16 PNG 序列，不进入 RGB 转码器，也不为了和
RGB 对齐而强制重采样。支持图片序列、无损视频以及 MCAP 内嵌 PNG；
逐帧对应关系和源时钟记录在 Parquet sample map 中。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from zpds_prepare.readers.session_model import DepthStream


def build_depth_sample_map(
    stream: DepthStream,
    source_start_ns: int,
    source_end_ns: int,
) -> pd.DataFrame:
    """生成深度输出帧到原始帧的一对一映射。"""
    if source_end_ns <= source_start_ns:
        raise ValueError("深度 Span 的结束时间必须晚于开始时间")
    if stream.frame_count != len(stream.index_frames):
        raise ValueError(
            "深度声明帧数与索引记录不一致："
            f"depth={stream.frame_count}, index={len(stream.index_frames)}"
        )
    if (
        stream.source_kind == "image_sequence"
        and len(stream.source_files) != len(stream.index_frames)
    ):
        raise ValueError(
            "深度图片数量与索引记录不一致："
            f"images={len(stream.source_files)}, index={len(stream.index_frames)}"
        )

    file_local_positions: dict[int, int] = {}
    rows: list[dict[str, Any]] = []

    for ordered_position, frame in enumerate(stream.index_frames):
        timestamp_ns = int(frame["timestamp_ns"])
        segment_index = int(frame.get("segment", 0))
        if len(stream.source_files) == 1:
            source_asset_index = 0
            source_frame_position = int(
                frame.get("source_frame_position", ordered_position)
            )
        else:
            if segment_index < 0 or segment_index >= len(stream.source_files):
                raise ValueError(
                    f"index.jsonl segment={segment_index} 超出深度文件数量 "
                    f"{len(stream.source_files)}"
                )
            source_asset_index = segment_index
            source_frame_position = file_local_positions.get(segment_index, 0)
            file_local_positions[segment_index] = source_frame_position + 1

        if timestamp_ns < source_start_ns or timestamp_ns > source_end_ns:
            continue

        if stream.source_kind == "image_sequence":
            source_asset_index = ordered_position
            source_frame_position = 0
            if source_asset_index >= len(stream.source_files):
                raise ValueError(
                    "深度图片数量少于 index.jsonl 帧数："
                    f"images={len(stream.source_files)}, index_position={ordered_position}"
                )

        source_path = stream.source_files[source_asset_index]
        row = {
            "output_frame_index": len(rows),
            "output_timestamp_ns": timestamp_ns - source_start_ns,
            "output_file": f"{len(rows):08d}.png",
            "source_frame_index": int(frame.get("seq", ordered_position)),
            "source_frame_position": source_frame_position,
            "source_timestamp_ns": timestamp_ns,
            "source_asset_index": source_asset_index,
            "source_file": source_path.name,
            "mapping_method": "identity",
            "time_error_ns": 0,
        }
        if frame.get("log_time_ns") is not None:
            row["source_log_time_ns"] = int(frame["log_time_ns"])
        if frame.get("publish_time_ns") is not None:
            row["source_publish_time_ns"] = int(frame["publish_time_ns"])
        rows.append(row)

    if not rows:
        raise ValueError("深度 Span 内没有源帧")

    return pd.DataFrame(rows)


def _decode_video_range(
    source: Path,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Guida 深度 MKV 需要外部 ffmpeg；请先安装 ffmpeg 并确认命令可用"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "decoded_%010d.png"
    select_filter = f"select=between(n\\,{start_frame}\\,{end_frame})"
    command = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        select_filter,
        "-fps_mode",
        "passthrough",
        "-start_number",
        str(start_frame),
        "-c:v",
        "png",
        str(output_pattern),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"ffmpeg 解码 Guida 深度失败: {source}; "
            f"exit={result.returncode}; stderr={stderr[-1000:]}"
        )


def _source_png_for_row(
    stream: DepthStream,
    row: Any,
    decoded_dirs: dict[int, Path],
) -> Path:
    asset_index = int(row.source_asset_index)
    if stream.source_kind == "image_sequence":
        return stream.source_files[asset_index]
    decoded_dir = decoded_dirs[asset_index]
    return decoded_dir / f"decoded_{int(row.source_frame_position):010d}.png"


def _iter_mcap_compressed_images(
    stream: DepthStream,
    sample_map: pd.DataFrame,
):
    """按 MCAP 消息位置流式解码选中的内嵌 PNG，避免整段驻留内存。"""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    if len(stream.source_files) != 1:
        raise ValueError("MCAP 深度流必须且只能引用一个 MCAP 文件")
    topic = str(stream.metadata.get("topic", ""))
    if not topic:
        raise ValueError("MCAP 深度流缺少 metadata.topic")

    selected = {
        int(row.source_frame_position): row
        for row in sample_map.itertuples(index=False)
    }
    yielded = 0
    source_path = stream.source_files[0]
    with source_path.open("rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        source_position = 0
        for _schema, channel, _message, decoded in reader.iter_decoded_messages():
            if channel.topic != topic:
                continue
            row = selected.get(source_position)
            if row is not None:
                compression_format = str(
                    getattr(decoded, "format", "")
                ).lower()
                if compression_format and compression_format != "png":
                    raise ValueError(
                        "MCAP 深度消息必须是 PNG，"
                        f"实际 format={compression_format}"
                    )
                image = cv2.imdecode(
                    np.frombuffer(decoded.data, np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                if image is None:
                    raise ValueError(
                        f"MCAP 深度消息无法解码: position={source_position}"
                    )
                yielded += 1
                yield row, image
            source_position += 1

    if yielded != len(selected):
        raise ValueError(
            "MCAP 深度消息数量不足："
            f"expected={len(selected)}, decoded={yielded}"
        )


def _swap_directory(staging_dir: Path, final_dir: Path) -> None:
    """用已完整写好的 staging 目录替换派生输出目录。"""
    backup_dir = final_dir.with_name(f".{final_dir.name}.backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if final_dir.exists():
        final_dir.rename(backup_dir)
    try:
        staging_dir.rename(final_dir)
    except Exception:
        if backup_dir.exists() and not final_dir.exists():
            backup_dir.rename(final_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def write_depth_stream(
    stream: DepthStream,
    output_dir: str,
    source_start_ns: int,
    source_end_ns: int,
) -> dict[str, Any]:
    """按原始频率无损写出一个深度流及其 sample map。"""
    segment_dir = Path(output_dir)
    depth_parent = segment_dir / "data" / "depth"
    depth_parent.mkdir(parents=True, exist_ok=True)
    final_dir = depth_parent / stream.stream_id
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{stream.stream_id}.", dir=str(depth_parent))
    )
    sample_map = build_depth_sample_map(stream, source_start_ns, source_end_ns)

    maps_dir = segment_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    sample_map_path = maps_dir / f"{stream.stream_id}_sample_map.parquet"
    sample_map_staging = maps_dir / f".{stream.stream_id}_sample_map.parquet.tmp"

    decoded_dirs: dict[int, Path] = {}
    dtypes: set[str] = set()
    resolutions: set[tuple[int, int]] = set()
    zero_pixels = 0
    invalid_pixels = 0
    total_pixels = 0

    try:
        if stream.source_kind == "video":
            for asset_index, group in sample_map.groupby("source_asset_index"):
                index = int(asset_index)
                decoded_dir = staging_dir / f".decoded_{index}"
                start_frame = int(group["source_frame_position"].min())
                end_frame = int(group["source_frame_position"].max())
                _decode_video_range(
                    stream.source_files[index],
                    start_frame,
                    end_frame,
                    decoded_dir,
                )
                decoded_dirs[index] = decoded_dir
        elif stream.source_kind not in {
            "image_sequence",
            "mcap_compressed_image",
        }:
            raise ValueError(f"不支持的深度 source_kind: {stream.source_kind}")

        if stream.source_kind == "mcap_compressed_image":
            image_rows = _iter_mcap_compressed_images(stream, sample_map)
        else:
            def read_file_rows():
                for row in sample_map.itertuples(index=False):
                    source_png = _source_png_for_row(
                        stream,
                        row,
                        decoded_dirs,
                    )
                    if not source_png.is_file():
                        raise FileNotFoundError(
                            f"深度源帧不存在: {source_png} "
                            f"(source_frame_index={row.source_frame_index})"
                        )
                    image = cv2.imread(
                        str(source_png),
                        cv2.IMREAD_UNCHANGED,
                    )
                    if image is None:
                        raise ValueError(f"深度帧无法解码: {source_png}")
                    yield row, image

            image_rows = read_file_rows()

        for row, image in image_rows:
            if image.ndim != 2:
                raise ValueError(
                    f"深度帧必须是单通道，实际 shape={image.shape}"
                )

            actual_dtype = str(image.dtype)
            if stream.dtype != "unknown" and actual_dtype != stream.dtype:
                raise ValueError(
                    "深度 dtype 与 meta.json 不一致（或源声明冲突）: "
                    f"declared={stream.dtype}, actual={actual_dtype}"
                )

            dtypes.add(actual_dtype)
            resolutions.add((int(image.shape[1]), int(image.shape[0])))
            total_pixels += int(image.size)
            zero_pixels += int(np.count_nonzero(image == 0))
            if stream.invalid_value is not None:
                invalid_pixels += int(np.count_nonzero(image == stream.invalid_value))

            output_path = staging_dir / str(row.output_file)
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"无法写出深度 PNG: {output_path}")

        for decoded_dir in decoded_dirs.values():
            shutil.rmtree(decoded_dir, ignore_errors=True)

        if len(dtypes) != 1:
            raise ValueError(f"深度帧 dtype 不一致: {sorted(dtypes)}")
        if len(resolutions) != 1:
            raise ValueError(f"深度帧分辨率不一致: {sorted(resolutions)}")

        sample_map.to_parquet(sample_map_staging, index=False)
        _swap_directory(staging_dir, final_dir)
        os.replace(sample_map_staging, sample_map_path)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if sample_map_staging.exists():
            sample_map_staging.unlink()
        raise

    width, height = next(iter(resolutions))
    timestamps = sample_map["source_timestamp_ns"].to_numpy(dtype=np.int64)
    if len(timestamps) >= 2:
        median_interval_ns = float(np.median(np.diff(timestamps)))
        rate_hz = 1_000_000_000 / median_interval_ns if median_interval_ns > 0 else stream.fps
    else:
        rate_hz = stream.fps

    return {
        "stream_id": stream.stream_id,
        "uri": f"data/depth/{stream.stream_id}/",
        "format": "png_sequence",
        "encoding": "png",
        "dtype": next(iter(dtypes)),
        "unit": stream.unit,
        "unit_status": stream.metadata.get("unit_status", "unverified"),
        "invalid_value": stream.invalid_value,
        "width": width,
        "height": height,
        "frames": len(sample_map),
        "rate_hz": round(float(rate_hz), 6),
        "frame_id": stream.frame_id,
        "sample_map_uri": f"maps/{stream.stream_id}_sample_map.parquet",
        "source_asset_id": stream.metadata.get(
            "source_asset_id",
            "raw_depth_0",
        ),
        "operation": stream.metadata.get(
            "operation",
            "trim_decode_lossless_png",
        ),
        "zero_ratio": zero_pixels / total_pixels if total_pixels else None,
        "invalid_ratio": invalid_pixels / total_pixels if total_pixels else None,
    }


__all__ = ["build_depth_sample_map", "write_depth_stream"]
