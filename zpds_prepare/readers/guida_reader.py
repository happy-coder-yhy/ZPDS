"""
读取墨现 (Guida) 数据集的所有原始文件。

复用已有的 reader.py，提供更结构化的接口。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from zpds_prepare.readers.session_model import DepthStream


def read_raw_meta(dataset_path: str) -> dict[str, Any]:
    """读取并返回未经扁平化的 Guida ``meta.json``。"""
    meta_path = Path(dataset_path) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise TypeError(f"meta.json 顶层必须是对象: {meta_path}")
    return meta


def read_meta(dataset_path: str) -> dict[str, Any]:
    """读取 meta.json，返回扁平化的元数据字典。"""
    meta = read_raw_meta(dataset_path)
    depth = meta.get("streams", {}).get("depth", {})

    return {
        "device": meta["device"]["name"],
        "fps": meta["streams"]["color"]["fps"],
        "frame_count": meta["recording_stats"]["total_frames"],
        "width": meta["streams"]["color"]["width"],
        "height": meta["streams"]["color"]["height"],
        "dropped_frames": meta["recording_stats"]["dropped_frames"],
        "imu_sample_rate": meta["imu"]["sample_rate_hz"],
        "depth": depth,
    }


def read_index_frames(dataset_path: str) -> list[dict]:
    """读取 index.jsonl 中所有 type=frame 的行。

    Returns:
        [{seq, timestamp_ns, type, ...}, ...]  按 seq 排序
    """
    index_path = Path(dataset_path) / "index.jsonl"
    frames = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "frame":
                frames.append(item)
    frames.sort(key=lambda f: f["seq"])
    return frames


def read_index_timestamps(dataset_path: str) -> list[int]:
    """读取 index.jsonl，只返回 type=frame 的纳秒时间戳列表（已排序）。"""
    frames = read_index_frames(dataset_path)
    return [f["timestamp_ns"] for f in frames]


def read_imu(dataset_path: str, imu_filename: str = "imu_000000.csv") -> pd.DataFrame:
    """读取 IMU CSV 文件。

    Returns:
        pd.DataFrame，列: timestamp_ns, ax, ay, az, gx, gy, gz
    """
    imu_path = Path(dataset_path) / "imu" / imu_filename
    return pd.read_csv(imu_path)


def get_color_mkv(dataset_path: str) -> str:
    """获取 RGB 原始 MKV 路径。"""
    return str(Path(dataset_path) / "color_000000.mkv")


def _as_path_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def discover_depth_sources(
    dataset_path: str,
    depth_meta: dict[str, Any] | None = None,
) -> tuple[str, list[Path]]:
    """发现 Guida 原始深度资产。

    优先使用 ``meta.json`` 中声明的路径；旧数据未声明路径时，兼容
    ``depth_*.mkv``、``depth_*.mp4`` 和 ``depth/*.png``。

    Returns:
        ``(source_kind, files)``，``source_kind`` 为 ``video``、
        ``image_sequence`` 或 ``missing``。
    """
    root = Path(dataset_path)
    depth_meta = depth_meta or {}

    declared: list[str] = []
    for key in ("uri", "path", "file", "filename", "files"):
        declared.extend(_as_path_list(depth_meta.get(key)))

    declared_paths = [
        path if path.is_absolute() else root / path
        for path in (Path(item) for item in declared)
    ]
    for path in declared_paths:
        if path.is_dir():
            images = sorted(
                item
                for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in {".png", ".tif", ".tiff"}
            )
            if images:
                return "image_sequence", images
        elif path.is_file():
            if path.suffix.lower() in {".png", ".tif", ".tiff"}:
                siblings = sorted(
                    item
                    for item in path.parent.iterdir()
                    if item.is_file()
                    and item.suffix.lower() in {".png", ".tif", ".tiff"}
                )
                return "image_sequence", siblings
            return "video", [path]

    video_files: list[Path] = []
    for pattern in ("depth_*.mkv", "depth_*.mp4", "depth/*.mkv", "depth/*.mp4"):
        video_files.extend(path for path in root.glob(pattern) if path.is_file())
    if video_files:
        return "video", sorted(set(video_files))

    fallback_images: list[Path] = []
    for pattern in ("depth/*.png", "depth/*.tif", "depth/*.tiff", "depth_*.png"):
        fallback_images.extend(path for path in root.glob(pattern) if path.is_file())
    if fallback_images:
        return "image_sequence", sorted(set(fallback_images))

    return "missing", []


def _normalize_depth_unit(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = value.strip().lower()
    aliases = {
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
    }
    return aliases.get(normalized, normalized)


def _normalize_depth_dtype(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = value.strip().lower()
    aliases = {
        "y16": "uint16",
        "z16": "uint16",
        "gray16": "uint16",
        "gray16le": "uint16",
        "uint16_t": "uint16",
    }
    return aliases.get(normalized, normalized)


def read_depth_stream(
    dataset_path: str,
    index_frames: list[dict[str, Any]] | None = None,
    *,
    required: bool = False,
) -> DepthStream | None:
    """构建 Guida 深度流描述，不在 Reader 阶段解码整段深度。

    ``index.jsonl.timestamp_ns`` 是权威时间轴。深度文件不存在时默认返回
    ``None``，调用方可以通过 ``required=True`` 将其提升为明确错误。
    """
    from zpds_prepare.readers.session_model import DepthStream

    raw_meta = read_raw_meta(dataset_path)
    depth_meta = raw_meta.get("streams", {}).get("depth", {})
    if not isinstance(depth_meta, dict):
        raise TypeError("meta.json 中 streams.depth 必须是对象")

    source_kind, source_files = discover_depth_sources(dataset_path, depth_meta)
    if not source_files:
        if required:
            raise FileNotFoundError(
                "Guida 深度已启用但未找到原始深度资产；"
                "请在 meta.json 的 streams.depth 中声明 path/uri，"
                "或提供 depth_*.mkv / depth/*.png"
            )
        return None

    frames = index_frames if index_frames is not None else read_index_frames(dataset_path)
    if not frames:
        raise ValueError("index.jsonl 中没有 type=frame 的权威时间戳")

    timestamps_ns = [int(frame["timestamp_ns"]) for frame in frames]
    declared_count = depth_meta.get("frame_count")
    if declared_count is None:
        declared_count = raw_meta.get("recording_stats", {}).get(
            "total_frames", len(timestamps_ns)
        )
    actual_source_count = len(source_files) if source_kind == "image_sequence" else None

    unit = _normalize_depth_unit(
        depth_meta.get("unit")
        or depth_meta.get("depth_unit")
        or depth_meta.get("measurement_unit")
    )
    dtype = _normalize_depth_dtype(
        depth_meta.get("dtype")
        or depth_meta.get("data_type")
        or depth_meta.get("pixel_type")
        or depth_meta.get("format")
    )
    invalid_value = depth_meta.get(
        "invalid_value",
        depth_meta.get("invalid_depth_value"),
    )

    return DepthStream(
        stream_id="ego_depth",
        timestamps_ns=timestamps_ns,
        index_frames=frames,
        source_files=source_files,
        source_kind=source_kind,
        fps=float(depth_meta.get("fps") or raw_meta["streams"]["color"]["fps"]),
        width=int(depth_meta.get("width", 0)),
        height=int(depth_meta.get("height", 0)),
        frame_count=int(declared_count),
        dtype=dtype,
        unit=unit,
        invalid_value=invalid_value,
        metadata={
            "unit_status": "verified_from_source_metadata"
            if unit != "unknown"
            else "unverified",
            "scale_to_meters": depth_meta.get("scale_to_meters"),
            "declared_frame_count": int(declared_count),
            "source_file_count": len(source_files),
            "source_frame_count": actual_source_count,
        },
    )


def get_session_id(dataset_path: str) -> str:
    """从数据集路径推导 session_id。"""
    folder = Path(dataset_path).name
    return f"guida_{folder}"


def read_session(
    dataset_path: str,
    *,
    include_depth: bool = True,
    require_depth: bool = False,
):
    """统一读取 Session 全部流数据。

    Returns:
        Session 对象，包含:
          - video_streams: {"ego_rgb": VideoStream}
          - depth_streams: {"ego_depth": DepthStream}（存在原始深度时）
          - imu_streams:  {"ego_imu": ImuStream}
    """
    from zpds_prepare.readers.session_model import ImuStream, Session, VideoStream

    meta = read_meta(dataset_path)
    index_frames = read_index_frames(dataset_path)
    timestamps_ns = [f["timestamp_ns"] for f in index_frames]
    video_path = get_color_mkv(dataset_path)
    imu_df = read_imu(dataset_path)
    depth_stream = (
        read_depth_stream(
            dataset_path,
            index_frames=index_frames,
            required=require_depth,
        )
        if include_depth
        else None
    )

    video_stream = VideoStream(
        stream_id="ego_rgb",
        timestamps_ns=timestamps_ns,
        index_frames=index_frames,
        video_path=video_path,
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
        frame_count=meta["frame_count"],
    )

    imu_stream = ImuStream(
        stream_id="ego_imu",
        dataframe=imu_df,
        sample_rate_hz=meta["imu_sample_rate"],
    )

    return Session(
        session_id=get_session_id(dataset_path),
        source_path=dataset_path,
        meta=meta,
        video_streams={"ego_rgb": video_stream},
        depth_streams=(
            {depth_stream.stream_id: depth_stream}
            if depth_stream is not None
            else {}
        ),
        imu_streams={"ego_imu": imu_stream},
    )
