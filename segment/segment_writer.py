"""
生成 segment.json — Prepared Segment 的核心控制文件。

streams 列表根据传入的 video_results 和 imu_results 动态生成，
文件名由各流的 stream_id 决定，不再硬编码。
"""

import json
import hashlib
from pathlib import Path


def sha256_hex(path: str) -> str:
    """计算文件 SHA-256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_segment_json(
    dataset_path: str,
    span: dict,
    video_results: list[dict] | None = None,
    imu_results: list[dict] | None = None,
    calibration_id: str = "calib_guida_001",
    revision: str = "r0001",
    segment_id: str = "seg_000001",
    session_id: str = "guida_session_001",
    quality_issues: list[dict] | None = None,
    source_assets: list[dict] | None = None,
    profile: str = "guida",
    depth_npz_path: str | None = None,
    calibrations: dict | None = None,
) -> dict:
    """构建 segment.json 内容。

    每个 video_result 应包含:
      - stream_id, width, height, output_fps, output_frames
      - sample_map_uri (相对于 segment 根目录)
      - frame_id (可选), role (可选, 默认 "observation")

    每个 imu_result 应包含:
      - stream_id, uri (相对于 segment 根目录), rows
    """
    data_dir = Path(dataset_path)
    index_path = data_dir / "index.jsonl"
    meta_path = data_dir / "meta.json"

    duration_ns = span["source_end_ns"] - span["source_start_ns"]

    # source_assets — 由调用方传入或按 guida 默认生成
    if source_assets is None:
        color_path = data_dir / "color_000000.mkv"
        imu_path = data_dir / "imu" / "imu_000000.csv"
        source_assets = [
            {
                "source_asset_id": "raw_color_0",
                "uri": "color_000000.mkv",
                "sha256": sha256_hex(str(color_path)) if color_path.exists() else "",
            },
            {
                "source_asset_id": "raw_index",
                "uri": "index.jsonl",
                "sha256": sha256_hex(str(index_path)) if index_path.exists() else "",
            },
            {
                "source_asset_id": "raw_imu_0",
                "uri": "imu/imu_000000.csv",
                "sha256": sha256_hex(str(imu_path)) if imu_path.exists() else "",
            },
            {
                "source_asset_id": "raw_meta",
                "uri": "meta.json",
                "sha256": sha256_hex(str(meta_path)) if meta_path.exists() else "",
            },
        ]

    # ---- 构建 streams 列表 ----
    streams: list[dict] = []

    # RGB 视频流 — 每个 video_result 生成一个 stream entry
    for vr in (video_results or []):
        stream_id = vr["stream_id"]
        streams.append({
            "stream_id": stream_id,
            "role": vr.get("role", "observation"),
            "modality": "rgb",
            "uri": f"data/{stream_id}.mp4",
            "format": "mp4",
            "encoding": "h264",
            "shape": [vr["height"], vr["width"], 3],
            "dtype": "uint8",
            "frame_id": vr.get("frame_id", stream_id),
            "time": {
                "clock_id": "segment",
                "sampling": "cfr",
                "rate_hz": vr["output_fps"],
                "start_ns": 0,
                "end_ns": duration_ns,
            },
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": source_assets[0]["source_asset_id"] if source_assets else "raw_color_0",
                "operation": "trim_transcode_resample",
                "sample_map_uri": vr.get("sample_map_uri", f"maps/{stream_id}_sample_map.parquet"),
            },
        })

    # 深度流
    if depth_npz_path is not None:
        streams.append({
            "stream_id": "ego_depth",
            "role": "observation",
            "modality": "depth",
            "uri": "data/ego_depth.mp4",
            "format": "mp4",
            "encoding": "ffv1",
            "dtype": "uint16",
            "frame_id": "depth_optical_frame",
            "time": {
                "clock_id": "segment",
                "sampling": "cfr",
                "rate_hz": 30.0,
                "start_ns": 0,
                "end_ns": duration_ns,
            },
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": "raw_mcap" if profile != "guida" else "raw_depth_0",
                "operation": "trim_decode_ffv1",
            },
        })

    # IMU 流 — 每个 imu_result 生成一个 stream entry
    for ir in (imu_results or []):
        streams.append({
            "stream_id": ir["stream_id"],
            "role": "state",
            "modality": "imu",
            "uri": ir["uri"],
            "format": "parquet",
            "time": {
                "clock_id": "segment",
                "sampling": "irregular",
                "timestamp_column": "timestamp_ns",
            },
            "fields": [
                {
                    "name": "linear_acceleration",
                    "shape": [3],
                    "dtype": "float32",
                    "unit": "m/s^2",
                    "frame_id": "imu",
                },
                {
                    "name": "angular_velocity",
                    "shape": [3],
                    "dtype": "float32",
                    "unit": "rad/s",
                    "frame_id": "imu",
                },
            ],
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": source_assets[0]["source_asset_id"] if source_assets else "raw_imu_0",
                "operation": "trim_and_unit_normalize",
            },
        })

    segment = {
        "zrds_version": "0.1.0",
        "record_revision": revision,
        "segment_id": segment_id,
        "source_type": "ego",

        "source_session": {
            "session_id": session_id,
            "session_uri": str(data_dir.resolve()),
        },

        "source_assets": source_assets,

        "timeline": {
            "start_ns": 0,
            "end_ns": duration_ns,
            "continuous": True,
        },

        "source_span": {
            "source_clock_id": "device_clock",
            "start_ns": span["source_start_ns"],
            "end_ns": span["source_end_ns"],
        },

        "streams": streams,

        "calibration_uri": "calibration/calibration.json",

        "quality": {
            "status": "warn" if (quality_issues and len(quality_issues) > 0) else "pass",
            "issues": quality_issues or [],
        },
    }

    return segment


def write_segment_json(segment: dict, output_dir: str) -> str:
    """写出 segment.json。

    Returns:
        输出文件路径
    """
    seg_dir = Path(output_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    output_path = seg_dir / "segment.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segment, f, indent=2, ensure_ascii=False)
    return str(output_path)
