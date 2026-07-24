"""
生成 segment.json — Prepared Segment 的核心控制文件。
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
    video_result: dict | None = None,
    sample_map_rows: int = 0,
    imu_rows: int = 0,
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

    Args:
        video_results: 多相机转码结果列表 (Dunjia)。单相机时兼容 video_result。
        depth_npz_path: 深度 .npz 文件路径 (Dunjia)
        calibrations: 多相机标定 dict {cam_name: {...}} (Dunjia)
    """
    data_dir = Path(dataset_path)
    color_path = data_dir / "color_000000.mkv"
    index_path = data_dir / "index.jsonl"
    imu_path = data_dir / "imu" / "imu_000000.csv"
    meta_path = data_dir / "meta.json"

    duration_ns = span["source_end_ns"] - span["source_start_ns"]

    # backward compat: single video_result → list
    if video_results is None and video_result is not None:
        video_results = [video_result]

    # source_assets
    if source_assets is None:
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

    streams = []

    # RGB 视频流
    if profile == "dunjia" and video_results:
        cam_configs = [
            ("camera0", "ego_rgb_center", "headcam_center_optical_frame"),
            ("camera1", "ego_rgb_left", "headcam_left_optical_frame"),
            ("camera2", "ego_rgb_right", "headcam_right_optical_frame"),
        ]
        for cam_name, stream_id, frame_id in cam_configs:
            # 找到对应的 video_result
            vr = next((v for v in video_results
                       if v.get("camera_name") == cam_name), None)
            if vr is None:
                continue
            streams.append({
                "stream_id": stream_id,
                "role": "observation",
                "modality": "rgb",
                "uri": f"data/{stream_id}.mp4",
                "format": "mp4",
                "encoding": "h264",
                "shape": [vr["height"], vr["width"], 3],
                "dtype": "uint8",
                "frame_id": frame_id,
                "time": {
                    "clock_id": "segment",
                    "sampling": "cfr",
                    "rate_hz": vr["output_fps"],
                    "start_ns": 0,
                    "end_ns": duration_ns,
                },
                "origin": {
                    "kind": "deterministic_transform",
                    "source_asset_id": "raw_mcap",
                    "operation": "trim_transcode_resample",
                    "sample_map_uri": "maps/rgb_sample_map.parquet",
                },
            })
    else:
        # Guida / 单相机
        vr = video_results[0] if video_results else {}
        streams.append({
            "stream_id": "ego_rgb",
            "role": "observation",
            "modality": "rgb",
            "uri": "data/ego_rgb.mp4",
            "format": "mp4",
            "encoding": "h264",
            "shape": [vr.get("height", 0), vr.get("width", 0), 3],
            "dtype": "uint8",
            "frame_id": "ego_camera_optical",
            "time": {
                "clock_id": "segment",
                "sampling": "cfr",
                "rate_hz": vr.get("output_fps", 30.0),
                "start_ns": 0,
                "end_ns": duration_ns,
            },
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": "raw_color_0",
                "operation": "trim_transcode_resample",
                "sample_map_uri": "maps/rgb_sample_map.parquet",
            },
        })

    # 深度流 (Dunjia) — H.265 无损 MP4
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
                "source_asset_id": "raw_mcap",
                "operation": "trim_decode_ffv1",
            },
        })

    # IMU 流
    streams.append({
        "stream_id": "ego_imu",
        "role": "state",
        "modality": "imu",
        "uri": "data/imu.parquet",
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
            "source_asset_id": "raw_imu_0" if profile == "guida" else "raw_mcap",
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
