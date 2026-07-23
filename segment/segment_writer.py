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
    video_result: dict,
    sample_map_rows: int,
    imu_rows: int,
    calibration_id: str = "calib_guida_001",
    revision: str = "r0001",
    segment_id: str = "seg_000001",
    session_id: str = "guida_session_001",
    quality_issues: list[dict] | None = None,
) -> dict:
    """构建 segment.json 内容。

    Args:
        dataset_path: 原始数据集根目录
        span: span_determiner 返回的区间信息
        video_result: video_transcoder 返回的视频信息
        sample_map_rows: sample_map 行数
        imu_rows: 规范化后的 IMU 行数
        calibration_id: 标定 ID
        revision: 修订版本号
        segment_id: Segment 唯一 ID
        session_id: 来源 Session ID
        quality_issues: 落在此 Segment 内的 QualityIssue 列表

    Returns:
        segment JSON dict
    """
    data_dir = Path(dataset_path)
    color_path = data_dir / "color_000000.mkv"
    index_path = data_dir / "index.jsonl"
    imu_path = data_dir / "imu" / "imu_000000.csv"
    meta_path = data_dir / "meta.json"

    duration_ns = span["source_end_ns"] - span["source_start_ns"]

    segment = {
        "zrds_version": "0.1.0",
        "record_revision": revision,
        "segment_id": segment_id,
        "source_type": "ego",

        "source_session": {
            "session_id": session_id,
            "session_uri": str(data_dir.resolve()),
        },

        "source_assets": [
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
        ],

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

        "streams": [
            {
                "stream_id": "ego_rgb",
                "role": "observation",
                "modality": "rgb",
                "uri": "data/ego_rgb.mp4",
                "format": "mp4",
                "encoding": "h264",
                "shape": [video_result["height"], video_result["width"], 3],
                "dtype": "uint8",
                "frame_id": "ego_camera_optical",
                "time": {
                    "clock_id": "segment",
                    "sampling": "cfr",
                    "rate_hz": video_result["output_fps"],
                    "start_ns": 0,
                    "end_ns": duration_ns,
                },
                "origin": {
                    "kind": "deterministic_transform",
                    "source_asset_id": "raw_color_0",
                    "operation": "trim_transcode_resample",
                    "sample_map_uri": "maps/rgb_sample_map.parquet",
                },
            },
            {
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
                    "source_asset_id": "raw_imu_0",
                    "operation": "trim_and_unit_normalize",
                },
            },
        ],

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
