"""
写出后验证：确认所有文件可读、数据一致。
"""

import json
import hashlib
from pathlib import Path

import cv2
import pandas as pd
import numpy as np


def sha256_hex(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_segment(output_dir: str) -> dict:
    """对已生成的 Prepared Segment 做写出后验证。

    Returns:
        {
            "status": "pass" | "fail",
            "checks": {...},
            "statistics": {...},
            "errors": [...],
        }
    """
    seg_dir = Path(output_dir)
    errors = []
    checks = {}
    stats = {}

    # ---- 1. segment.json 存在且可解析 ----
    seg_path = seg_dir / "segment.json"
    if not seg_path.exists():
        return {"status": "fail", "checks": {}, "statistics": {}, "errors": ["segment.json not found"]}

    with open(seg_path) as f:
        segment = json.load(f)

    # ---- 2. 引用的文件全部存在 ----
    referenced = []
    for stream in segment.get("streams", []):
        uri = seg_dir / stream["uri"]
        referenced.append(str(uri))
        if not uri.exists():
            errors.append(f"Missing stream file: {uri}")

    calib_uri = seg_dir / segment.get("calibration_uri", "")
    if calib_uri.exists():
        referenced.append(str(calib_uri))
    else:
        errors.append(f"Missing calibration: {calib_uri}")

    # sample_map
    for stream in segment.get("streams", []):
        sm_uri = stream.get("origin", {}).get("sample_map_uri", "")
        if sm_uri:
            sm_path = seg_dir / sm_uri
            referenced.append(str(sm_path))
            if not sm_path.exists():
                errors.append(f"Missing sample_map: {sm_path}")

    checks["referenced_files_exist"] = "pass" if not any(
        "Missing" in e for e in errors
    ) else "fail"

    # ---- 3. 视频流可解码（按 segment.json 中的 streams 遍历） ----
    video_streams = [s for s in segment.get("streams", [])
                     if s.get("format") == "mp4"]
    all_video_ok = True
    for vs in video_streams:
        vpath = seg_dir / vs["uri"]
        try:
            cap = cv2.VideoCapture(str(vpath))
            video_ok, frame = cap.read()
            cap.release()
            if not video_ok or frame is None:
                all_video_ok = False
                errors.append(f"Video decode failed: {vs['uri']}")
        except Exception:
            all_video_ok = False
            errors.append(f"Video open failed: {vs['uri']}")
    checks["video_decode"] = "pass" if all_video_ok and video_streams else "fail"

    # ---- 4. 视频帧数 == sample_map 行数（测试主相机的 sample_map） ----
    sm_path = seg_dir / "maps" / "rgb_sample_map.parquet"
    if sm_path.exists():
        sm = pd.read_parquet(str(sm_path))
        stats["sample_map_rows"] = len(sm)

        if video_streams:
            cap = cv2.VideoCapture(str(seg_dir / video_streams[0]["uri"]))
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            stats["rgb_frames"] = video_frames

            match = abs(video_frames - len(sm)) <= 2
            checks["video_sample_map_count_match"] = "pass" if match else "fail"
            if not match:
                errors.append(
                    f"Video frames ({video_frames}) != sample_map rows ({len(sm)})"
                )
    else:
        checks["video_sample_map_count_match"] = "skip"

    # ---- 5. sample_map 时间单调 ----
    if sm_path.exists():
        ts = sm["output_timestamp_ns"].values
        monotonic = bool(np.all(np.diff(ts) > 0))
        checks["sample_map_monotonic"] = "pass" if monotonic else "fail"
        if not monotonic:
            errors.append("Sample map timestamps not monotonic")
        stats["max_rgb_mapping_error_ns"] = int(sm["time_error_ns"].abs().max())

    # ---- 6. IMU 可读且时间单调 ----
    imu_path = seg_dir / "data" / "imu.parquet"
    if imu_path.exists():
        imu = pd.read_parquet(str(imu_path))
        stats["imu_samples"] = len(imu)
        ts = imu["timestamp_ns"].values
        imu_mono = bool(np.all(np.diff(ts) >= 0))
        checks["imu_timestamp_monotonic"] = "pass" if imu_mono else "fail"
        if not imu_mono:
            errors.append("IMU timestamps not monotonic")

        # 检查是否从接近 0 开始
        min_ts = imu["timestamp_ns"].min()
        checks["imu_starts_near_zero"] = (
            "pass" if min_ts >= 0 and min_ts < 1_000_000_000 else "warn"
        )

    # ---- 7. 统计 ----
    stats["duration_ns"] = segment["timeline"]["end_ns"] - segment["timeline"]["start_ns"]

    # ---- 汇总 ----
    all_pass = len(errors) == 0
    status = "pass" if all_pass else "fail"

    return {
        "status": status,
        "checks": checks,
        "statistics": stats,
        "errors": errors,
    }


def write_validation_report(validation: dict, output_dir: str) -> str:
    """写出 validation.json。

    Returns:
        输出文件路径
    """
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "validation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)
    return str(output_path)
