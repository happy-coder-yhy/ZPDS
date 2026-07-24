"""
读取遁甲 (Dunjia) 数据集 — Foxglove MCAP 容器。

消费 MCAP 文件，暴露与 guida_reader.py 相同签名的函数，
使现有检测器、规划器和写入器零改动即可复用。

MCAP 内部使用 foxglove protobuf schema：
  - foxglove.CompressedVideo  (H264, Annex B 起始码)
  - foxglove.CompressedImage   (PNG depth)
  - foxglove.Imu               (Vector3 accel + gyro)
  - foxglove.CameraCalibration (pinhole model)

硬约束 (per AGENTS.md)：
  - 同时保留 MCAP log_time 和消息内 timestamp，不得静默合并。
  - H264 重建保持消息时间、关键帧/GOP 和 topic 映射。
  - 深度 PNG 的 dtype、invalid 值和物理单位必须实测。
"""

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


# ---- 默认 topic 名称 ----
TOPIC_CAMERA0 = "/robot0/sensor/camera0/compressed"
TOPIC_CAMERA1 = "/robot0/sensor/camera1/compressed"
TOPIC_CAMERA2 = "/robot0/sensor/camera2/compressed"
TOPIC_DEPTH = "/robot0/sensor/depth/compressed"
TOPIC_IMU = "/robot0/sensor/imu"
TOPIC_CAMERA0_CALIB = "/robot0/sensor/camera0/camera_info"
TOPIC_CAMERA1_CALIB = "/robot0/sensor/camera1/camera_info"
TOPIC_CAMERA2_CALIB = "/robot0/sensor/camera2/camera_info"
TOPIC_DEPTH_CALIB = "/robot0/sensor/depth/calibration"


def _open_mcap(mcap_path: str):
    """打开 MCAP 文件，返回 (reader, file_handle)。

    调用方负责在完成后 close file_handle。
    """
    path = Path(mcap_path)
    if not path.exists():
        raise FileNotFoundError(f"MCAP 文件不存在: {mcap_path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {mcap_path}")
    fh = open(str(path), "rb")
    reader = make_reader(fh, decoder_factories=[DecoderFactory()])
    return reader, fh


# ================================================================
# 元数据
# ================================================================

def read_meta(dataset_path: str) -> dict[str, Any]:
    """扫描 MCAP，提取 Session 级元数据。

    Returns:
        {
            "device": "Dunjia",
            "fps": float,           # 从实际帧间隔推算
            "frame_count": int,     # 主相机帧数 (camera0)
            "width": int,           # 主相机宽度
            "height": int,          # 主相机高度
            "dropped_frames": int,  # 通过时间戳间隔估算
            "imu_sample_rate": float,
            "session_id": str,
        }
    """
    reader, fh = _open_mcap(dataset_path)
    try:
        camera0_ts = []
        imu_ts = []
        width = 0
        height = 0
        has_calib = False

        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            if topic == TOPIC_CAMERA0:
                camera0_ts.append(decoded.timestamp.seconds * 1_000_000_000
                                  + decoded.timestamp.nanos)
            elif topic == TOPIC_IMU:
                imu_ts.append(decoded.timestamp.seconds * 1_000_000_000
                              + decoded.timestamp.nanos)
            elif topic == TOPIC_CAMERA0_CALIB and not has_calib:
                width = decoded.width
                height = decoded.height
                has_calib = True

        # 从实际时间戳推算帧率
        if len(camera0_ts) >= 2:
            intervals = np.diff(np.sort(camera0_ts))
            median_interval_ns = np.median(intervals)
            fps = 1e9 / median_interval_ns if median_interval_ns > 0 else 25.0
        else:
            fps = 25.0

        # 检测丢帧（间隔 > 2x 中位数）
        if len(camera0_ts) >= 2:
            threshold = median_interval_ns * 2
            dropped = int(np.sum(intervals > threshold))
        else:
            dropped = 0

        # IMU 采样率
        if len(imu_ts) >= 2:
            imu_intervals = np.diff(np.sort(imu_ts))
            imu_median_ns = np.median(imu_intervals)
            imu_rate = 1e9 / imu_median_ns if imu_median_ns > 0 else 196.0
        else:
            imu_rate = 196.0

        return {
            "device": "Dunjia",
            "fps": round(fps, 1),
            "frame_count": len(camera0_ts),
            "width": width or 1600,
            "height": height or 1300,
            "dropped_frames": dropped,
            "imu_sample_rate": round(imu_rate, 1),
        }
    finally:
        fh.close()


# ================================================================
# 帧索引 (index.jsonl 等价)
# ================================================================

def read_index_frames(dataset_path: str) -> list[dict]:
    """从 camera0 CompressedVideo 消息构建帧索引列表。

    每项包含：
      - seq: 帧序号 (0-based)
      - timestamp_ns: 消息内 timestamp (int, 权威时间轴)
      - log_time_ns: MCAP 容器 log_time (int)
      - publish_time_ns: MCAP publish_time (int)
      - h264_size: H264 数据字节数

    保留三重时间戳以满足双时间戳硬约束。
    """
    reader, fh = _open_mcap(dataset_path)
    try:
        frames = []
        local_seq = 0  # 0-based 视频帧索引，非 MCAP 全局序号
        for (_schema, channel, msg, decoded) in reader.iter_decoded_messages():
            if channel.topic != TOPIC_CAMERA0:
                continue
            ts = decoded.timestamp
            ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
            frames.append({
                "seq": local_seq,
                "timestamp_ns": ts_ns,
                "log_time_ns": msg.log_time,
                "publish_time_ns": msg.publish_time,
                "h264_size": len(decoded.data),
            })
            local_seq += 1
        return frames
    finally:
        fh.close()


def read_index_timestamps(dataset_path: str) -> list[int]:
    """返回主相机消息内 timestamp_ns 的有序列表。"""
    frames = read_index_frames(dataset_path)
    return [f["timestamp_ns"] for f in frames]


# ================================================================
# IMU
# ================================================================

def read_imu(dataset_path: str) -> pd.DataFrame:
    """解析 foxglove.Imu 消息，返回与 guida_reader 兼容的 DataFrame。

    Columns:
      timestamp_ns, ax, ay, az, gx, gy, gz

    保留消息内 timestamp 作为 timestamp_ns。
    加速度单位由 foxglove schema 规定为 m/s²，
    角速度单位为 rad/s。
    """
    reader, fh = _open_mcap(dataset_path)
    try:
        rows = []
        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if channel.topic != TOPIC_IMU:
                continue
            ts = decoded.timestamp
            ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
            rows.append({
                "timestamp_ns": ts_ns,
                "ax": decoded.linear_acceleration.x,
                "ay": decoded.linear_acceleration.y,
                "az": decoded.linear_acceleration.z,
                "gx": decoded.angular_velocity.x,
                "gy": decoded.angular_velocity.y,
                "gz": decoded.angular_velocity.z,
            })
        if not rows:
            return pd.DataFrame(columns=[
                "timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz",
            ])
        return pd.DataFrame(rows)
    finally:
        fh.close()


# ================================================================
# 视频文件 (color_000000.mkv 等价)
# ================================================================

def reconstruct_h264(mcap_path: str, output_path: str | None = None) -> str:
    """从 camera0 CompressedVideo 消息重构 H264 比特流。

    MCAP 中的 H264 数据已使用 Annex B 起始码 (0x00000001)。
    先拼接为 raw .h264，再用 ffmpeg 重封装为 .mp4，确保 OpenCV 可读。

    Args:
        mcap_path: MCAP 文件路径
        output_path: 输出 .mp4 路径。默认在 MCAP 同目录生成 .cache.mp4

    Returns:
        写入的 .mp4 文件路径
    """
    reader, fh = _open_mcap(mcap_path)
    try:
        if output_path is None:
            output_path = str(Path(mcap_path).with_suffix(".cache.mp4"))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Step 1: 拼接 raw .h264
        raw_h264 = output_path.replace(".mp4", ".h264")
        nal_count = 0
        with open(raw_h264, "wb") as f_out:
            for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
                if channel.topic != TOPIC_CAMERA0:
                    continue
                f_out.write(decoded.data)
                nal_count += 1

        if nal_count == 0:
            raise ValueError(f"MCAP 中没有 {TOPIC_CAMERA0} 消息")

        # Step 2: ffmpeg 重封装为 .mp4（不重新编码，只换容器）
        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", raw_h264,
                "-c", "copy",
                output_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 重封装失败: {result.stderr.strip()}"
            )

        # 清理临时 .h264
        try:
            os.remove(raw_h264)
        except OSError:
            pass

        return output_path
    finally:
        fh.close()


def get_color_video(dataset_path: str) -> str:
    """返回视频文件路径用于 OpenCV 检测和转码。

    对遁甲而言，从 MCAP 提取 H264 并重封装为 .mp4，缓存在 MCAP 同目录。
    """
    cache_path = str(Path(dataset_path).with_suffix(".cache.mp4"))
    if Path(cache_path).exists():
        return cache_path
    return reconstruct_h264(dataset_path, cache_path)


def get_session_id(dataset_path: str) -> str:
    """从文件名推导 session_id。

    Example:
        20260618_084650_00.mcap -> dunjia_20260618_084650_00
    """
    stem = Path(dataset_path).stem
    return f"dunjia_{stem}"


# ================================================================
# 标定
# ================================================================

def read_calibration(mcap_path: str) -> dict[str, Any]:
    """从 MCAP CameraCalibration 消息提取相机标定。

    Returns:
        {
            "width": int, "height": int,
            "frame_id": str,
            "K": [9], "D": [n], "R": [9], "P": [12],
            "distortion_model": str,
        }
    """
    reader, fh = _open_mcap(mcap_path)
    try:
        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if channel.topic == TOPIC_CAMERA0_CALIB:
                return {
                    "width": decoded.width,
                    "height": decoded.height,
                    "frame_id": decoded.frame_id,
                    "K": list(decoded.K),
                    "D": list(decoded.D),
                    "R": list(decoded.R),
                    "P": list(decoded.P),
                    "distortion_model": decoded.distortion_model,
                }
        raise ValueError(f"MCAP 中没有 {TOPIC_CAMERA0_CALIB} 消息")
    finally:
        fh.close()


# ================================================================
# 时间范围
# ================================================================

def read_session_bounds(mcap_path: str) -> tuple[int, int]:
    """读取 Session 时间范围 (start_ns, end_ns)。

    基于 camera0 的消息内 timestamp。
    """
    timestamps = read_index_timestamps(mcap_path)
    if not timestamps:
        raise ValueError("MCAP 中没有 camera0 帧")
    return timestamps[0], timestamps[-1]


# ================================================================
# 导出符号 (与 guida_reader 一致)
# ================================================================

__all__ = [
    "read_meta",
    "read_index_frames",
    "read_index_timestamps",
    "read_imu",
    "get_color_video",
    "get_session_id",
    "reconstruct_h264",
    "read_calibration",
    "read_session_bounds",
]
