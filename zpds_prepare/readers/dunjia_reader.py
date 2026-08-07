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

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

# ---- topic 名称 ----
TOPIC_CAMERA0 = "/robot0/sensor/camera0/compressed"
TOPIC_CAMERA1 = "/robot0/sensor/camera1/compressed"
TOPIC_CAMERA2 = "/robot0/sensor/camera2/compressed"
TOPIC_DEPTH = "/robot0/sensor/depth/compressed"
TOPIC_IMU = "/robot0/sensor/imu"
TOPIC_CAMERA0_CALIB = "/robot0/sensor/camera0/camera_info"
TOPIC_CAMERA1_CALIB = "/robot0/sensor/camera1/camera_info"
TOPIC_CAMERA2_CALIB = "/robot0/sensor/camera2/camera_info"
TOPIC_DEPTH_CALIB = "/robot0/sensor/depth/calibration"
TOPIC_AUDIO = "/robot0/sensor/audio"

# ---- 映射表 ----
CAMERA_TOPICS = {
    "camera0": TOPIC_CAMERA0,
    "camera1": TOPIC_CAMERA1,
    "camera2": TOPIC_CAMERA2,
    "depth": TOPIC_DEPTH,
}
CALIB_TOPICS = {
    "camera0": TOPIC_CAMERA0_CALIB,
    "camera1": TOPIC_CAMERA1_CALIB,
    "camera2": TOPIC_CAMERA2_CALIB,
    "depth": TOPIC_DEPTH_CALIB,
}
CAMERA_IDS = {
    "camera0": "headcam_center_optical_frame",
    "camera1": "headcam_left_optical_frame",
    "camera2": "headcam_right_optical_frame",
    "depth": "depth_optical_frame",
}


def _open_mcap(mcap_path: str):
    """打开 MCAP 文件，返回 (reader, file_handle)。"""
    path = Path(mcap_path)
    if not path.exists():
        raise FileNotFoundError(f"MCAP 文件不存在: {mcap_path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {mcap_path}")
    fh = open(str(path), "rb")  # noqa: SIM115 - caller closes returned handle
    reader = make_reader(fh, decoder_factories=[DecoderFactory()])
    return reader, fh


# ================================================================
# 元数据
# ================================================================

def read_meta(dataset_path: str) -> dict[str, Any]:
    """扫描 MCAP，提取 Session 级元数据。

    Returns:
        {device, fps, frame_count, width, height, dropped_frames, imu_sample_rate}
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

        if len(camera0_ts) >= 2:
            intervals = np.diff(np.sort(camera0_ts))
            median_interval_ns = np.median(intervals)
            fps = 1e9 / median_interval_ns if median_interval_ns > 0 else 25.0
            dropped = int(np.sum(intervals > median_interval_ns * 2))
        else:
            fps = 25.0
            dropped = 0

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


def count_messages(mcap_path: str, topic: str) -> int:
    """快速统计指定 topic 的消息数。"""
    reader, fh = _open_mcap(mcap_path)
    try:
        cnt = 0
        for _schema, channel, _msg, _decoded in reader.iter_decoded_messages():
            if channel.topic == topic:
                cnt += 1
        return cnt
    finally:
        fh.close()


# ================================================================
# 帧索引 (index.jsonl 等价)
# ================================================================

def read_index_frames(dataset_path: str, topic: str | None = None) -> list[dict]:
    """从指定 CompressedVideo topic 构建帧索引列表。

    Args:
        dataset_path: MCAP 文件路径
        topic: topic 名称，默认 camera0

    每项包含 seq, timestamp_ns, log_time_ns, publish_time_ns, h264_size。
    """
    if topic is None:
        topic = TOPIC_CAMERA0

    reader, fh = _open_mcap(dataset_path)
    try:
        frames = []
        local_seq = 0
        for (_schema, channel, msg, decoded) in reader.iter_decoded_messages():
            if channel.topic != topic:
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


def read_index_timestamps(dataset_path: str, topic: str | None = None) -> list[int]:
    """返回指定 topic 的消息内 timestamp_ns 有序列表。"""
    frames = read_index_frames(dataset_path, topic)
    return [f["timestamp_ns"] for f in frames]


# ================================================================
# IMU
# ================================================================

@dataclass
class AudioPacket:
    """一条 foxglove.CompressedAudio 消息的解码结果。

    Attributes:
        timestamp_ns: 消息内 timestamp（protobuf Timestamp → ns）。
        data: 压缩音频负载。format="opus" 时为单个 raw Opus 包（无 Ogg 封装）。
        format: 压缩格式字符串（如 "opus"）。
        log_time_ns: MCAP log_time（原始记录时刻），与 timestamp_ns 分开保留。
    """

    timestamp_ns: int
    data: bytes
    format: str = "opus"
    log_time_ns: int = 0


def has_audio_topic(mcap_path: str) -> bool:
    """探测 MCAP 是否包含音频 topic（快速，不遍历全部消息）。"""
    path = Path(mcap_path)
    if not path.is_file():
        return False
    try:
        with open(str(path), "rb") as fh:
            reader = make_reader(fh)
            for _schema, channel, _msg in reader.iter_messages():
                if channel.topic == TOPIC_AUDIO:
                    return True
    except Exception:
        return False
    return False


def read_audio(dataset_path: str) -> list[AudioPacket]:
    """读取遁甲 MCAP 的 /robot0/sensor/audio 音频消息。

    解析 foxglove.CompressedAudio（protobuf）：
        message CompressedAudio {
          google.protobuf.Timestamp timestamp = 1;  // 采集时刻
          bytes  data   = 2;                         // 压缩音频负载
          string format = 3;                         // "opus" 等
        }

    Returns:
        按时间戳排序的 AudioPacket 列表；MCAP 无音频 topic 时返回空列表。
    """
    reader, fh = _open_mcap(dataset_path)
    try:
        packets: list[AudioPacket] = []
        for _schema, channel, msg, decoded in reader.iter_decoded_messages():
            if channel.topic != TOPIC_AUDIO:
                continue
            ts = decoded.timestamp
            ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
            fmt = getattr(decoded, "format", None) or "opus"
            packets.append(AudioPacket(
                timestamp_ns=ts_ns,
                data=bytes(decoded.data),
                format=fmt,
                log_time_ns=msg.log_time,
            ))
        packets.sort(key=lambda p: p.timestamp_ns)
        return packets
    finally:
        fh.close()


def read_imu(dataset_path: str) -> pd.DataFrame:
    """解析 foxglove.Imu 消息，返回与 guida_reader 兼容的 DataFrame。

    Columns: timestamp_ns, ax, ay, az, gx, gy, gz
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

def reconstruct_video(
    mcap_path: str,
    topic: str | None = None,
    output_path: str | None = None,
) -> str:
    """从 MCAP 消息重构 H264 视频并重封装为 .mp4。

    Args:
        mcap_path: MCAP 文件路径
        topic: 视频 topic，默认 camera0
        output_path: 输出 .mp4 路径。未指定时写入项目输出缓存，不写 MCAP 目录。

    Returns:
        写入的 .mp4 文件路径
    """
    if topic is None:
        topic = TOPIC_CAMERA0

    reader, fh = _open_mcap(mcap_path)
    try:
        if output_path is None:
            cache_dir = _default_cache_dir(mcap_path)
            # 生成短名: camera0→cam0, camera1→cam1, etc.
            for cam_name, t in CAMERA_TOPICS.items():
                if t == topic:
                    short = cam_name.replace("camera", "cam").replace("depth", "depth")
                    output_path = str(cache_dir / f"{short}.mp4")
                    break
            else:
                output_path = str(cache_dir / "video.mp4")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Step 1: 拼接 raw .h264
        raw_h264 = output_path.replace(".mp4", ".h264")
        nal_count = 0
        with open(raw_h264, "wb") as f_out:
            for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
                if channel.topic != topic:
                    continue
                f_out.write(decoded.data)
                nal_count += 1

        if nal_count == 0:
            raise ValueError(f"MCAP 中没有 {topic} 消息")

        # Step 2: ffmpeg 重封装
        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", raw_h264, "-c", "copy", output_path],
            capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 重封装失败 ({topic}): {result.stderr.strip()}")

        try:
            os.remove(raw_h264)
        except OSError:
            pass

        return output_path
    finally:
        fh.close()


def _default_cache_dir(dataset_path: str) -> Path:
    """返回源目录之外的稳定缓存目录。"""
    source = Path(dataset_path).resolve()
    cache_key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    return Path.cwd() / "output" / ".cache" / "dunjia" / (
        f"{source.stem}-{cache_key}"
    )


def get_color_video(
    dataset_path: str,
    cache_dir: str | Path | None = None,
) -> str:
    """返回主相机 camera0 的 .mp4 缓存路径（向后兼容）。"""
    return get_video_for_topic(dataset_path, TOPIC_CAMERA0, cache_dir=cache_dir)


def get_video_for_topic(
    dataset_path: str,
    topic: str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """返回指定 topic 的 .mp4 缓存路径，不存在则在独立缓存目录重建。"""
    if topic is None:
        topic = TOPIC_CAMERA0
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else _default_cache_dir(dataset_path)
    )
    cache_root = cache_root.resolve()
    source_parent = Path(dataset_path).resolve().parent
    if cache_root == source_parent or cache_root.is_relative_to(source_parent):
        raise ValueError("Dunjia 缓存目录不能位于原始 MCAP 目录内")

    # 生成缓存路径
    for cam_name, t in CAMERA_TOPICS.items():
        if t == topic:
            short = cam_name.replace("camera", "cam").replace("depth", "depth")
            cache_path = str(cache_root / f"{short}.mp4")
            if Path(cache_path).exists():
                return cache_path
            return reconstruct_video(dataset_path, topic, cache_path)
    # fallback
    return reconstruct_video(
        dataset_path,
        topic,
        str(cache_root / "video.mp4"),
    )


def get_session_id(dataset_path: str) -> str:
    """从文件名推导 session_id。"""
    stem = Path(dataset_path).stem
    return f"dunjia_{stem}"


# ================================================================
# 深度
# ================================================================

def read_depth_frames(
    mcap_path: str,
    topic: str | None = None,
) -> list[dict]:
    """从 MCAP 读取深度 PNG 帧，解码并返回结构化列表。

    Returns:
        [{seq, timestamp_ns, log_time_ns, width, height, dtype, min_val, max_val}, ...]
        不返回原始像素数据（太大），只返回元信息。
        实际像素数据通过 write_depth_npz 写出。
    """
    import cv2

    if topic is None:
        topic = TOPIC_DEPTH

    reader, fh = _open_mcap(mcap_path)
    try:
        frames = []
        local_seq = 0
        for _schema, channel, msg, decoded in reader.iter_decoded_messages():
            if channel.topic != topic:
                continue
            ts = decoded.timestamp
            ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
            # 解码 PNG → numpy 获取元信息
            nparr = np.frombuffer(decoded.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            frames.append({
                "seq": local_seq,
                "timestamp_ns": ts_ns,
                "log_time_ns": msg.log_time,
                "width": img.shape[1] if img is not None else 0,
                "height": img.shape[0] if img is not None else 0,
                "dtype": str(img.dtype) if img is not None else "unknown",
                "min_val": int(img.min()) if img is not None else 0,
                "max_val": int(img.max()) if img is not None else 0,
            })
            local_seq += 1
        return frames
    finally:
        fh.close()


def write_depth_npz(
    mcap_path: str,
    output_path: str,
    source_start_ns: int,
    source_end_ns: int,
    topic: str | None = None,
) -> str:
    """解码深度 PNG 帧 → 裁剪 → 写出 .npz 文件。"""
    import cv2

    if topic is None:
        topic = TOPIC_DEPTH

    reader, fh = _open_mcap(mcap_path)
    try:
        timestamps = []
        images = []
        for _schema, channel, msg, decoded in reader.iter_decoded_messages():
            if channel.topic != topic:
                continue
            ts = decoded.timestamp
            ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
            if ts_ns < source_start_ns or ts_ns > source_end_ns:
                continue
            nparr = np.frombuffer(decoded.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                timestamps.append(ts_ns)
                images.append(img)

        if not images:
            raise ValueError("时间范围内没有深度帧")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        stack = np.stack(images, axis=0)
        np.savez_compressed(
            output_path,
            frames=stack,
            timestamps=np.array(timestamps, dtype=np.int64),
            source_start_ns=np.int64(source_start_ns),
            source_end_ns=np.int64(source_end_ns),
        )
        return output_path
    finally:
        fh.close()


def transcode_depth_video(
    mcap_path: str,
    output_path: str,
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
    topic: str | None = None,
) -> dict:
    """解码深度 PNG 帧 → H.265 无损 MP4 视频。

    使用 libx265 lossless 模式保留 uint16 深度精度。
    FFmpeg 管道: rawvideo gray16le → libx265 lossless → mp4

    Returns:
        {output_frames, output_fps, width, height, codec, output_path}
    """
    import cv2

    if topic is None:
        topic = TOPIC_DEPTH

    reader, fh = _open_mcap(mcap_path)
    try:
        images = []
        timestamps_ns = []
        width, height = 0, 0
        for _schema, channel, msg, decoded in reader.iter_decoded_messages():
            if channel.topic != topic:
                continue
            ts = decoded.timestamp
            ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
            if ts_ns < source_start_ns or ts_ns > source_end_ns:
                continue
            nparr = np.frombuffer(decoded.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if width == 0:
                    height, width = img.shape
                images.append(img)
                timestamps_ns.append(ts_ns)

        if not images:
            raise ValueError("时间范围内没有深度帧")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        timestamps_arr = np.array(timestamps_ns, dtype=np.int64)
        frame_interval_ns = int(1_000_000_000 / target_fps)
        output_count = int((source_end_ns - source_start_ns) / frame_interval_ns)

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "gray16le",
            "-s", f"{width}x{height}",
            "-r", str(target_fps),
            "-i", "-",
            "-c:v", "ffv1",
            "-level", "3",
            "-coder", "1",
            "-context", "1",
            "-pix_fmt", "gray16le",
            output_path,
        ]
        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.stdin is None or proc.stderr is None:
            proc.kill()
            raise RuntimeError("无法创建 FFmpeg 深度编码管道")
        stdin = proc.stdin
        stderr_pipe = proc.stderr

        total_output = 0
        try:
            for out_idx in range(output_count):
                target_ts = source_start_ns + out_idx * frame_interval_ns
                if target_ts > source_end_ns:
                    break
                nearest_idx = int(np.argmin(np.abs(timestamps_arr - target_ts)))
                frame = images[nearest_idx]
                stdin.write(frame.tobytes())
                total_output += 1

            stdin.close()
            ret = proc.wait(timeout=120)
            if ret != 0:
                stderr = stderr_pipe.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg 深度编码失败: {stderr.strip()}")
        except Exception:
            proc.kill()
            raise

        return {
            "output_frames": total_output,
            "output_fps": target_fps,
            "width": width,
            "height": height,
            "codec": "ffv1",
            "output_path": output_path,
        }
    finally:
        fh.close()


# ================================================================
# 深度帧采样（供 QC Stage 5 使用）
# ================================================================


def sample_depth_frames(
    mcap_path: str,
    total_count: int,
    *,
    topic: str | None = None,
    max_samples: int = 100,
) -> list:
    """从 MCAP 深度通道采样解码帧（纯内存，不落盘）。

    每 ``ceil(total_count / max_samples)`` 帧取 1 帧，上限 ``max_samples``。
    返回解码后的 numpy 数组列表，可直接传入 Stage 5 的 ``depth_frames``。
    """
    import cv2

    if total_count <= 0:
        return []

    step = max(1, total_count // max_samples)
    sample_indices: set[int] = set(range(0, total_count, step))
    # 确保不超过 max_samples
    if len(sample_indices) > max_samples:
        sample_indices = set(sorted(sample_indices)[:max_samples])

    if topic is None:
        topic = TOPIC_DEPTH

    samples: list = []
    reader, fh = _open_mcap(mcap_path)
    try:
        depth_idx = 0
        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if channel.topic != topic:
                continue
            if depth_idx in sample_indices:
                nparr = np.frombuffer(decoded.data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 2:
                    samples.append(img)
            depth_idx += 1
        return samples
    finally:
        fh.close()


# ================================================================
# 标定
# ================================================================

def read_calibration(mcap_path: str, topic: str | None = None) -> dict[str, Any]:
    """从 MCAP CameraCalibration 消息提取相机标定。

    Args:
        mcap_path: MCAP 文件路径
        topic: 标定 topic，默认 camera0 标定
    """
    if topic is None:
        topic = TOPIC_CAMERA0_CALIB

    reader, fh = _open_mcap(mcap_path)
    try:
        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if channel.topic == topic:
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
        raise ValueError(f"MCAP 中没有 {topic} 消息")
    finally:
        fh.close()


# ================================================================
# 时间范围
# ================================================================

def read_session_bounds(mcap_path: str) -> tuple[int, int]:
    """读取 Session 时间范围 (start_ns, end_ns)，基于 camera0。"""
    timestamps = read_index_timestamps(mcap_path)
    if not timestamps:
        raise ValueError("MCAP 中没有 camera0 帧")
    return timestamps[0], timestamps[-1]


# ================================================================
# 导出
# ================================================================

def read_session(
    dataset_path: str,
    cache_dir: str | Path | None = None,
    include_depth: bool = True,
    require_depth: bool = False,
):
    """统一读取 Session 全部流数据。

    一次扫描 MCAP，收集所有 camera 视频流 + IMU 流，
    返回包含 video_streams 和 imu_streams 的 Session 对象。

    Returns:
        Session 对象，包含:
          - video_streams: {"camera0": VideoStream, "camera1": ..., "camera2": ...}
          - imu_streams:  {"robot0_imu": ImuStream}
    """
    if require_depth and not include_depth:
        raise ValueError("require_depth=True 时不能禁用 include_depth")

    from zpds_prepare.readers.session_model import (
        AudioStream,
        DepthStream,
        ImuStream,
        Session,
        VideoStream,
    )

    reader, fh = _open_mcap(dataset_path)
    try:
        # ---- 第一遍扫描：收集所有流数据 ----
        cam_data: dict[str, dict] = {}
        for cam_name in ["camera0", "camera1", "camera2"]:
            cam_data[cam_name] = {
                "frames": [], "has_calib": False,
                "width": 0, "height": 0,
                "calibration": None,
            }

        imu_rows: list[dict] = []
        depth_frames: list[dict] = []
        depth_width = 0
        depth_height = 0
        depth_dtype = "unknown"
        depth_frame_id = CAMERA_IDS["depth"]
        depth_format = "png"
        camera0_fps = 25.0
        camera0_dropped = 0
        imu_rate = 196.0

        for _schema, channel, msg, decoded in reader.iter_decoded_messages():
            topic = channel.topic

            # ---- 视频消息 ----
            for cam_name, cam_topic in CAMERA_TOPICS.items():
                if topic == cam_topic and cam_name in cam_data:
                    ts = decoded.timestamp
                    ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
                    cam_data[cam_name]["frames"].append({
                        "seq": len(cam_data[cam_name]["frames"]),
                        "timestamp_ns": ts_ns,
                        "log_time_ns": msg.log_time,
                        "publish_time_ns": msg.publish_time,
                        "h264_size": len(decoded.data),
                    })
                    break

            # ---- 标定消息 (提取分辨率 + 完整标定) ----
            for cam_name, calib_topic in CALIB_TOPICS.items():
                if topic == calib_topic and cam_name in cam_data:
                    if not cam_data[cam_name]["has_calib"]:
                        cam_data[cam_name]["width"] = decoded.width
                        cam_data[cam_name]["height"] = decoded.height
                        cam_data[cam_name]["has_calib"] = True
                        cam_data[cam_name]["calibration"] = {
                            "width": decoded.width,
                            "height": decoded.height,
                            "frame_id": getattr(decoded, "frame_id", ""),
                            "K": list(getattr(decoded, "K", [])),
                            "D": list(getattr(decoded, "D", [])),
                            "R": list(getattr(decoded, "R", [])),
                            "P": list(getattr(decoded, "P", [])),
                            "distortion_model": getattr(
                                decoded, "distortion_model", "unknown",
                            ),
                        }
                    break

            # ---- IMU 消息 ----
            if topic == TOPIC_IMU:
                ts = decoded.timestamp
                ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
                imu_rows.append({
                    "timestamp_ns": ts_ns,
                    "ax": decoded.linear_acceleration.x,
                    "ay": decoded.linear_acceleration.y,
                    "az": decoded.linear_acceleration.z,
                    "gx": decoded.angular_velocity.x,
                    "gy": decoded.angular_velocity.y,
                    "gz": decoded.angular_velocity.z,
                })

            # ---- 深度 PNG 消息 ----
            if include_depth and topic == TOPIC_DEPTH:
                ts = decoded.timestamp
                ts_ns = ts.seconds * 1_000_000_000 + ts.nanos
                depth_frames.append({
                    "seq": len(depth_frames),
                    "timestamp_ns": ts_ns,
                    "log_time_ns": msg.log_time,
                    "publish_time_ns": msg.publish_time,
                    "source_frame_position": len(depth_frames),
                    "encoded_size": len(decoded.data),
                })
                if len(depth_frames) == 1:
                    import cv2

                    image = cv2.imdecode(
                        np.frombuffer(decoded.data, np.uint8),
                        cv2.IMREAD_UNCHANGED,
                    )
                    if image is None or image.ndim != 2:
                        raise ValueError("Dunjia 首张深度 PNG 无法解码为单通道图像")
                    depth_height, depth_width = image.shape
                    depth_dtype = str(image.dtype)
                    depth_frame_id = decoded.frame_id or depth_frame_id
                    depth_format = decoded.format or depth_format

        # ---- 计算 camera0 元数据 ----
        c0_frames = cam_data["camera0"]["frames"]
        if c0_frames:
            c0_ts = [f["timestamp_ns"] for f in c0_frames]
            if len(c0_ts) >= 2:
                intervals = np.diff(np.sort(c0_ts))
                median_ns = np.median(intervals)
                camera0_fps = 1e9 / median_ns if median_ns > 0 else 25.0
                camera0_dropped = int(np.sum(intervals > median_ns * 2))
            camera0_frame_count = len(c0_ts)
            c0_width = cam_data["camera0"]["width"] or 1600
            c0_height = cam_data["camera0"]["height"] or 1300
        else:
            camera0_frame_count = 0
            c0_width = 1600
            c0_height = 1300

        if len(imu_rows) >= 2:
            imu_ts_sorted = np.sort([r["timestamp_ns"] for r in imu_rows])
            imu_intervals = np.diff(imu_ts_sorted)
            imu_median_ns = np.median(imu_intervals)
            imu_rate = 1e9 / imu_median_ns if imu_median_ns > 0 else 196.0
    finally:
        fh.close()

    # ---- 构建标定 dict（模拟 A2D 格式）----
    calibration: dict[str, object] = {
        "calibration_id": "dunjia_mcap_calibration",
        "cameras": [],
    }
    calib_cameras: list[dict] = []
    cam_name_map = {"camera0": "camera0", "camera1": "camera1", "camera2": "camera2"}
    for cam_name in ["camera0", "camera1", "camera2"]:
        cd = cam_data.get(cam_name, {})
        calib_raw = cd.get("calibration")
        if calib_raw is None:
            continue
        K = calib_raw.get("K", [])
        calib_cameras.append({
            "camera_id": cam_name_map.get(cam_name, cam_name),
            "model": "pinhole",
            "intrinsics": {
                "fx": K[0] if len(K) > 0 else None,
                "fy": K[4] if len(K) > 4 else None,
                "cx": K[2] if len(K) > 2 else None,
                "cy": K[5] if len(K) > 5 else None,
                "distortion_model": calib_raw.get("distortion_model", "unknown"),
                "distortion_coeffs": list(calib_raw.get("D", [])),
            },
            "resolution": {
                "width": calib_raw.get("width"),
                "height": calib_raw.get("height"),
            },
            "source": "mcap_camera_calibration",
        })
    calibration["cameras"] = calib_cameras

    # ---- 构建 meta ----
    meta = {
        "device": "Dunjia",
        "fps": round(camera0_fps, 1),
        "frame_count": camera0_frame_count,
        "width": c0_width,
        "height": c0_height,
        "dropped_frames": camera0_dropped,
        "imu_sample_rate": round(imu_rate, 1),
        "calibration": calibration,
    }

    # ---- 构建 video_streams ----
    video_streams: dict[str, VideoStream] = {}
    vid_topics = {
        "camera0": TOPIC_CAMERA0,
        "camera1": TOPIC_CAMERA1,
        "camera2": TOPIC_CAMERA2,
    }
    for cam_name in ["camera0", "camera1", "camera2"]:
        frames = cam_data[cam_name]["frames"]
        if not frames:
            continue
        ts_list = [f["timestamp_ns"] for f in frames]
        if len(ts_list) >= 2:
            intervals = np.diff(np.sort(ts_list))
            median_ns = np.median(intervals)
            cam_fps = 1e9 / median_ns if median_ns > 0 else 25.0
        else:
            cam_fps = camera0_fps

        # 获取视频路径 (缓存优先)
        video_path = get_video_for_topic(
            dataset_path,
            vid_topics[cam_name],
            cache_dir=cache_dir,
        )

        video_streams[cam_name] = VideoStream(
            stream_id=cam_name,
            timestamps_ns=ts_list,
            index_frames=frames,
            video_path=video_path,
            fps=round(cam_fps, 1),
            width=cam_data[cam_name]["width"],
            height=cam_data[cam_name]["height"],
            frame_count=len(frames),
        )

    # ---- 构建 imu_streams ----
    imu_streams: dict[str, ImuStream] = {}
    if imu_rows:
        imu_df = pd.DataFrame(imu_rows)
        imu_streams["robot0_imu"] = ImuStream(
            stream_id="robot0_imu",
            dataframe=imu_df,
            sample_rate_hz=float(meta["imu_sample_rate"]),
        )

    # ---- 构建 audio_streams（遁甲 MCAP 有 /robot0/sensor/audio 时）----
    audio_streams: dict[str, AudioStream] = {}
    if has_audio_topic(dataset_path):
        audio_pkts = read_audio(dataset_path)
        if audio_pkts:
            audio_streams["ego_audio"] = AudioStream(
                stream_id="ego_audio",
                packets=[
                    {
                        "timestamp_ns": p.timestamp_ns,
                        "data": p.data,
                        "format": p.format,
                        "log_time_ns": p.log_time_ns,
                    }
                    for p in audio_pkts
                ],
                sample_rate_hz=48000,  # Opus 内部固定 48kHz
                channels=1,
                format=audio_pkts[0].format,
            )

    # ---- 构建正式深度流 ----
    depth_streams: dict[str, DepthStream] = {}
    if depth_frames:
        depth_timestamps = [int(frame["timestamp_ns"]) for frame in depth_frames]
        if len(depth_timestamps) >= 2:
            depth_intervals = np.diff(np.array(depth_timestamps, dtype=np.int64))
            depth_median_ns = float(np.median(depth_intervals))
            depth_fps = (
                1_000_000_000 / depth_median_ns
                if depth_median_ns > 0
                else 25.0
            )
        else:
            depth_fps = 25.0
        depth_streams["ego_depth"] = DepthStream(
            stream_id="ego_depth",
            timestamps_ns=depth_timestamps,
            index_frames=depth_frames,
            source_files=[Path(dataset_path)],
            source_kind="mcap_compressed_image",
            fps=round(float(depth_fps), 6),
            width=depth_width,
            height=depth_height,
            frame_count=len(depth_frames),
            dtype=depth_dtype,
            unit="unknown",
            invalid_value=None,
            frame_id=depth_frame_id,
            metadata={
                "topic": TOPIC_DEPTH,
                "compression_format": depth_format,
                "unit_status": "unverified",
                "source_asset_id": "raw_mcap",
                "operation": "trim_decode_embedded_png",
                "preserves_log_time": True,
            },
        )
        # 采样解码深度帧供 QC Stage 5 使用（纯内存，不落盘）
        if include_depth and depth_frames:
            depth_streams["ego_depth"].depth_frames = sample_depth_frames(
                dataset_path,
                len(depth_frames),
            )
    elif require_depth:
        raise FileNotFoundError(
            f"Dunjia MCAP 缺少必需深度 Topic: {TOPIC_DEPTH}"
        )

    return Session(
        session_id=get_session_id(dataset_path),
        source_path=dataset_path,
        meta=meta,
        video_streams=video_streams,
        depth_streams=depth_streams,
        imu_streams=imu_streams,
        audio_streams=audio_streams,
    )


__all__ = [
    "CALIB_TOPICS",
    "CAMERA_IDS",
    "CAMERA_TOPICS",
    "TOPIC_CAMERA0",
    "TOPIC_CAMERA1",
    "TOPIC_CAMERA2",
    "TOPIC_DEPTH",
    "TOPIC_IMU",
    "count_messages",
    "get_color_video",
    "get_session_id",
    "get_video_for_topic",
    "read_calibration",
    "read_depth_frames",
    "read_imu",
    "read_index_frames",
    "read_index_timestamps",
    "read_meta",
    "read_session",
    "read_session_bounds",
    "reconstruct_video",
    "transcode_depth_video",
    "write_depth_npz",
]
