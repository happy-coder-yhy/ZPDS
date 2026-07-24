"""
读取 UMI (简智新创) 数据集 — Foxglove MCAP 容器，双端夹爪。

消费 MCAP 文件，解析:
  - /robot{N}/sensor/camera0/compressed  (foxglove.CompressedImage, h264)
  - /robot{N}/sensor/imu                  (foxglove.IMUMeasurement)
  - /robot{N}/sensor/camera0/camera_info  (foxglove.CameraCalibration)

第一版只实现: 两路 RGB + 两路 IMU + 两套相机标定。
暂不处理磁编码器和 VIO 位姿。

与遁甲的关键差异:
  - 视频消息类型为 CompressedImage (非 CompressedVideo)，但 data 字段同为 H264 字节
  - IMU 消息类型为 IMUMeasurement (非 foxglove.Imu)，时间戳在 header.timestamp (int64 ns)
  - 双 robot (robot0/robot1)，各一套 camera + IMU
  - 标定含 T_b_c (body→camera 齐次变换) 和 equidistant 畸变模型
  - 视频时间戳也取自 header.timestamp (arnold.common.Header)，非顶层 Timestamp
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


# ---- topic 名称 ----
TOPIC_ROBOT0_CAMERA = "/robot0/sensor/camera0/compressed"
TOPIC_ROBOT1_CAMERA = "/robot1/sensor/camera0/compressed"
TOPIC_ROBOT0_IMU = "/robot0/sensor/imu"
TOPIC_ROBOT1_IMU = "/robot1/sensor/imu"
TOPIC_ROBOT0_CALIB = "/robot0/sensor/camera0/camera_info"
TOPIC_ROBOT1_CALIB = "/robot1/sensor/camera0/camera_info"

# ---- 映射表 ----
CAMERA_TOPICS = {
    "robot0": TOPIC_ROBOT0_CAMERA,
    "robot1": TOPIC_ROBOT1_CAMERA,
}
IMU_TOPICS = {
    "robot0": TOPIC_ROBOT0_IMU,
    "robot1": TOPIC_ROBOT1_IMU,
}
CALIB_TOPICS = {
    "robot0": TOPIC_ROBOT0_CALIB,
    "robot1": TOPIC_ROBOT1_CALIB,
}


def _open_mcap(mcap_path: str):
    """打开 MCAP 文件，返回 (reader, file_handle)。"""
    path = Path(mcap_path)
    if not path.exists():
        raise FileNotFoundError(f"MCAP 文件不存在: {mcap_path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {mcap_path}")
    fh = open(str(path), "rb")
    reader = make_reader(fh, decoder_factories=[DecoderFactory()])
    return reader, fh


# ================================================================
# 标定
# ================================================================

def read_calibration(mcap_path: str, topic: str) -> dict:
    """从 MCAP CameraCalibration 消息提取相机标定。

    UMI 的 CameraCalibration 包含:
      - width, height, distortion_model ("equidistant"), D[4]
      - K[9], R[9], P[12]
      - T_b_c: body→camera 齐次变换
      - frame_id

    Args:
        mcap_path: MCAP 文件路径
        topic: 标定 topic，如 /robot0/sensor/camera0/camera_info
    """
    reader, fh = _open_mcap(mcap_path)
    try:
        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if channel.topic == topic:
                return {
                    "width": decoded.width,
                    "height": decoded.height,
                    "frame_id": decoded.frame_id,
                    "distortion_model": decoded.distortion_model,
                    "K": list(decoded.K),
                    "D": list(decoded.D),
                    "R": list(decoded.R) if decoded.R else [],
                    "P": list(decoded.P) if decoded.P else [],
                    "T_b_c": list(decoded.T_b_c) if decoded.T_b_c else [],
                }
        raise ValueError(f"MCAP 中没有 {topic} 消息")
    finally:
        fh.close()


# ================================================================
# 视频文件 (color_000000.mkv 等价)
# ================================================================

def reconstruct_video(
    mcap_path: str,
    topic: str,
    output_path: str | None = None,
) -> str:
    """从 MCAP CompressedImage 消息提取 H264 视频。

    UMI 视频为 CompressedImage (format="h264")，每消息一帧。
    先尝试拼接原始 H264 字节 → ffmpeg 重封装为 .mp4；
    若 ffmpeg 不可用，则回退为原始 .h264 文件（OpenCV 可直读）。

    Args:
        mcap_path: MCAP 文件路径
        topic: 视频 topic
        output_path: 输出 .mp4 路径（会尝试改为 .h264 若 ffmpeg 不可用）

    Returns:
        写入的视频文件路径（.mp4 或 .h264）
    """
    reader, fh = _open_mcap(mcap_path)
    try:
        if output_path is None:
            robot_id = "robot0" if "robot0" in topic else "robot1"
            output_path = str(
                Path(mcap_path).parent
                / f"{Path(mcap_path).stem}.{robot_id}_camera0.cache.mp4"
            )

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

        # Step 2: ffmpeg 重封装 → .mp4（若可用）
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", raw_h264, "-c", "copy", output_path,
                ],
                capture_output=True, text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
            # ffmpeg 成功 → 清理 raw .h264
            try:
                os.remove(raw_h264)
            except OSError:
                pass
            return output_path
        except (FileNotFoundError, RuntimeError):
            # ffmpeg 不可用 → 用 OpenCV 重编码为 .mp4（可随机寻帧）
            return _remux_with_opencv(raw_h264, output_path)
    finally:
        fh.close()


def _remux_with_opencv(h264_path: str, mp4_path: str) -> str:
    """用 OpenCV 将 raw .h264 重编码为 .mp4。

    raw .h264 不支持 CAP_PROP_POS_FRAMES 寻帧，
    必须重编码为容器格式后才能被 transcode_rgb 正常裁剪。

    Returns:
        .mp4 文件路径
    """
    import cv2

    cap = cv2.VideoCapture(h264_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开 {h264_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        # 回退 mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV VideoWriter 无法创建 {mp4_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    if frame_count == 0:
        raise RuntimeError("没有解码到任何帧")

    # 清理 raw .h264
    try:
        os.remove(h264_path)
    except OSError:
        pass

    return mp4_path


def get_video_for_topic(mcap_path: str, topic: str) -> str:
    """返回指定 topic 的 .mp4 缓存路径，不存在则重建。

    缓存为 .mp4 容器（ffmpeg 无损重封装，或 OpenCV 重编码回退）。
    """
    robot_id = "robot0" if "robot0" in topic else "robot1"
    cache_path = str(
        Path(mcap_path).parent
        / f"{Path(mcap_path).stem}.{robot_id}_camera0.cache.mp4"
    )
    if Path(cache_path).exists():
        return cache_path
    return reconstruct_video(mcap_path, topic, cache_path)


def get_session_id(dataset_path: str) -> str:
    """从文件名推导 session_id。"""
    stem = Path(dataset_path).stem
    return f"umi_{stem}"


# ================================================================
# 统一 Session 读取
# ================================================================

def read_session(dataset_path: str):
    """统一读取 UMI Session 全部流数据。

    一次扫描 MCAP，收集 robot0 + robot1 的:
      - CompressedImage 视频帧 (h264)
      - IMUMeasurement 数据
      - CameraCalibration (分辨率)

    Returns:
        Session 对象，包含:
          - video_streams: {"robot0_camera0": VideoStream, "robot1_camera0": VideoStream}
          - imu_streams:  {"robot0_imu": ImuStream, "robot1_imu": ImuStream}
    """
    from zpds_prepare.readers.session_model import Session, VideoStream, ImuStream

    reader, fh = _open_mcap(dataset_path)
    try:
        # ---- 收集所有流数据 ----
        cam_data: dict[str, dict] = {
            "robot0": {"frames": [], "has_calib": False, "width": 0, "height": 0},
            "robot1": {"frames": [], "has_calib": False, "width": 0, "height": 0},
        }
        imu_data: dict[str, list[dict]] = {"robot0": [], "robot1": []}

        for _schema, channel, msg, decoded in reader.iter_decoded_messages():
            topic = channel.topic

            # ---- 视频消息 (CompressedImage, h264) ----
            if topic == TOPIC_ROBOT0_CAMERA:
                cam_data["robot0"]["frames"].append({
                    "seq": len(cam_data["robot0"]["frames"]),
                    "timestamp_ns": decoded.header.timestamp,
                    "log_time_ns": msg.log_time,
                    "publish_time_ns": msg.publish_time,
                    "h264_size": len(decoded.data),
                })
            elif topic == TOPIC_ROBOT1_CAMERA:
                cam_data["robot1"]["frames"].append({
                    "seq": len(cam_data["robot1"]["frames"]),
                    "timestamp_ns": decoded.header.timestamp,
                    "log_time_ns": msg.log_time,
                    "publish_time_ns": msg.publish_time,
                    "h264_size": len(decoded.data),
                })

            # ---- 标定消息 (提取分辨率) ----
            elif topic == TOPIC_ROBOT0_CALIB:
                if not cam_data["robot0"]["has_calib"]:
                    cam_data["robot0"]["width"] = decoded.width
                    cam_data["robot0"]["height"] = decoded.height
                    cam_data["robot0"]["has_calib"] = True
            elif topic == TOPIC_ROBOT1_CALIB:
                if not cam_data["robot1"]["has_calib"]:
                    cam_data["robot1"]["width"] = decoded.width
                    cam_data["robot1"]["height"] = decoded.height
                    cam_data["robot1"]["has_calib"] = True

            # ---- IMU 消息 (IMUMeasurement) ----
            elif topic == TOPIC_ROBOT0_IMU:
                imu_data["robot0"].append({
                    "timestamp_ns": decoded.header.timestamp,
                    "ax": decoded.linear_acceleration.x,
                    "ay": decoded.linear_acceleration.y,
                    "az": decoded.linear_acceleration.z,
                    "gx": decoded.angular_velocity.x,
                    "gy": decoded.angular_velocity.y,
                    "gz": decoded.angular_velocity.z,
                })
            elif topic == TOPIC_ROBOT1_IMU:
                imu_data["robot1"].append({
                    "timestamp_ns": decoded.header.timestamp,
                    "ax": decoded.linear_acceleration.x,
                    "ay": decoded.linear_acceleration.y,
                    "az": decoded.linear_acceleration.z,
                    "gx": decoded.angular_velocity.x,
                    "gy": decoded.angular_velocity.y,
                    "gz": decoded.angular_velocity.z,
                })

        # ---- 计算 robot0 camera 元数据 (用作 session 主元数据) ----
        r0_frames = cam_data["robot0"]["frames"]
        if r0_frames:
            r0_ts = [f["timestamp_ns"] for f in r0_frames]
            if len(r0_ts) >= 2:
                intervals = np.diff(np.sort(r0_ts))
                median_ns = np.median(intervals)
                fps = 1e9 / median_ns if median_ns > 0 else 30.0
            else:
                fps = 30.0
            frame_count = len(r0_ts)
            width = cam_data["robot0"]["width"] or 640
            height = cam_data["robot0"]["height"] or 480
        else:
            fps = 30.0
            frame_count = 0
            width = 640
            height = 480

        # ---- 计算 IMU 采样率 (基于 robot0) ----
        r0_imu_rows = imu_data["robot0"]
        if len(r0_imu_rows) >= 2:
            imu_ts_sorted = np.sort([r["timestamp_ns"] for r in r0_imu_rows])
            imu_intervals = np.diff(imu_ts_sorted)
            imu_median_ns = np.median(imu_intervals)
            imu_rate = 1e9 / imu_median_ns if imu_median_ns > 0 else 200.0
        else:
            imu_rate = 200.0
    finally:
        fh.close()

    # ---- 构建 meta ----
    meta = {
        "device": "UMI",
        "fps": round(fps, 1),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "imu_sample_rate": round(imu_rate, 1),
    }

    # ---- 构建 video_streams ----
    video_streams: dict[str, VideoStream] = {}
    for robot_id in ["robot0", "robot1"]:
        frames = cam_data[robot_id]["frames"]
        if not frames:
            continue
        ts_list = [f["timestamp_ns"] for f in frames]
        if len(ts_list) >= 2:
            intervals = np.diff(np.sort(ts_list))
            median_ns = np.median(intervals)
            cam_fps = 1e9 / median_ns if median_ns > 0 else fps
        else:
            cam_fps = fps

        stream_id = f"{robot_id}_camera0"
        topic = CAMERA_TOPICS[robot_id]
        video_path = get_video_for_topic(dataset_path, topic)

        video_streams[stream_id] = VideoStream(
            stream_id=stream_id,
            timestamps_ns=ts_list,
            index_frames=frames,
            video_path=video_path,
            fps=round(cam_fps, 1),
            width=cam_data[robot_id]["width"],
            height=cam_data[robot_id]["height"],
            frame_count=len(frames),
        )

    # ---- 构建 imu_streams ----
    imu_streams: dict[str, ImuStream] = {}
    for robot_id in ["robot0", "robot1"]:
        rows = imu_data[robot_id]
        if not rows:
            continue
        stream_id = f"{robot_id}_imu"
        imu_df = pd.DataFrame(rows)
        # 计算每路 IMU 的实际采样率
        if len(rows) >= 2:
            ts_arr = np.sort([r["timestamp_ns"] for r in rows])
            imu_intervals = np.diff(ts_arr)
            imu_median_ns = np.median(imu_intervals)
            rate = 1e9 / imu_median_ns if imu_median_ns > 0 else imu_rate
        else:
            rate = imu_rate

        imu_streams[stream_id] = ImuStream(
            stream_id=stream_id,
            dataframe=imu_df,
            sample_rate_hz=round(rate, 1),
        )

    return Session(
        session_id=get_session_id(dataset_path),
        source_path=dataset_path,
        meta=meta,
        video_streams=video_streams,
        imu_streams=imu_streams,
    )


__all__ = [
    "CAMERA_TOPICS", "CALIB_TOPICS", "IMU_TOPICS",
    "TOPIC_ROBOT0_CAMERA", "TOPIC_ROBOT1_CAMERA",
    "TOPIC_ROBOT0_IMU", "TOPIC_ROBOT1_IMU",
    "TOPIC_ROBOT0_CALIB", "TOPIC_ROBOT1_CALIB",
    "read_calibration",
    "get_video_for_topic", "reconstruct_video",
    "get_session_id",
    "read_session",
]
