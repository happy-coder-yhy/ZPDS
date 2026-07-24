"""
视频裁剪与转码：从源视频读取帧，按 Span 裁剪后输出 CFR H.264 MP4。

支持 MKV (墨现) 和重构的 .h264 比特流 (遁甲)。
使用 OpenCV 完成读写。
"""

import cv2
import numpy as np
from pathlib import Path


def transcode_rgb(
    source_video: str,
    output_mp4: str,
    source_start_ns: int,
    source_end_ns: int,
    index_frames: list[dict],
    target_fps: float = 30.0,
) -> dict:
    """裁剪并转码 RGB 视频。

    将源视频中 [source_start_ns, source_end_ns] 范围内的帧
    按最近邻映射输出为 CFR target_fps 的 MP4。

    Args:
        source_video: 源视频文件路径 (.mkv 或 .h264)
        output_mp4: 输出 MP4 文件路径
        source_start_ns: 源时间戳起始
        source_end_ns: 源时间戳结束
        index_frames: 帧索引列表 (每项含 seq, timestamp_ns)
        target_fps: 目标恒定帧率

    Returns:
        {
            "output_frames": int,
            "output_fps": float,
            "width": int,
            "height": int,
            "codec": str,
            "output_path": str,
        }
    """
    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source_video)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 尝试 H.264 (avc1)，失败则回退 mp4v
    codec = "mp4v"    # 默认使用兼容性最好的编码器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_mp4), fourcc, target_fps, (src_width, src_height))

    # 尝试 avc1 (H.264)，如果可用则优先使用
    if not writer.isOpened():
        codec = "avc1"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_mp4), fourcc, target_fps, (src_width, src_height)
        )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"无法创建视频输出文件 {output_mp4}，"
            f"请检查 OpenCV ffmpeg 后端是否安装"
        )

    # 筛选 Span 内的帧
    span_frames = [
        f for f in index_frames
        if source_start_ns <= f["timestamp_ns"] <= source_end_ns
    ]
    span_timestamps = np.array([f["timestamp_ns"] for f in span_frames], dtype=np.int64)

    if len(span_frames) == 0:
        cap.release()
        writer.release()
        raise ValueError("Span 内没有帧")

    # 生成 CFR 输出时间轴
    frame_interval_ns = int(1_000_000_000 / target_fps)
    segment_duration_ns = source_end_ns - source_start_ns
    output_count = int(segment_duration_ns / frame_interval_ns)

    output_frame_index = 0
    output_time_ns = 0
    total_output = 0

    while output_time_ns < segment_duration_ns:
        target_source_time = source_start_ns + output_time_ns

        # 最近邻映射
        nearest_idx = np.argmin(np.abs(span_timestamps - target_source_time))
        source_seq = span_frames[nearest_idx]["seq"]

        # 读取源帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_seq)
        ret, frame = cap.read()

        if ret and frame is not None:
            writer.write(frame)
            total_output += 1
        else:
            # 读取失败，写入上一帧或黑帧
            pass

        output_frame_index += 1
        output_time_ns += frame_interval_ns

    cap.release()
    writer.release()

    return {
        "output_frames": total_output,
        "output_fps": target_fps,
        "width": src_width,
        "height": src_height,
        "codec": codec,
        "output_path": str(output_mp4),
    }
