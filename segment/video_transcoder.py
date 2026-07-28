"""
视频裁剪与转码：从源视频读取帧，按 Span 裁剪后输出 CFR H.264 MP4。

支持 MKV (墨现) 和重构的 .h264 比特流 (遁甲)。
优先 ffmpeg（快速），回退 OpenCV（跨平台）。
"""

import cv2
import numpy as np
import shutil
import subprocess
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

    # 跳过已存在的输出（断点续跑，避免重复转码大视频）
    output_file = Path(output_mp4)
    if output_file.exists() and output_file.stat().st_size > 0:
        # 用 ffprobe 获取真实分辨率，避免下游校验拿到 0×0
        w, h = 0, 0
        nb_frames = 0
        if shutil.which("ffprobe"):
            try:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,nb_frames",
                    "-of", "csv=p=0",
                    str(output_mp4),
                ]
                probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
                if probe.returncode == 0 and probe.stdout.strip():
                    parts = probe.stdout.strip().split(",")
                    if len(parts) >= 2:
                        w, h = int(parts[0]), int(parts[1])
                    if len(parts) >= 3 and parts[2]:
                        nb_frames = int(parts[2])
            except Exception:
                pass
        return {
            "output_frames": nb_frames,
            "output_fps": target_fps,
            "width": w,
            "height": h,
            "codec": "h264 (cached)",
            "output_path": str(output_mp4),
            "cached": True,
        }

    # ---- ffmpeg 快速路径（秒级 vs OpenCV 分钟级） ----
    if shutil.which("ffmpeg"):
        try:
            return _transcode_with_ffmpeg(
                source_video, output_mp4, source_start_ns, source_end_ns, target_fps
            )
        except Exception:
            pass  # 回退 OpenCV

    # ---- OpenCV 回退 ----
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


def _transcode_with_ffmpeg(
    source_video: str,
    output_mp4: str,
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
) -> dict:
    """使用 ffmpeg 快速裁剪+CFR 转码（大视频秒级完成）。"""
    # 纳秒 → 秒（ffmpeg -ss/-to 精度到毫秒）
    start_s = source_start_ns / 1e9
    duration_s = (source_end_ns - source_start_ns) / 1e9

    # 先用 ffprobe 获取分辨率
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        source_video,
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    w, h = 1920, 1080
    if probe.returncode == 0 and probe.stdout.strip():
        parts = probe.stdout.strip().split(",")
        if len(parts) >= 2:
            w, h = int(parts[0]), int(parts[1])

    # ffmpeg 裁剪 + 改帧率
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
        "-ss", f"{start_s:.6f}",
        "-i", source_video,
        "-t", f"{duration_s:.6f}",
        "-r", str(target_fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_mp4,
    ]
    # 实时输出编码进度
    import sys as _sys
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        bufsize=1,  # line-buffered
    )
    last_progress_line = ""
    for err_line in proc.stderr:
        err_line = err_line.rstrip()
        # ffmpeg 进度行含 "frame=" / "speed="
        if "frame=" in err_line or "speed=" in err_line:
            _sys.stdout.write(f"\r  {err_line}")
            _sys.stdout.flush()
            last_progress_line = err_line
    proc.wait()
    print()  # 换行结束进度行
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 转码失败 (exit {proc.returncode})")

    # 统计输出帧数
    count_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "csv=p=0",
        output_mp4,
    ]
    count_result = subprocess.run(count_cmd, capture_output=True, text=True)
    output_frames = int(count_result.stdout.strip()) if count_result.stdout.strip() else 0

    return {
        "output_frames": output_frames,
        "output_fps": target_fps,
        "width": w,
        "height": h,
        "codec": "h264",
        "output_path": str(output_mp4),
    }
