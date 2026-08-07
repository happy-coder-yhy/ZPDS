"""
视频裁剪与转码：从源视频读取帧，按 Span 裁剪后输出 CFR H.264 MP4。

支持 MKV (墨现) 和重构的 .h264 比特流 (遁甲)。
优先 ffmpeg（快速），回退 OpenCV（跨平台）。
"""

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


def _remove_incomplete_output(path: Path) -> None:
    """删除本次转码产生的空容器，避免后续被误判为有效缓存。"""
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _probe_video(path: Path) -> dict | None:
    """探测一个视频是否真实可解码；ffprobe 缺失时回退 OpenCV。"""
    if not path.is_file() or path.stat().st_size <= 0:
        return None

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            probe_cmd = [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames",
                "-of", "csv=p=0",
                str(path),
            ]
            probe = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                parts = probe.stdout.strip().split(",")
                width = int(parts[0]) if len(parts) >= 1 else 0
                height = int(parts[1]) if len(parts) >= 2 else 0
                frame_count = (
                    int(parts[2])
                    if len(parts) >= 3 and parts[2].isdigit()
                    else 0
                )
                if width > 0 and height > 0 and frame_count > 0:
                    return {
                        "width": width,
                        "height": height,
                        "frame_count": frame_count,
                    }
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        readable, frame = cap.read()
        if (
            not readable
            or frame is None
            or width <= 0
            or height <= 0
            or frame_count <= 0
        ):
            return None
        return {
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }
    finally:
        cap.release()


def transcode_rgb(
    source_video: str,
    output_mp4: str,
    source_start_ns: int,
    source_end_ns: int,
    index_frames: list[dict],
    target_fps: float = 30.0,
    frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    use_cache: bool = True,
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
        frame_transform: 可选逐帧确定性变换。提供时使用 OpenCV 写出路径。
        use_cache: 帧数匹配时复用已有产物（默认 True）。
            脱敏流程必须传 False——已有产物可能是上次脱敏的重编码版本
            （二次编码会抹平人脸细节导致脱敏漏检）。

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

    span_frames = [
        frame
        for frame in index_frames
        if source_start_ns <= int(frame["timestamp_ns"]) <= source_end_ns
    ]
    if not span_frames:
        raise ValueError("Span 内没有帧")
    frame_interval_ns = int(1_000_000_000 / target_fps)
    duration_ns = source_end_ns - source_start_ns
    expected_frames = (
        duration_ns + frame_interval_ns - 1
    ) // frame_interval_ns

    # 只复用可解码且帧数精确匹配 sample map 的缓存。
    output_file = Path(output_mp4)
    if (
        use_cache
        and frame_transform is None
        and output_file.exists()
        and output_file.stat().st_size > 0
    ):
        cached_probe = _probe_video(output_file)
        if (
            cached_probe is not None
            and cached_probe["frame_count"] == expected_frames
        ):
            return {
                "output_frames": cached_probe["frame_count"],
                "output_fps": target_fps,
                "width": cached_probe["width"],
                "height": cached_probe["height"],
                "codec": "h264 (cached)",
                "output_path": str(output_mp4),
                "cached": True,
            }
        _remove_incomplete_output(output_file)

    # ---- ffmpeg 快速路径（秒级 vs OpenCV 分钟级） ----
    ffmpeg_error: Exception | None = None
    if frame_transform is None and shutil.which("ffmpeg"):
        source_timestamps = np.asarray(
            [int(frame["timestamp_ns"]) for frame in span_frames],
            dtype=np.int64,
        )
        positive_intervals = np.diff(source_timestamps)
        positive_intervals = positive_intervals[positive_intervals > 0]
        source_fps = (
            1_000_000_000 / float(np.median(positive_intervals))
            if len(positive_intervals)
            else target_fps
        )
        try:
            return _transcode_with_ffmpeg(
                source_video,
                output_mp4,
                source_start_ns,
                source_end_ns,
                target_fps,
                start_frame=min(int(frame["seq"]) for frame in span_frames),
                end_frame=max(int(frame["seq"]) for frame in span_frames) + 1,
                source_fps=source_fps,
                expected_frames=int(expected_frames),
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            ffmpeg_error = exc

    # ---- OpenCV 帧级路径（去畸变）或回退 ----
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        cap.release()
        detail = f"；ffmpeg 错误: {ffmpeg_error}" if ffmpeg_error else ""
        raise RuntimeError(f"无法解码源视频 {source_video}{detail}")
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if src_width <= 0 or src_height <= 0:
        cap.release()
        detail = f"；ffmpeg 错误: {ffmpeg_error}" if ffmpeg_error else ""
        raise RuntimeError(f"源视频尺寸无效: {source_video}{detail}")

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
    span_timestamps = np.array([f["timestamp_ns"] for f in span_frames], dtype=np.int64)

    if len(span_frames) == 0:
        cap.release()
        writer.release()
        raise ValueError("Span 内没有帧")

    # 生成 CFR 输出时间轴
    frame_interval_ns = int(1_000_000_000 / target_fps)
    segment_duration_ns = source_end_ns - source_start_ns

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
            if frame_transform is not None:
                frame = frame_transform(frame)
            writer.write(frame)
            total_output += 1
        else:
            # 读取失败，写入上一帧或黑帧
            pass

        output_time_ns += frame_interval_ns

    cap.release()
    writer.release()

    output_probe = _probe_video(output_file)
    if total_output <= 0 or output_probe is None:
        _remove_incomplete_output(output_file)
        detail = f"；ffmpeg 错误: {ffmpeg_error}" if ffmpeg_error else ""
        raise RuntimeError(
            f"视频转码未生成任何可解码帧: {source_video}{detail}"
        )

    return {
        "output_frames": output_probe["frame_count"],
        "output_fps": target_fps,
        "width": output_probe["width"],
        "height": output_probe["height"],
        "codec": codec,
        "output_path": str(output_mp4),
    }


def _transcode_with_ffmpeg(
    source_video: str,
    output_mp4: str,
    source_start_ns: int,
    source_end_ns: int,
    target_fps: float = 30.0,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    source_fps: float | None = None,
    expected_frames: int | None = None,
) -> dict:
    """使用 ffmpeg 按源帧位置裁剪并输出精确 CFR 帧数。"""
    duration_s = (source_end_ns - source_start_ns) / 1e9

    # MCAP 重封装视频的容器 FPS 可能不可信，按真实源时间戳恢复 PTS。
    filters: list[str] = []
    if start_frame is not None and end_frame is not None:
        filters.append(
            f"trim=start_frame={start_frame}:end_frame={end_frame}"
        )
    if source_fps is not None and source_fps > 0:
        filters.append(f"setpts=N/({source_fps:.9f}*TB)")
    else:
        filters.append("setpts=PTS-STARTPTS")
    filters.append(f"fps={target_fps:.9f}")
    filters.append(
        f"tpad=stop_mode=clone:stop_duration={duration_s:.9f}"
    )

    # 转码本身不依赖 ffprobe。
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg 不可用")
    ffmpeg_cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "info", "-stats",
        "-i", source_video,
        "-vf", ",".join(filters),
    ]
    if expected_frames is not None:
        ffmpeg_cmd.extend(["-frames:v", str(expected_frames)])
    else:
        ffmpeg_cmd.extend(["-t", f"{duration_s:.6f}"])
    ffmpeg_cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_mp4,
    ])
    # 实时输出编码进度
    import sys as _sys
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,  # line-buffered
    )
    stderr_tail: list[str] = []
    for err_line in proc.stderr:
        err_line = err_line.rstrip()
        if err_line:
            stderr_tail.append(err_line)
            stderr_tail = stderr_tail[-20:]
        # ffmpeg 进度行含 "frame=" / "speed="
        if "frame=" in err_line or "speed=" in err_line:
            _sys.stdout.write(f"\r  {err_line}")
            _sys.stdout.flush()
    proc.wait()
    print()  # 换行结束进度行
    if proc.returncode != 0:
        _remove_incomplete_output(Path(output_mp4))
        detail = "\n".join(stderr_tail[-5:])
        raise RuntimeError(
            f"ffmpeg 转码失败 (exit {proc.returncode}): {detail}"
        )

    output_probe = _probe_video(Path(output_mp4))
    if output_probe is None:
        _remove_incomplete_output(Path(output_mp4))
        raise RuntimeError("ffmpeg 返回成功，但输出视频没有可解码帧")

    return {
        "output_frames": output_probe["frame_count"],
        "output_fps": target_fps,
        "width": output_probe["width"],
        "height": output_probe["height"],
        "codec": "h264",
        "output_path": str(output_mp4),
    }
