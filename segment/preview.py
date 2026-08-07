"""为清洗结果 MP4 生成前端预览压缩文件（保留原视频）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2


def _require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError(
            "未找到 ffmpeg，无法生成预览视频；请安装 ffmpeg 后重试"
        )
    return executable


def _probe_video(path: Path) -> dict[str, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"预览视频无法解码: {path}")
    try:
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"预览视频元数据非法: {path}")
    return {"width": width, "height": height, "fps": fps}


def create_preview(
    source_mp4: str | Path,
    preview_mp4: str | Path,
    *,
    max_width: int = 1280,
    crf: int = 28,
    preset: str = "veryfast",
) -> dict[str, float | int]:
    """把 source_mp4 压缩为 preview_mp4（H.264 CRF + 可选降分辨率）。

    超过 ``max_width`` 时按比例缩放（高度保持偶数），否则保持原分辨率，
    不放大。返回预览视频的宽度/高度/fps 与源文件字节数。
    """
    source = Path(source_mp4).expanduser().resolve()
    preview = Path(preview_mp4).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"源视频不存在: {source}")
    if not 0 < max_width < 100_000:
        raise ValueError("max_width 必须是正整数")
    if not 0 <= crf <= 51:
        raise ValueError("crf 必须在 0~51 范围内")

    ffmpeg = _require_ffmpeg()
    preview.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale='min({max_width},iw)':-2"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-nostats",
        "-i",
        str(source),
        "-vf",
        scale,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(preview),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"预览压缩失败: {source} → {preview}; "
            f"ffmpeg stderr: {result.stderr.strip()[:500]}"
        )
    if not preview.is_file() or preview.stat().st_size == 0:
        raise RuntimeError(f"预览文件未生成或为空: {preview}")
    probe = _probe_video(preview)
    probe["source_size_bytes"] = source.stat().st_size
    probe["preview_size_bytes"] = preview.stat().st_size
    return probe


__all__ = ["create_preview"]
