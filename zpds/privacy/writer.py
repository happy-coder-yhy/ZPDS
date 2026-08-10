"""脱敏产物写出：redacted.mp4 + redaction_manifest.parquet + run_summary.json。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from zpds.privacy.contracts import FrameRedactionRecord
from zpds.privacy.schemas import PrivacyRunManifest


def write_redacted_video(
    records: list[FrameRedactionRecord],
    output_path: str | Path,
    *,
    source_video: str | Path | None = None,
    fps: float | None = None,
    codec: str = "mp4v",
    recode_h264: bool = True,
) -> Path:
    """将脱敏帧写出为 MP4 视频（等长、无丢帧）。

    Args:
        records: 逐帧记录列表。``redacted_frame`` 仅在存在遮挡区域的帧
            非 None（PrivacyPipeline 产出语义）。
        output_path: 输出路径。
        source_video: 底帧源视频。提供时逐帧与 records 对齐：无遮挡帧
            （redacted_frame=None）补写源帧，保证输出与源等长；尺寸/帧率
            也以源为准。缺失时（旧行为）无遮挡帧被跳过、首帧必须有遮挡。
        fps: 输出帧率（仅 source_video 未提供时生效，默认 30.0）。
        codec: 中间产物 FourCC 编码器（默认 mp4v；avc1 在部分 OpenCV
            版本构造成功但写帧才失败，Windows 下不可靠）。
        recode_h264: 完成后用 ffmpeg 重编码为 H.264（libx264 crf 23，
            体积显著缩小）；失败或 ffmpeg 不可用时回退保留 mp4v 产物。

    Returns:
        输出文件路径。

    Raises:
        ValueError: records 为空；未提供 source_video 且首帧无遮挡
            （无法确定尺寸）；source_video 帧数不足（脱敏流与源不一致）。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("records 不能为空")

    # ---- 尺寸 / 帧率：source_video 优先（并作为无遮挡帧的底帧） ----
    src_cap = None
    if source_video is not None:
        src_cap = cv2.VideoCapture(str(source_video))
        if not src_cap.isOpened():
            raise ValueError(f"无法打开源视频: {source_video}")
        src_fps = float(src_cap.get(cv2.CAP_PROP_FPS))
        fps = src_fps if src_fps > 0 else (fps or 30.0)
        h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        first_frame = records[0].redacted_frame
        if first_frame is None:
            raise ValueError(
                "第一帧 redacted_frame 为 None 且未提供 source_video，"
                "无法确定视频尺寸"
            )
        h, w = first_frame.shape[:2]
        fps = fps or 30.0

    # ---- 先写 mp4v 中间产物（tmp），随后 h264 重编码或原地替换 ----
    tmp_path = output_path.with_name(f"{output_path.stem}.redacting.mp4")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        # fallback to mp4v
        writer = cv2.VideoWriter(
            str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    def _align_source() -> None:
        """推进源视频游标到与当前 record 对齐（帧即写即对齐）。"""
        if src_cap is None:
            return
        ok, _ = src_cap.read()
        if not ok:
            raise ValueError(
                f"源视频帧数不足（records={len(records)} 条），脱敏流与源不一致"
            )

    try:
        for record in records:
            frame = record.redacted_frame
            if frame is None:
                if src_cap is None:
                    continue  # 旧行为：无 source 时跳过（调用方需保证首帧有遮挡）
                ok, frame = src_cap.read()
                if not ok:
                    raise ValueError(
                        f"源视频帧数不足（records={len(records)} 条），"
                        "脱敏流与源不一致"
                    )
            else:
                _align_source()
            # 确保 BGR（cv2.VideoWriter 期望）
            if frame.shape[-1] == 3:
                writer.write(frame)
    finally:
        if src_cap is not None:
            src_cap.release()
        writer.release()

    # 体积优化：mp4v 中间产物 → H.264 重编码（libx264 crf 23）；
    # 成功则丢弃中间产物，失败/未启用则把 mp4v 产物移动到目标位置。
    if recode_h264 and _recode_h264(tmp_path, output_path):
        try:
            tmp_path.unlink()
        except OSError:
            pass
    else:
        os.replace(tmp_path, output_path)

    return output_path


def _recode_h264(src: Path, dst: Path) -> bool:
    """把 mp4v 中间产物重编码为 H.264 (libx264, crf 23)。

    OpenCV 的 mp4v 编码效率低（脱敏产物会从 22M 涨到 65M），
    用 ffmpeg 重编码可显著缩小体积。失败时返回 False（回退 mp4v）。
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def write_manifest(
    records: list[FrameRedactionRecord],
    manifest: PrivacyRunManifest,
    output_path: str | Path,
) -> Path:
    """将逐帧遮挡区域写出为 Parquet manifest。

    每行一个遮挡区域。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in records:
        for region in record.regions:
            x1, y1, x2, y2 = region.bbox_xyxy
            rows.append({
                "session_id": manifest.session_id,
                "source_uri": manifest.source_uri,
                "profile": manifest.profile,
                "producer": manifest.producer,
                "version": manifest.version,
                "config_hash": manifest.config_hash,
                "frame_index": record.frame_index,
                "timestamp_ns": record.timestamp_ns,
                "kind": region.kind,
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "method": region.method,
                "category": region.category,
                "confidence": region.confidence,
                "classifier_backend": record.pii_classifier_used,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    return output_path


def write_run_summary(
    manifest: PrivacyRunManifest,
    output_path: str | Path,
) -> Path:
    """写出 run_summary.json。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "session_id": manifest.session_id,
        "source_uri": manifest.source_uri,
        "profile": manifest.profile,
        "producer": manifest.producer,
        "version": manifest.version,
        "config_hash": manifest.config_hash,
        "face_model_hash": manifest.face_model_hash,
        "llm_endpoint": manifest.llm_endpoint,
        "stats": {
            "total_frames": manifest.total_frames,
            "frames_with_faces": manifest.frames_with_faces,
            "frames_with_text": manifest.frames_with_text,
            "total_face_regions": manifest.total_face_regions,
            "total_text_regions": manifest.total_text_regions,
            "pii_categories_found": list(manifest.pii_categories_found),
            "llm_available": manifest.llm_available,
        },
        "elapsed_seconds": manifest.elapsed_seconds,
        "error": manifest.error,
    }
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "write_manifest",
    "write_redacted_video",
    "write_run_summary",
]
