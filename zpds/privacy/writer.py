"""脱敏产物写出：redacted.mp4 + redaction_manifest.parquet + run_summary.json。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from zpds.privacy.contracts import FrameRedactionRecord
from zpds.privacy.schemas import PrivacyRunManifest


def write_redacted_video(
    records: list[FrameRedactionRecord],
    output_path: str | Path,
    fps: float = 30.0,
    codec: str = "avc1",
) -> Path:
    """将脱敏帧写出为 MP4 视频。

    Args:
        records: 逐帧记录列表（需含 redacted_frame）。无脱敏的帧跳过（写原帧）。
        output_path: 输出路径。
        fps: 输出帧率。
        codec: FourCC 编码器（默认 avc1 = H.264）。

    Returns:
        输出文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("records 不能为空")

    # 确定尺寸
    first_frame = records[0].redacted_frame
    if first_frame is None:
        raise ValueError("第一帧 redacted_frame 为 None，无法确定视频尺寸")
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        # fallback to mp4v
        fallback_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fallback_fourcc, fps, (w, h))

    try:
        for record in records:
            frame = record.redacted_frame
            if frame is None:
                continue
            # 确保 BGR（cv2.VideoWriter 期望）
            if frame.shape[-1] == 3:
                writer.write(frame)
    finally:
        writer.release()

    return output_path


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
