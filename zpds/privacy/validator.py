"""脱敏产物回读校验：可解码、帧数一致、字段合法、Raw 未修改。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd


def validate_video(output_path: str | Path, expected_frames: int) -> list[str]:
    """校验脱敏视频可解码且帧数正确。

    Returns:
        错误消息列表（空 = 通过）。
    """
    errors = []
    path = Path(output_path)
    if not path.is_file():
        errors.append(f"脱敏视频不存在: {path}")
        return errors

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        errors.append(f"无法解码脱敏视频: {path}")
        return errors

    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if actual_frames != expected_frames:
        errors.append(
            f"脱敏视频帧数不一致: 期望 {expected_frames}，实际 {actual_frames}"
        )
    return errors


def validate_manifest(
    manifest_path: str | Path,
    min_rows: int = 0,
) -> list[str]:
    """校验 manifest.parquet 可读且字段合法。"""
    errors = []
    path = Path(manifest_path)
    if not path.is_file():
        errors.append(f"Manifest 不存在: {path}")
        return errors

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        errors.append(f"无法读取 manifest: {exc}")
        return errors

    required_cols = {
        "session_id", "frame_index", "timestamp_ns", "kind",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "method", "category",
    }
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Manifest 缺少字段: {sorted(missing)}")

    if len(df) < min_rows:
        errors.append(f"Manifest 行数 ({len(df)}) < 最低要求 ({min_rows})")

    # 校验 bbox 范围
    if "bbox_x1" in df.columns:
        for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
            if (df[col] < 0).any() or (df[col] > 1).any():
                errors.append(f"Manifest {col} 超出 [0, 1] 范围")

    return errors


def validate_raw_unchanged(
    original_path: str | Path,
    original_sha256: str,
) -> list[str]:
    """校验原始文件 SHA-256 未变。"""
    errors = []
    path = Path(original_path)
    if not path.is_file():
        errors.append(f"原始文件不存在: {path}")
        return errors

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != original_sha256:
        errors.append(
            f"原始文件 SHA-256 已变更: 期望 {original_sha256[:16]}..., 实际 {actual[:16]}..."
        )
    return errors


def validate_run_summary(summary_path: str | Path) -> list[str]:
    """校验 run_summary.json 结构合法。"""
    errors = []
    path = Path(summary_path)
    if not path.is_file():
        errors.append(f"run_summary 不存在: {path}")
        return errors

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"无法解析 run_summary: {exc}")
        return errors

    required_keys = {"session_id", "producer", "version", "stats"}
    missing = required_keys - set(data.keys())
    if missing:
        errors.append(f"run_summary 缺少字段: {sorted(missing)}")

    if "stats" in data:
        stats = data["stats"]
        for key in ("total_frames", "frames_with_faces", "frames_with_text"):
            if key not in stats:
                errors.append(f"run_summary.stats 缺少 {key}")

    return errors


__all__ = [
    "validate_manifest",
    "validate_raw_unchanged",
    "validate_run_summary",
    "validate_video",
]
