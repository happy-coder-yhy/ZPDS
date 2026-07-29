"""
Hands Parquet Writer。

将逐帧的 RawHandResult 列表写入符合 Hands V1 Schema 的 hands_2d.parquet。

Schema (一只手一行):
  prep_revision, segment_id, video_stream_id,
  output_frame_index, timestamp_ns, source_frame_index, source_timestamp_ns,
  detection_id, handedness, handedness_score,
  bbox_x1, bbox_y1, bbox_x2, bbox_y2,
  keypoints_2d, keypoints_z_relative,
  model_name, model_version, checkpoint_sha256, config_sha256
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from zpds.hands.base import RawHandResult

# ---- Parquet 列定义 ----
PARQUET_COLUMNS = [
    "prep_revision",
    "segment_id",
    "video_stream_id",
    "output_frame_index",
    "timestamp_ns",
    "source_frame_index",
    "source_timestamp_ns",
    "detection_id",
    "handedness",
    "handedness_score",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "keypoints_2d",
    "keypoints_z_relative",
    "keypoints_any_clipped",
    "keypoints_clipped_count",
    "model_name",
    "model_version",
    "checkpoint_sha256",
    "config_sha256",
    "backend_requested",
    "backend_active",
    "backend_fallback_used",
    "backend_fallback_reason",
    "backend_delegate",
]

# 手部关键点连线（MediaPipe Hand Landmarks 官方拓扑），用于可视化。
# 避免把 wrist 直接连到所有 MCP，预览里掌心会显得杂乱且难以判断错位。
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (9, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20), # 小指
    (0, 17),                               # 掌缘
]


def _dict_to_row(
    frame_meta: dict,
    hand: RawHandResult,
    detection_id: int,
    prep_revision: str,
    model_meta: dict | None = None,
    run_meta: dict | None = None,
) -> dict:
    """将单只手的检测结果展平为一行 Parquet 数据。"""
    kp_pixel = hand.keypoints.pixel          # [(x, y), ...] 21个
    kp_normalized = hand.keypoints.normalized  # [(x, y, z), ...] 21个

    row = {
        "prep_revision": prep_revision,
        "segment_id": frame_meta.get("segment_id", ""),
        "video_stream_id": frame_meta.get("video_stream_id", ""),
        "output_frame_index": int(frame_meta.get("output_frame_index", -1)),
        "timestamp_ns": int(frame_meta.get("timestamp_ns", 0)),
        "source_frame_index": frame_meta.get("source_frame_index"),
        "source_timestamp_ns": frame_meta.get("source_timestamp_ns"),
        "detection_id": detection_id,
        "handedness": hand.handedness,
        "handedness_score": float(hand.handedness_score),
        "bbox_x1": float(hand.bbox.x1),
        "bbox_y1": float(hand.bbox.y1),
        "bbox_x2": float(hand.bbox.x2),
        "bbox_y2": float(hand.bbox.y2),
        "keypoints_2d": [[float(x), float(y)] for (x, y) in kp_pixel],
        "keypoints_z_relative": [float(z) for (_, _, z) in kp_normalized],
        "keypoints_any_clipped": bool(hand.keypoints.any_clipped),
        "keypoints_clipped_count": int(hand.keypoints.clipped_count),
        "model_name": (model_meta or {}).get("model_name", ""),
        "model_version": (model_meta or {}).get("model_version", ""),
        "checkpoint_sha256": (model_meta or {}).get("checkpoint_sha256", ""),
        "config_sha256": (model_meta or {}).get("config_sha256", ""),
        "backend_requested": (run_meta or {}).get("backend_requested", ""),
        "backend_active": (run_meta or {}).get("backend_active", ""),
        "backend_fallback_used": bool((run_meta or {}).get("backend_fallback_used", False)),
        "backend_fallback_reason": (run_meta or {}).get("backend_fallback_reason", ""),
        "backend_delegate": (run_meta or {}).get("backend_delegate", ""),
    }
    return row


def write_hands_parquet(
    observations: list[dict],
    output_path: str,
    prep_revision: str = "r0001",
    model_meta: dict | None = None,
    run_meta: dict | None = None,
) -> str:
    """将逐帧检测结果写入 hands_2d.parquet。

    Args:
        observations: [
            {
                "frame_meta": {
                    "segment_id": str,
                    "video_stream_id": str,
                    "output_frame_index": int,
                    "timestamp_ns": int,
                    "source_frame_index": int | None,
                    "source_timestamp_ns": int | None,
                },
                "hands": [RawHandResult, ...],
            },
            ...
        ]
        output_path: 输出 .parquet 文件路径
        prep_revision: Prepared Revision 标识
        model_meta: {"model_name", "model_version", "checkpoint_sha256", "config_sha256"}
        run_meta: {"backend_requested", "backend_active", "backend_fallback_used",
            "backend_fallback_reason", "backend_delegate"}

    Returns:
        写入的文件路径
    """
    rows = []
    for obs in observations:
        frame_meta = obs["frame_meta"]
        hands = obs.get("hands", [])
        for det_id, hand in enumerate(hands):
            rows.append(
                _dict_to_row(frame_meta, hand, det_id, prep_revision, model_meta, run_meta)
            )

    df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)

    # 确保可空列的类型正确
    df["source_frame_index"] = df["source_frame_index"].astype("Int64")
    df["source_timestamp_ns"] = df["source_timestamp_ns"].astype("Int64")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out), index=False)

    return str(out.resolve())


def compute_config_sha256(config: dict) -> str:
    """计算配置字典的 SHA-256 摘要（用于可追溯性）。"""
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def estimator_provenance(estimator: Any, config: dict | None = None) -> tuple[dict, dict]:
    """Build Parquet and run-report provenance from a MediaPipe estimator.

    The estimator deliberately remains duck-typed here so Writer does not depend on
    a particular backend implementation.  This permits an alternative estimator to
    provide the same public ``model_info`` and ``backend_info`` properties.
    """
    model_info = _as_plain_dict(getattr(estimator, "model_info", None))
    backend_info = _as_plain_dict(getattr(estimator, "backend_info", None))
    session_stats = _as_plain_dict(getattr(estimator, "session_stats", None))

    model_meta = {
        "model_name": _model_name(backend_info.get("active_backend", "")),
        "model_version": _mediapipe_version(),
        "checkpoint_sha256": model_info.get("sha256", ""),
        "config_sha256": compute_config_sha256(config) if config is not None else "",
    }
    run_meta = {
        "backend_requested": backend_info.get("requested_backend", ""),
        "backend_active": backend_info.get("active_backend", ""),
        "backend_fallback_used": backend_info.get("fallback_used", False),
        "backend_fallback_reason": backend_info.get("fallback_reason", ""),
        "backend_delegate": backend_info.get("delegate", ""),
    }
    report = {
        "model": model_info,
        "backend": backend_info,
        "session_statistics": session_stats,
        "config_sha256": model_meta["config_sha256"],
    }
    return {**model_meta, **run_meta}, report


def write_hands_run_report(report: dict, output_path: str) -> str:
    """Write segment-level estimator provenance and timing statistics as JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(out.resolve())


def _as_plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Expected a dataclass or dict, got {type(value).__name__}")


def _model_name(active_backend: str) -> str:
    return f"mediapipe_{active_backend}" if active_backend else "mediapipe"


def _mediapipe_version() -> str:
    try:
        return version("mediapipe")
    except PackageNotFoundError:
        return ""
