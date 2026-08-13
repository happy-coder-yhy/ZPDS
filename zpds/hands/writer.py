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
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zpds.hands.base import RawHandResult
from zpds.hands.schemas import HandObservation

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


def _observation_to_row(
    observation: HandObservation,
    prep_revision: str,
    checkpoint_sha256: str,
    config_sha256: str,
    run_meta: dict | None,
) -> dict:
    """将 Pipeline 的统一观测转换为 Hands V1 Parquet 行。"""
    handedness = observation.handedness.capitalize()
    return {
        "prep_revision": prep_revision,
        "segment_id": observation.segment_id,
        "video_stream_id": observation.video_stream_id,
        "output_frame_index": observation.output_frame_index,
        "timestamp_ns": observation.timestamp_ns,
        "source_frame_index": observation.source_frame_index,
        "source_timestamp_ns": observation.source_timestamp_ns,
        "detection_id": observation.detection_id,
        "handedness": handedness,
        "handedness_score": observation.handedness_score,
        "bbox_x1": observation.bbox_xyxy[0],
        "bbox_y1": observation.bbox_xyxy[1],
        "bbox_x2": observation.bbox_xyxy[2],
        "bbox_y2": observation.bbox_xyxy[3],
        "keypoints_2d": [
            [float(x), float(y)] for x, y in observation.keypoints_2d
        ],
        "keypoints_z_relative": [
            float(z) for z in observation.keypoints_z_relative
        ],
        "keypoints_any_clipped": observation.keypoints_any_clipped,
        "keypoints_clipped_count": observation.keypoints_clipped_count,
        "model_name": observation.model_name,
        "model_version": observation.model_version,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "backend_requested": (
            observation.backend_requested
            or (run_meta or {}).get("backend_requested", "")
        ),
        "backend_active": (
            observation.backend_active or (run_meta or {}).get("backend_active", "")
        ),
        "backend_fallback_used": bool(
            observation.backend_fallback_used
            or (run_meta or {}).get("backend_fallback_used", False)
        ),
        "backend_fallback_reason": (
            observation.backend_fallback_reason
            or (run_meta or {}).get("backend_fallback_reason", "")
        ),
        "backend_delegate": (
            observation.backend_delegate or (run_meta or {}).get("backend_delegate", "")
        ),
    }


def write_hand_observations(
    observations: Iterable[HandObservation],
    output_path: str | Path,
    *,
    prep_revision: str = "r0001",
    checkpoint_sha256: str = "",
    config_sha256: str = "",
    run_meta: dict | None = None,
) -> str:
    """将人员 A Pipeline 输出直接写为 ``hands_2d.parquet``。

    该接口与旧的 :func:`write_hands_parquet` 并存，避免破坏人员 C 已有调用方。
    """
    rows = [
        _observation_to_row(
            observation,
            prep_revision,
            checkpoint_sha256,
            config_sha256,
            run_meta,
        )
        for observation in observations
    ]
    frame = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    frame["source_frame_index"] = frame["source_frame_index"].astype("Int64")
    frame["source_timestamp_ns"] = frame["source_timestamp_ns"].astype("Int64")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return str(destination.resolve())


def compute_config_sha256(config: dict) -> str:
    """计算配置字典的 SHA-256 摘要（用于可追溯性）。"""
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def wilor_provenance(estimator: Any, config: dict | None = None) -> tuple[dict, dict]:
    """Build WiLoR parquet provenance and a serializable run summary.

    The estimator remains duck-typed so this module does not import optional WiLoR
    dependencies.  Frame-level fields on ``HandObservation`` carry per-frame
    provenance; ``run_meta`` is retained for the legacy writer interface and for
    frames without observations.
    """
    model_info = _as_plain_dict(getattr(estimator, "model_info", None))
    frame_stats = _as_plain_dict(getattr(estimator, "frame_stats", None))
    report_document: dict[str, Any] = {}
    build_run_report = getattr(estimator, "build_run_report", None)
    if callable(build_run_report):
        report = build_run_report()
        to_dict = getattr(report, "to_dict", None)
        report_document = to_dict() if callable(to_dict) else _as_plain_dict(report)

    fallback_used = int(frame_stats.get("fallback_used", 0)) > 0
    fallback_reason = _first_wilor_failure_reason(report_document)

    model_meta = {
        "model_name": "wilor",
        "model_version": model_info.get("model_version", ""),
        "checkpoint_sha256": model_info.get("checkpoint_sha256", ""),
        "config_sha256": compute_config_sha256(config) if config is not None else "",
    }
    run_meta = {
        "backend_requested": "wilor",
        "backend_active": "wilor",
        "backend_fallback_used": fallback_used,
        "backend_fallback_reason": fallback_reason,
        "backend_delegate": model_info.get("device", ""),
    }
    report_document.update(
        {
            "model": report_document.get("model", model_info),
            "session_statistics": frame_stats,
            "config_sha256": model_meta["config_sha256"],
        }
    )
    return {**model_meta, **run_meta}, report_document


def _first_wilor_failure_reason(report: dict[str, Any]) -> str:
    for error in report.get("errors", []):
        if isinstance(error, dict) and error.get("failure_reason"):
            return str(error["failure_reason"])
    return ""


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
