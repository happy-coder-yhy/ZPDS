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
from pathlib import Path

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
    "model_name",
    "model_version",
    "checkpoint_sha256",
    "config_sha256",
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
        "model_name": (model_meta or {}).get("model_name", ""),
        "model_version": (model_meta or {}).get("model_version", ""),
        "checkpoint_sha256": (model_meta or {}).get("checkpoint_sha256", ""),
        "config_sha256": (model_meta or {}).get("config_sha256", ""),
    }
    return row


def write_hands_parquet(
    observations: list[dict],
    output_path: str,
    prep_revision: str = "r0001",
    model_meta: dict | None = None,
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

    Returns:
        写入的文件路径
    """
    rows = []
    for obs in observations:
        frame_meta = obs["frame_meta"]
        hands = obs.get("hands", [])
        for det_id, hand in enumerate(hands):
            rows.append(_dict_to_row(frame_meta, hand, det_id, prep_revision, model_meta))

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
        "model_name": observation.model_name,
        "model_version": observation.model_version,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
    }


def write_hand_observations(
    observations: Iterable[HandObservation],
    output_path: str | Path,
    *,
    prep_revision: str = "r0001",
    checkpoint_sha256: str = "",
    config_sha256: str = "",
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
