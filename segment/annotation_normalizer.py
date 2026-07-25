"""
Hand-object 标注标准化 — Pickle → Parquet + 帧映射。

将 EPIC-KITCHENS-100 Pickle 标注转换为安全、稳定、可查询的 Parquet 格式，
并完成：
  原始标注帧 → 源视频时间 → Prepared Segment 输出帧

用法:
    from segment.annotation_normalizer import normalize_hand_objects

    df = normalize_hand_objects(
        annotation_stream=session.annotation_streams["hand_objects"],
        video_timestamps_ns=session.primary_video.timestamps_ns,
        sample_map=pd.read_parquet("maps/ego_rgb_sample_map.parquet"),
        source_start_ns=span_start,
        source_end_ns=span_end,
        video_width=456,
        video_height=256,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---- 帧号 → 时间戳 ----

def frame_to_timestamp(
    frame_index: int,
    timestamps_ns: list[int],
) -> int:
    """将 0-based 帧号转换为纳秒时间戳。

    Raises:
        IndexError: 帧号越界
    """
    if frame_index < 0 or frame_index >= len(timestamps_ns):
        raise IndexError(
            f"标注帧号越界：frame_index={frame_index}, "
            f"视频帧数={len(timestamps_ns)}"
        )
    return timestamps_ns[frame_index]


# ---- 最近邻输出帧映射 ----

def nearest_output_frame(
    source_timestamp_ns: int,
    output_source_timestamps: np.ndarray,
) -> int:
    """在输出帧的源时间戳数组中，找到最近邻的输出帧索引。

    Args:
        source_timestamp_ns: 标注帧的源时间戳 (ns)
        output_source_timestamps: 输出帧对应的源时间戳数组 (ns)

    Returns:
        output_frame_index
    """
    position = np.searchsorted(output_source_timestamps, source_timestamp_ns)

    candidates = []
    if position > 0:
        candidates.append(position - 1)
    if position < len(output_source_timestamps):
        candidates.append(position)

    if not candidates:
        return 0

    return int(min(
        candidates,
        key=lambda idx: abs(
            int(output_source_timestamps[idx]) - source_timestamp_ns
        ),
    ))


# ---- BBox 校验与裁剪 ----

def validate_bbox(
    x1: float, y1: float,
    x2: float, y2: float,
    width: int, height: int,
) -> bool:
    """检查 bbox 是否在有效像素范围内。

    Returns:
        True 如果 bbox 有效 (0 <= x1 < x2 <= width, 0 <= y1 < y2 <= height)
    """
    return (
        0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    )


def clip_bbox(
    x1: float, y1: float,
    x2: float, y2: float,
    width: int, height: int,
) -> tuple[float, float, float, float, bool]:
    """将 bbox 裁剪到 [0, width] × [0, height] 范围内。

    Returns:
        (clipped_x1, clipped_y1, clipped_x2, clipped_y2, was_clipped)
    """
    cx1 = max(0.0, min(x1, float(width)))
    cy1 = max(0.0, min(y1, float(height)))
    cx2 = max(0.0, min(x2, float(width)))
    cy2 = max(0.0, min(y2, float(height)))

    was_clipped = (
        cx1 != x1 or cy1 != y1
        or cx2 != x2 or cy2 != y2
    )

    return cx1, cy1, cx2, cy2, was_clipped


# ---- 主标准化函数 ----

def normalize_hand_objects(
    annotation_stream,          # AnnotationStream
    video_timestamps_ns: list[int],
    sample_map: pd.DataFrame,
    source_start_ns: int,
    source_end_ns: int,
    video_width: int,
    video_height: int,
) -> pd.DataFrame:
    """将 hand-object 标注标准化为 Parquet-ready DataFrame。

    每个 Pickle 记录展开为 1 条 hand 行 + (可选) 1 条 object 行。
    归一化坐标转为像素坐标，裁剪到 Segment 区间，映射到输出帧。

    Args:
        annotation_stream: 包含 hand_object_detection 记录的 AnnotationStream
        video_timestamps_ns: 原视频帧时间戳列表 (0-based 索引)
        sample_map: 输出帧 ↔ 源帧映射表 (含 output_frame_index, source_timestamp_ns 等)
        source_start_ns: Segment 源起始时间 (ns)
        source_end_ns: Segment 源结束时间 (ns)
        video_width: 视频帧宽 (像素)
        video_height: 视频帧高 (像素)

    Returns:
        DataFrame，字段:
          timestamp_ns, output_frame_index,
          source_timestamp_ns, source_frame_index, source_record_index,
          entity_type, entity_id, track_id, hand_side,
          bbox_x1, bbox_y1, bbox_x2, bbox_y2,
          confidence, interaction_state, linked_entity_id,
          original_bbox, bbox_was_clipped,
          mapping_method, mapping_error_ns,
          source_file
    """
    records = annotation_stream.records
    source_file = str(annotation_stream.source_path)

    if not records:
        return pd.DataFrame()

    # 输出帧源时间戳数组（用于最近邻映射）
    sm_ts = sample_map["source_timestamp_ns"].values.astype(np.int64)

    rows: list[dict] = []

    for rec_idx, rec in enumerate(records):
        frame_idx = rec["frame_index"]

        # ---- 源时间戳 ----
        try:
            src_ts = frame_to_timestamp(frame_idx, video_timestamps_ns)
        except IndexError:
            # 标注帧号越界 → 跳过
            continue

        # ---- Segment 裁剪 ----
        if src_ts < source_start_ns or src_ts >= source_end_ns:
            continue

        # ---- 输出帧映射 ----
        out_frame_idx = nearest_output_frame(src_ts, sm_ts)
        out_ts = int(sample_map.iloc[out_frame_idx]["output_timestamp_ns"])
        out_src_ts = int(sm_ts[out_frame_idx])
        mapping_error_ns = abs(src_ts - out_src_ts)

        # ---- 判断是否有物体交互 ----
        has_object = "object_bbox" in rec

        # ============================================================
        # Hand 实体
        # ============================================================
        hand_bbox_norm = rec["hand_bbox"]  # [x1, y1, x2, y2] 归一化
        hx1 = hand_bbox_norm[0] * video_width
        hy1 = hand_bbox_norm[1] * video_height
        hx2 = hand_bbox_norm[2] * video_width
        hy2 = hand_bbox_norm[3] * video_height

        hx1_c, hy1_c, hx2_c, hy2_c, clipped = clip_bbox(
            hx1, hy1, hx2, hy2, video_width, video_height
        )

        hand_row = {
            "timestamp_ns": out_ts,
            "output_frame_index": out_frame_idx,
            "source_timestamp_ns": src_ts,
            "source_frame_index": frame_idx,
            "source_record_index": rec_idx,
            "entity_type": "hand",
            "entity_id": f"hand_{rec_idx}",
            "track_id": None,
            "hand_side": None,
            "bbox_x1": hx1_c,
            "bbox_y1": hy1_c,
            "bbox_x2": hx2_c,
            "bbox_y2": hy2_c,
            "confidence": rec.get("hand_score", 0.0),
            "interaction_state": has_object,
            "linked_entity_id": f"object_{rec_idx}" if has_object else None,
            "original_bbox": [hx1, hy1, hx2, hy2],
            "bbox_was_clipped": clipped,
            "mapping_method": "nearest_timestamp",
            "mapping_error_ns": mapping_error_ns,
            "source_file": source_file,
        }
        rows.append(hand_row)

        # ============================================================
        # Object 实体 (可选)
        # ============================================================
        if has_object:
            obj_bbox_norm = rec["object_bbox"]
            ox1 = obj_bbox_norm[0] * video_width
            oy1 = obj_bbox_norm[1] * video_height
            ox2 = obj_bbox_norm[2] * video_width
            oy2 = obj_bbox_norm[3] * video_height

            ox1_c, oy1_c, ox2_c, oy2_c, o_clipped = clip_bbox(
                ox1, oy1, ox2, oy2, video_width, video_height
            )

            obj_row = {
                "timestamp_ns": out_ts,
                "output_frame_index": out_frame_idx,
                "source_timestamp_ns": src_ts,
                "source_frame_index": frame_idx,
                "source_record_index": rec_idx,
                "entity_type": "object",
                "entity_id": f"object_{rec_idx}",
                "track_id": None,
                "hand_side": None,
                "bbox_x1": ox1_c,
                "bbox_y1": oy1_c,
                "bbox_x2": ox2_c,
                "bbox_y2": oy2_c,
                "confidence": rec.get("object_score", 0.0),
                "interaction_state": True,
                "linked_entity_id": f"hand_{rec_idx}",
                "original_bbox": [ox1, oy1, ox2, oy2],
                "bbox_was_clipped": o_clipped,
                "mapping_method": "nearest_timestamp",
                "mapping_error_ns": mapping_error_ns,
                "source_file": source_file,
            }
            rows.append(obj_row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 确保列顺序
    column_order = [
        "timestamp_ns", "output_frame_index",
        "source_timestamp_ns", "source_frame_index", "source_record_index",
        "entity_type", "entity_id", "track_id", "hand_side",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "confidence", "interaction_state", "linked_entity_id",
        "original_bbox", "bbox_was_clipped",
        "mapping_method", "mapping_error_ns",
        "source_file",
    ]
    existing_cols = [c for c in column_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in column_order]
    df = df[existing_cols + extra_cols]

    return df


def write_annotation_parquet(
    df: pd.DataFrame,
    output_dir: str,
    stream_id: str = "hand_objects",
) -> str:
    """将标准化标注 DataFrame 写出为 Parquet。

    Args:
        df: normalize_hand_objects() 返回的 DataFrame
        output_dir: Prepared Segment 根目录
        stream_id: 流标识（用于文件命名）

    Returns:
        输出文件路径
    """
    from pathlib import Path

    annotations_dir = Path(output_dir) / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    output_path = annotations_dir / f"{stream_id}.parquet"

    # 将 object 类型的列转为合适的类型确保 Parquet 兼容
    df_out = df.copy()
    for col in ["original_bbox"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda v: str(v) if v is not None else None
            )

    df_out.to_parquet(str(output_path), index=False)
    return str(output_path)


__all__ = [
    "frame_to_timestamp",
    "nearest_output_frame",
    "validate_bbox",
    "clip_bbox",
    "normalize_hand_objects",
    "write_annotation_parquet",
]
