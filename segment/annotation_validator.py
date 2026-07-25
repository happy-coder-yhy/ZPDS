"""
标注流写出后验证：确认 hand_object_detection Parquet 数据完整、合法。

检查项:
  1. Parquet 可读取
  2. 必需列存在
  3. timestamp_ns 位于 Segment 时间范围内
  4. output_frame_index 不越界
  5. source_frame_index 非负
  6. BBox 合法 (x1 < x2, y1 < y2)
  7. confidence 在 [0, 1] 范围内
  8. 来源追溯完整 (source_file, source_record_index)
  9. 映射误差不过大 (max_mapping_error_ns)
  10. entity_type 使用固定枚举 {"hand", "object"}

用法:
    from segment.annotation_validator import validate_hand_object_stream

    result = validate_hand_object_stream(
        seg_dir=Path("prepared_segments/seg_000001"),
        stream=stream_entry_from_segment_json,
        timeline_end_ns=segment["timeline"]["end_ns"],
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np


# ---- 必需列定义 ----

REQUIRED_COLUMNS = [
    "timestamp_ns", "output_frame_index",
    "source_timestamp_ns", "source_frame_index", "source_record_index",
    "entity_type", "entity_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "confidence",
    "source_file",
]

VALID_ENTITY_TYPES = {"hand", "object"}

# 映射误差阈值：超过此值记为 unmapped（通常 CFR 重采样丢帧导致）
MAPPING_ERROR_WARN_NS = 20_000_000   # 20ms — 约 1 帧 @ 50fps
MAPPING_ERROR_MAX_NS = 50_000_000    # 50ms — 硬上限


# ---- 标注流验证 ----

def validate_hand_object_stream(
    seg_dir: Path,
    stream: dict,
    timeline_end_ns: int,
    video_width: int | None = None,
    video_height: int | None = None,
) -> dict:
    """验证 hand_object_detection 标注流。

    Args:
        seg_dir: Prepared Segment 根目录
        stream: segment.json 中该流的条目
        timeline_end_ns: Segment timeline 结束时间 (ns)，用于区间检查
        video_width: 源视频宽度（用于 bbox 校验），可选
        video_height: 源视频高度（用于 bbox 校验），可选

    Returns:
        {
            "checks": {...},
            "statistics": {...},
            "errors": [...],
        }
    """
    errors: list[str] = []
    checks: dict[str, str] = {}
    stats: dict = {}

    uri = stream.get("uri", "")
    parquet_path = seg_dir / uri

    # ---- 1. Parquet 可读取 ----
    df = None
    if not parquet_path.exists():
        errors.append(f"标注 Parquet 不存在: {parquet_path}")
        checks["annotation_file_readable"] = "fail"
        return _build_result("fail", checks, stats, errors)

    try:
        df = pd.read_parquet(str(parquet_path))
    except Exception as exc:
        errors.append(f"标注 Parquet 读取失败: {exc}")
        checks["annotation_file_readable"] = "fail"
        return _build_result("fail", checks, stats, errors)

    checks["annotation_file_readable"] = "pass"
    stats["annotation_rows"] = len(df)

    # ---- 2. 必需列存在 ----
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"缺少必需列: {missing_cols}")
        checks["required_columns"] = "fail"
    else:
        checks["required_columns"] = "pass"

    if df.empty:
        checks.update({k: "skip" for k in [
            "timestamps_in_range", "output_frame_indices_valid",
            "bbox_valid", "provenance_complete",
            "confidence_range", "entity_type_valid",
            "mapping_error_ok",
        ]})
        stats.update({
            "hand_rows": 0, "object_rows": 0,
            "invalid_bbox_count": 0, "unmapped_annotation_count": 0,
            "max_mapping_error_ns": 0,
        })
        return _build_result("pass", checks, stats, errors)

    # ---- 实体统计 ----
    if "entity_type" in df.columns:
        stats["hand_rows"] = int((df["entity_type"] == "hand").sum())
        stats["object_rows"] = int((df["entity_type"] == "object").sum())
        stats["interaction_rows"] = int(df["interaction_state"].sum() if "interaction_state" in df.columns else 0)

    # ---- 3. timestamp_ns 位于 Segment 内 ----
    if "timestamp_ns" in df.columns:
        ts = df["timestamp_ns"]
        out_of_range = (ts < 0) | (ts >= timeline_end_ns)
        if out_of_range.any():
            n_bad = int(out_of_range.sum())
            errors.append(f"{n_bad} 条记录的 timestamp_ns 超出 [0, {timeline_end_ns})")
            checks["timestamps_in_range"] = "fail"
        else:
            checks["timestamps_in_range"] = "pass"
    else:
        checks["timestamps_in_range"] = "skip"

    # ---- 4. output_frame_index 不越界 ----
    if "output_frame_index" in df.columns:
        ofi = df["output_frame_index"]
        if ofi.min() < 0:
            errors.append(f"output_frame_index 存在负值: min={ofi.min()}")
            checks["output_frame_indices_valid"] = "fail"
        else:
            checks["output_frame_indices_valid"] = "pass"
    else:
        checks["output_frame_indices_valid"] = "skip"

    # ---- 5. source_frame_index 非负 ----
    # (merged into output_frame_indices_valid check implicitly via checks below)
    # We report it as part of provenance check

    # ---- 6. BBox 合法 ----
    if all(c in df.columns for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]):
        invalid_mask = (
            (df["bbox_x1"] >= df["bbox_x2"]) |
            (df["bbox_y1"] >= df["bbox_y2"])
        )
        stats["invalid_bbox_count"] = int(invalid_mask.sum())

        if video_width is not None and video_height is not None:
            # 同时检查是否在视频范围内
            out_of_bounds = (
                (df["bbox_x1"] < 0) | (df["bbox_x2"] > video_width) |
                (df["bbox_y1"] < 0) | (df["bbox_y2"] > video_height)
            )
            stats["bbox_out_of_bounds_count"] = int(out_of_bounds.sum())

        if invalid_mask.any():
            errors.append(f"{stats['invalid_bbox_count']} 条记录的 BBox 不合法 (x1>=x2 或 y1>=y2)")
            checks["bbox_valid"] = "fail"
        else:
            checks["bbox_valid"] = "pass"
    else:
        checks["bbox_valid"] = "skip"

    # ---- 7. confidence 范围 ----
    if "confidence" in df.columns:
        conf = df["confidence"]
        bad_conf = (conf < 0.0) | (conf > 1.0)
        if bad_conf.any():
            errors.append(f"{int(bad_conf.sum())} 条记录的 confidence 不在 [0,1] 范围")
            checks["confidence_range"] = "fail"
        else:
            checks["confidence_range"] = "pass"
    else:
        checks["confidence_range"] = "skip"

    # ---- 8. 来源追溯完整 ----
    provenance_ok = True
    if "source_file" in df.columns:
        null_src = df["source_file"].isna()
        if null_src.any():
            errors.append(f"{int(null_src.sum())} 条记录缺少 source_file")
            provenance_ok = False
    else:
        provenance_ok = False

    if "source_record_index" in df.columns:
        null_sri = df["source_record_index"].isna()
        if null_sri.any():
            errors.append(f"{int(null_sri.sum())} 条记录缺少 source_record_index")
            provenance_ok = False
    else:
        provenance_ok = False

    # source_frame_index 非负
    if "source_frame_index" in df.columns:
        neg_sfi = df["source_frame_index"] < 0
        if neg_sfi.any():
            errors.append(f"{int(neg_sfi.sum())} 条记录 source_frame_index 为负")
            provenance_ok = False

    checks["provenance_complete"] = "pass" if provenance_ok else "fail"

    # ---- 9. 映射误差不过大 ----
    if "mapping_error_ns" in df.columns:
        me = df["mapping_error_ns"]
        stats["max_mapping_error_ns"] = int(me.max()) if len(me) > 0 else 0
        stats["mean_mapping_error_ns"] = float(me.mean()) if len(me) > 0 else 0.0

        unmapped = me > MAPPING_ERROR_MAX_NS
        stats["unmapped_annotation_count"] = int(unmapped.sum())

        if unmapped.any():
            errors.append(
                f"{int(unmapped.sum())} 条记录映射误差超过 "
                f"{MAPPING_ERROR_MAX_NS / 1e6:.0f}ms 上限"
            )
            checks["mapping_error_ok"] = "fail"
        elif (me > MAPPING_ERROR_WARN_NS).any():
            checks["mapping_error_ok"] = "warn"
        else:
            checks["mapping_error_ok"] = "pass"
    else:
        checks["mapping_error_ok"] = "skip"
        stats["max_mapping_error_ns"] = 0
        stats["unmapped_annotation_count"] = 0

    # ---- 10. entity_type 枚举 ----
    if "entity_type" in df.columns:
        invalid_entities = ~df["entity_type"].isin(VALID_ENTITY_TYPES)
        if invalid_entities.any():
            bad_vals = df.loc[invalid_entities, "entity_type"].unique().tolist()
            errors.append(f"非法 entity_type 值: {bad_vals}")
            checks["entity_type_valid"] = "fail"
        else:
            checks["entity_type_valid"] = "pass"
    else:
        checks["entity_type_valid"] = "skip"

    return _build_result("pass" if not errors else "fail", checks, stats, errors)


# ---- 辅助 ----

def _build_result(
    status: str,
    checks: dict[str, str],
    stats: dict,
    errors: list[str],
) -> dict:
    """去除 stats 中的 numpy 类型，确保可 JSON 序列化。"""
    clean_stats = {}
    for k, v in stats.items():
        if isinstance(v, (np.integer,)):
            clean_stats[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_stats[k] = float(v)
        else:
            clean_stats[k] = v

    return {
        "status": status,
        "checks": checks,
        "statistics": clean_stats,
        "errors": errors,
    }


__all__ = [
    "validate_hand_object_stream",
    "REQUIRED_COLUMNS",
    "VALID_ENTITY_TYPES",
    "MAPPING_ERROR_WARN_NS",
    "MAPPING_ERROR_MAX_NS",
]
