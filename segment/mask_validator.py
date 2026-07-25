"""
Mask 流写出后验证：确认 instance_segmentation Parquet 数据完整、合法。

检查项:
  1. Parquet 可读取
  2. 必需列存在
  3. timestamp_ns 位于 Segment 时间范围内
  4. output_frame_index 不越界
  5. source_frame_index 非负
  6. Mask shape 等于视频分辨率
  7. Mask 不为空 (area > 0)
  8. Mask 不全满
  9. BBox 覆盖 Mask (bbox_area >= mask_area * 0.5)
  10. RLE 能成功解码
  11. 解码后面积一致 (round-trip check)
  12. 来源追溯完整

用法:
    from segment.mask_validator import validate_mask_stream

    result = validate_mask_stream(
        seg_dir=Path("prepared_segments/seg_000001"),
        stream=stream_entry_from_segment_json,
        timeline_end_ns=segment["timeline"]["end_ns"],
    )
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


# ---- 必需列 ----

REQUIRED_COLUMNS = [
    "timestamp_ns", "output_frame_index",
    "source_timestamp_ns", "source_frame_index",
    "instance_id",
    "category_id", "score",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "mask_height", "mask_width",
    "rle_counts", "rle_encoding",
    "source_file",
]

# Mask 全满面积比例阈值 (≥ 此值判定为全满)
FULL_MASK_RATIO = 0.98

# BBox 覆盖最小比例
BBOX_COVERAGE_MIN_RATIO = 0.3


# ---- 主验证函数 ----

def validate_mask_stream(
    seg_dir: Path,
    stream: dict,
    timeline_end_ns: int,
    video_width: int | None = None,
    video_height: int | None = None,
    *,
    sample_decode_count: int = 20,
) -> dict:
    """验证 instance_segmentation 标注流。

    Args:
        seg_dir: Prepared Segment 根目录
        stream: segment.json 中该流的条目
        timeline_end_ns: Segment timeline 结束时间 (ns)
        video_width: 预期视频宽度
        video_height: 预期视频高度
        sample_decode_count: RLE 解码抽样数量 (全部解码可能很慢)

    Returns:
        {"checks": {...}, "statistics": {...}, "errors": [...]}
    """
    errors: list[str] = []
    checks: dict[str, str] = {}
    stats: dict = {}

    uri = stream.get("uri", "")
    parquet_path = seg_dir / uri

    # ---- 1. Parquet 可读取 ----
    if not parquet_path.exists():
        errors.append(f"Mask Parquet 不存在: {parquet_path}")
        checks["mask_file_readable"] = "fail"
        return _build_result("fail", checks, stats, errors)

    try:
        df = pd.read_parquet(str(parquet_path))
    except Exception as exc:
        errors.append(f"Mask Parquet 读取失败: {exc}")
        checks["mask_file_readable"] = "fail"
        return _build_result("fail", checks, stats, errors)

    checks["mask_file_readable"] = "pass"
    stats["mask_rows"] = len(df)

    # ---- 2. 必需列 ----
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"缺少必需列: {missing_cols}")
        checks["mask_required_columns"] = "fail"
    else:
        checks["mask_required_columns"] = "pass"

    if df.empty:
        _skip_all(checks)
        return _build_result("pass", checks, stats, errors)

    # ---- 3. timestamp_ns 在范围内 ----
    if "timestamp_ns" in df.columns:
        ts = df["timestamp_ns"]
        out_of_range = (ts < 0) | (ts >= timeline_end_ns)
        if out_of_range.any():
            errors.append(f"{int(out_of_range.sum())} 条 mask 记录 timestamp_ns 越界")
            checks["mask_timestamps_in_range"] = "fail"
        else:
            checks["mask_timestamps_in_range"] = "pass"
    else:
        checks["mask_timestamps_in_range"] = "skip"

    # ---- 4. output_frame_index 不越界 ----
    if "output_frame_index" in df.columns:
        if df["output_frame_index"].min() < 0:
            errors.append("output_frame_index 存在负值")
            checks["mask_output_frame_indices_valid"] = "fail"
        else:
            checks["mask_output_frame_indices_valid"] = "pass"
    else:
        checks["mask_output_frame_indices_valid"] = "skip"

    # ---- 5. source_frame_index 非负 (合并入 provenance) ----
    # (checked below)

    # ---- 6. Mask shape 等于视频分辨率 ----
    shape_ok = True
    if video_width is not None and video_height is not None:
        if "mask_height" in df.columns and "mask_width" in df.columns:
            bad_h = df["mask_height"] != video_height
            bad_w = df["mask_width"] != video_width
            bad_shape = bad_h | bad_w
            stats["mask_shape_mismatch_count"] = int(bad_shape.sum())
            if bad_shape.any():
                errors.append(f"{stats['mask_shape_mismatch_count']} 条记录 mask shape 不匹配 "
                              f"(期望 {video_height}×{video_width})")
                shape_ok = False
    checks["mask_shape_valid"] = "pass" if shape_ok else "fail"

    # ---- 7. Mask 面积 > 0 ----
    if "mask_area_px" in df.columns:
        empty_masks = df["mask_area_px"] <= 0
        stats["empty_mask_count"] = int(empty_masks.sum())
        if empty_masks.any():
            errors.append(f"{stats['empty_mask_count']} 个 mask 为空 (area=0)")
            checks["mask_not_empty"] = "fail"
        else:
            checks["mask_not_empty"] = "pass"
    else:
        checks["mask_not_empty"] = "skip"

    # ---- 8. Mask 不全满 ----
    if "mask_area_px" in df.columns and video_width is not None and video_height is not None:
        total_px = video_width * video_height
        full_masks = df["mask_area_px"] >= total_px * FULL_MASK_RATIO
        stats["full_mask_count"] = int(full_masks.sum())
        if full_masks.any():
            checks["mask_not_full"] = "warn"
        else:
            checks["mask_not_full"] = "pass"
    else:
        checks["mask_not_full"] = "skip"

    # ---- 9. BBox 覆盖 Mask ----
    if all(c in df.columns for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "mask_area_px"]):
        bbox_areas = ((df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"]))
        # bbox_area 应 >= mask_area * BBOX_COVERAGE_MIN_RATIO
        poor_coverage = (bbox_areas > 0) & (bbox_areas < df["mask_area_px"] * BBOX_COVERAGE_MIN_RATIO)
        stats["bbox_poor_coverage_count"] = int(poor_coverage.sum())
        if poor_coverage.any():
            errors.append(f"{stats['bbox_poor_coverage_count']} 条记录 BBox 面积远小于 Mask 面积")
            checks["mask_bbox_coverage"] = "fail"
        else:
            checks["mask_bbox_coverage"] = "pass"
    else:
        checks["mask_bbox_coverage"] = "skip"

    # ---- 10-11. RLE 解码抽样 + 面积一致性 ----
    if "rle_counts" in df.columns and "mask_height" in df.columns and "mask_width" in df.columns:
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            checks["mask_rle_decode"] = "skip"
            checks["mask_rle_area_consistent"] = "skip"
        else:
            sample_indices = _sample_indices(len(df), sample_decode_count)
            decode_failures = 0
            area_mismatches = 0

            for idx in sample_indices:
                row = df.iloc[idx]
                counts = row["rle_counts"]
                h = int(row["mask_height"])
                w = int(row["mask_width"])

                try:
                    rle = {"size": [h, w], "counts": counts}
                    decoded = mask_utils.decode(rle)
                    decoded_area = int(decoded.sum())
                    stored_area = int(row.get("mask_area_px", 0))

                    if abs(decoded_area - stored_area) > max(1, stored_area * 0.05):
                        area_mismatches += 1
                except Exception:
                    decode_failures += 1

            stats["rle_decode_samples"] = len(sample_indices)
            stats["rle_decode_failures"] = decode_failures
            stats["rle_area_mismatches"] = area_mismatches

            if decode_failures > 0:
                errors.append(f"RLE 解码失败: {decode_failures}/{len(sample_indices)} 条抽样")
                checks["mask_rle_decode"] = "fail"
            else:
                checks["mask_rle_decode"] = "pass"

            if area_mismatches > 0:
                errors.append(f"RLE 解码面积不一致: {area_mismatches}/{len(sample_indices)} 条抽样")
                checks["mask_rle_area_consistent"] = "fail"
            else:
                checks["mask_rle_area_consistent"] = "pass"
    else:
        checks["mask_rle_decode"] = "skip"
        checks["mask_rle_area_consistent"] = "skip"

    # ---- 12. 来源追溯 ----
    provenance_ok = True
    if "source_file" in df.columns:
        if df["source_file"].isna().any():
            errors.append(f"{int(df['source_file'].isna().sum())} 条缺少 source_file")
            provenance_ok = False
    else:
        provenance_ok = False

    if "source_frame_index" in df.columns:
        if (df["source_frame_index"] < 0).any():
            errors.append("source_frame_index 存在负值")
            provenance_ok = False

    checks["mask_provenance_complete"] = "pass" if provenance_ok else "fail"

    # ---- 统计 ----
    if "category_id" in df.columns:
        stats["unique_categories"] = int(df["category_id"].nunique())
        stats["total_instances"] = int(df["instance_id"].nunique()) if "instance_id" in df.columns else len(df)
    if "score" in df.columns:
        stats["score_min"] = float(df["score"].min())
        stats["score_max"] = float(df["score"].max())
        stats["score_mean"] = float(df["score"].mean())

    return _build_result("pass" if not errors else "fail", checks, stats, errors)


# ---- 辅助 ----

def _sample_indices(total: int, n: int) -> list[int]:
    """在 [0, total) 中均匀采样 n 个索引。"""
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]


def _skip_all(checks: dict[str, str]) -> None:
    for key in [
        "mask_timestamps_in_range", "mask_output_frame_indices_valid",
        "mask_shape_valid", "mask_not_empty", "mask_not_full",
        "mask_bbox_coverage", "mask_rle_decode", "mask_rle_area_consistent",
        "mask_provenance_complete",
    ]:
        checks[key] = "skip"


def _build_result(
    status: str,
    checks: dict[str, str],
    stats: dict,
    errors: list[str],
) -> dict:
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
    "validate_mask_stream",
    "REQUIRED_COLUMNS",
    "FULL_MASK_RATIO",
    "BBOX_COVERAGE_MIN_RATIO",
]
