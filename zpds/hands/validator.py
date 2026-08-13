"""
Hands Validator。

校验 hands_2d.parquet 的结构完整性、数据类型、坐标范围和来源信息。

检查项:
  1. Parquet 可读
  2. 必需字段存在且类型正确
  3. keypoints_2d 为 21×2 向量，无 NaN/Inf
  4. BBox 坐标合法 (x1<x2, y1<y2)
  5. handedness ∈ {Left, Right, Unknown}
  6. 置信度 ∈ [0, 1]
  7. output_frame_index 非负且有限
  8. timestamp_ns 在 Segment 范围内 (若提供 segment.json)
  9. 来源信息完整 (model_name/version/checkpoint/config hash)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


def _is_normalized_points(kp: np.ndarray) -> bool:
    if kp.shape != (21, 2) or not np.isfinite(kp).all():
        return False
    return (
        float(kp[:, 0].min()) >= -0.05
        and float(kp[:, 0].max()) <= 1.05
        and float(kp[:, 1].min()) >= -0.05
        and float(kp[:, 1].max()) <= 1.05
    )


def _is_normalized_bbox(bbox: tuple[float, float, float, float]) -> bool:
    values = np.array(bbox, dtype=np.float32)
    return bool(np.isfinite(values).all() and values.min() >= -0.05 and values.max() <= 1.05)


def validate_hands_parquet(
    parquet_path: str,
    segment_json_path: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict:
    """校验 hands_2d.parquet，返回结构化验证报告。

    Args:
        parquet_path: Parquet 文件路径
        segment_json_path: 可选，segment.json 路径（用于时间范围校验和分分辨率）
        image_width / image_height: 可选，图像分辨率（用于 BBox/关键点边界检查）

    Returns:
        {
            "status": "pass" | "fail" | "warn",
            "checks": {check_name: "pass"|"fail"|"warn", ...},
            "statistics": {...},
            "errors": [...],
        }
    """
    errors = []
    warnings = []
    checks = {}
    stats = {}

    # ---- 1. Parquet 可读 ----
    try:
        df = pd.read_parquet(parquet_path)
        stats["total_rows"] = len(df)
        stats["columns"] = list(df.columns)
    except (OSError, ValueError, ImportError, pa.ArrowException) as e:
        return {
            "status": "fail",
            "checks": {"parquet_readable": "fail"},
            "statistics": {},
            "errors": [f"Cannot read parquet: {e}"],
        }

    checks["parquet_readable"] = "pass"

    # ---- 2. 必需字段 ----
    required = [
        "segment_id", "video_stream_id", "output_frame_index", "timestamp_ns",
        "detection_id", "handedness", "handedness_score",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "keypoints_2d", "keypoints_z_relative",
        "model_name", "model_version",
    ]
    missing = [f for f in required if f not in df.columns]
    checks["required_fields"] = "fail" if missing else "pass"
    if missing:
        errors.append(f"Missing required fields: {missing}")

    if missing:
        return {
            "status": "fail",
            "checks": checks,
            "statistics": stats,
            "errors": errors,
        }

    # ---- 3. keypoints_2d 为 21×2，无 NaN/Inf ----
    all_kp_ok = True
    all_kp_in_bounds = True
    kp_bad_rows = []
    kp_oob_rows = []
    normalized_kp_rows = []
    clipped_rows = []
    for i, row in df.iterrows():
        kp = row["keypoints_2d"]
        if not isinstance(kp, (list, np.ndarray)) or len(kp) != 21:
            all_kp_ok = False
            kp_bad_rows.append(i)
            continue
        kp_points = []
        for pt in kp:
            if not isinstance(pt, (list, np.ndarray)) or len(pt) != 2:
                all_kp_ok = False
                kp_bad_rows.append(i)
                break
            x, y = float(pt[0]), float(pt[1])
            kp_points.append((x, y))
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                all_kp_ok = False
                kp_bad_rows.append(i)
                break
        if len(kp_points) != 21:
            continue
        kp_array = np.array(kp_points, dtype=np.float32)
        if _is_normalized_points(kp_array):
            normalized_kp_rows.append(i)
        if bool(row.get("keypoints_any_clipped", False)):
            clipped_rows.append(i)
        if image_width is not None and image_height is not None:
            xs = kp_array[:, 0]
            ys = kp_array[:, 1]
            if (xs < 0).any() or (xs >= image_width).any() or (ys < 0).any() or (ys >= image_height).any():
                all_kp_in_bounds = False
                kp_oob_rows.append(i)
    checks["keypoints_valid"] = "pass" if all_kp_ok else "fail"
    if kp_bad_rows:
        errors.append(f"Invalid keypoints in rows: {kp_bad_rows[:20]}...")
    if image_width is not None and image_height is not None:
        checks["keypoints_in_image"] = "pass" if all_kp_in_bounds else "fail"
        if kp_oob_rows:
            errors.append(f"Keypoints outside image bounds in rows: {kp_oob_rows[:20]}...")
    if normalized_kp_rows and image_width is not None and image_height is not None:
        checks["keypoints_coordinate_space"] = "warn"
        warnings.append(
            "keypoints_2d looks normalized in rows "
            f"{normalized_kp_rows[:20]}..., expected pixel coordinates"
        )
    else:
        checks["keypoints_coordinate_space"] = "pass"
    stats["keypoints_dim"] = "21×2"
    if "keypoints_any_clipped" in df.columns or "keypoints_clipped_count" in df.columns:
        clipped_counts = pd.to_numeric(
            df.get("keypoints_clipped_count", pd.Series(0, index=df.index)), errors="coerce"
        )
        clipped_flags = df.get(
            "keypoints_any_clipped", pd.Series(False, index=df.index)
        ).astype(bool)
        clipped_metadata_ok = (
            clipped_counts.notna().all()
            and ((clipped_counts >= 0) & (clipped_counts <= 21)).all()
            and (clipped_flags == (clipped_counts > 0)).all()
        )
        checks["keypoint_clipping_metadata"] = "pass" if clipped_metadata_ok else "fail"
        if not clipped_metadata_ok:
            errors.append("Invalid keypoint clipping metadata")
        stats["clipped_hand_rows"] = int(clipped_flags.sum())
        stats["clipped_keypoints"] = int(clipped_counts.sum())
        if clipped_rows:
            warnings.append(
                f"Keypoints were clipped to image bounds in {len(clipped_rows)} hand rows"
            )

    # keypoints_z_relative
    z_ok = True
    for i, row in df.iterrows():
        zs = row["keypoints_z_relative"]
        if not isinstance(zs, (list, np.ndarray)) or len(zs) != 21:
            z_ok = False
            break
    checks["keypoints_z_valid"] = "pass" if z_ok else "fail"

    # ---- 4. BBox 合法 ----
    bbox_ok = True
    bbox_contains_kp = True
    normalized_bbox_rows = []
    bbox_miss_rows = []
    for i, row in df.iterrows():
        b = (float(row["bbox_x1"]), float(row["bbox_y1"]),
             float(row["bbox_x2"]), float(row["bbox_y2"]))
        if any(np.isnan(v) or np.isinf(v) for v in b):
            bbox_ok = False
            errors.append(f"Row {i}: bbox contains NaN/Inf")
            break
        if b[0] >= b[2] or b[1] >= b[3]:
            bbox_ok = False
            errors.append(f"Row {i}: bbox inverted (x1={b[0]} >= x2={b[2]} or y1={b[1]} >= y2={b[3]})")
            break
        if _is_normalized_bbox(b):
            normalized_bbox_rows.append(i)
        if image_width and (b[0] < 0 or b[2] > image_width):
            warnings.append(f"Row {i}: bbox exceeds image width ({image_width})")
        if image_height and (b[1] < 0 or b[3] > image_height):
            warnings.append(f"Row {i}: bbox exceeds image height ({image_height})")
        kp = row["keypoints_2d"]
        if isinstance(kp, (list, np.ndarray)) and len(kp) == 21:
            pts = np.array(list(kp), dtype=np.float32)
            if pts.shape == (21, 2) and np.isfinite(pts).all() and not _is_normalized_points(pts):
                xs = pts[:, 0]
                ys = pts[:, 1]
                # Allow one pixel tolerance for float/int roundtrips.
                if (xs < b[0] - 1).any() or (xs > b[2] + 1).any() or (ys < b[1] - 1).any() or (ys > b[3] + 1).any():
                    bbox_contains_kp = False
                    bbox_miss_rows.append(i)
    checks["bbox_valid"] = "pass" if bbox_ok else "fail"
    checks["bbox_contains_keypoints"] = "pass" if bbox_contains_kp else "warn"
    if bbox_miss_rows:
        warnings.append(f"BBox does not contain all keypoints in rows: {bbox_miss_rows[:20]}...")
    if normalized_bbox_rows and image_width is not None and image_height is not None:
        checks["bbox_coordinate_space"] = "warn"
        warnings.append(
            "bbox xyxy looks normalized in rows "
            f"{normalized_bbox_rows[:20]}..., expected pixel coordinates"
        )
    else:
        checks["bbox_coordinate_space"] = "pass"
    stats["bbox_count"] = int((~df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().any(axis=1)).sum())

    # ---- 5. handedness 合法 ----
    valid_hands = df["handedness"].isin(["Left", "Right", "Unknown"])
    checks["handedness_valid"] = "pass" if valid_hands.all() else "fail"
    if not valid_hands.all():
        bad = df.loc[~valid_hands, "handedness"].unique().tolist()
        errors.append(f"Invalid handedness values: {bad}")
    stats["handedness_dist"] = df["handedness"].value_counts().to_dict()

    # ---- 6. 置信度 ∈ [0, 1] ----
    hs = df["handedness_score"]
    score_ok = ((hs >= 0) & (hs <= 1)).all()
    checks["confidence_range"] = "pass" if score_ok else "fail"
    if not score_ok:
        errors.append(f"handedness_score out of [0, 1]: min={hs.min()}, max={hs.max()}")

    # ---- 7. output_frame_index 有效 ----
    ofi = df["output_frame_index"]
    ofi_ok = (ofi >= 0).all() and not ofi.isna().any()
    checks["frame_index_valid"] = "pass" if ofi_ok else "fail"
    if not ofi_ok:
        errors.append(
            f"output_frame_index invalid: min={ofi.min()}, "
            f"has_nan={ofi.isna().any()}"
        )
    stats["frame_range"] = None if df.empty else [int(ofi.min()), int(ofi.max())]

    # ---- 8. timestamp_ns 范围 ----
    if segment_json_path and Path(segment_json_path).exists():
        with open(segment_json_path, encoding="utf-8") as f:
            seg = json.load(f)
        timeline_end = seg["timeline"]["end_ns"]
        ts = df["timestamp_ns"]
        ts_ok = (ts >= 0).all() and (ts <= timeline_end).all()
        checks["timestamp_in_segment"] = "pass" if ts_ok else "fail"
        if not ts_ok:
            errors.append(
                f"timestamp_ns out of segment range [0, {timeline_end}]: "
                f"min={ts.min()}, max={ts.max()}"
            )

    # ---- 9. 来源完整性 ----
    source_fields = ["model_name", "model_version", "checkpoint_sha256", "config_sha256"]
    has_source = all(f in df.columns for f in source_fields)
    if has_source:
        missing_source = not df.empty and (df["model_name"] == "").all()
        checks["provenance_complete"] = "warn" if missing_source else "pass"
        if missing_source:
            warnings.append("Model provenance fields are empty")
    else:
        checks["provenance_complete"] = "fail"

    backend_fields = [
        "backend_requested",
        "backend_active",
        "backend_fallback_used",
        "backend_fallback_reason",
        "backend_delegate",
    ]
    present_backend_fields = [field for field in backend_fields if field in df.columns]
    if present_backend_fields:
        required_backend_fields = ["backend_requested", "backend_active", "backend_fallback_used"]
        backend_complete = all(field in df.columns for field in required_backend_fields)
        if backend_complete:
            backend_complete = not (
                (df["backend_requested"] == "").any() | (df["backend_active"] == "").any()
            )
        checks["backend_provenance"] = "pass" if backend_complete else "warn"
        if not backend_complete:
            warnings.append("Backend provenance is incomplete")

    # ---- 汇总 ----
    all_pass = len(errors) == 0
    has_warnings = len(warnings) > 0
    status = "fail" if not all_pass else ("warn" if has_warnings else "pass")

    return {
        "status": status,
        "checks": checks,
        "statistics": stats,
        "errors": errors,
        "warnings": warnings,
    }


def validate_wilor_hands(
    parquet_path: str,
    *,
    hands_run_path: str,
    image_width: int,
    image_height: int,
    segment_json_path: str | None = None,
    max_failure_ratio: float = 0.02,
    expected_model_version: str = "wilor_cvpr2025",
    wrist_tolerance_px: float = 2.0,
) -> dict:
    """Validate the additional contracts required for a WiLoR Hands V1 run.

    This intentionally layers WiLoR checks on top of the model-agnostic validator.
    Single-backend runs must be attributed entirely to WiLoR; any other backend
    attribution fails the check.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width 和 image_height 必须为正数")
    if not 0.0 <= max_failure_ratio <= 1.0:
        raise ValueError("max_failure_ratio 必须在 [0, 1] 范围内")
    if wrist_tolerance_px < 0:
        raise ValueError("wrist_tolerance_px 不能为负数")

    result = validate_hands_parquet(
        parquet_path,
        segment_json_path=segment_json_path,
        image_width=image_width,
        image_height=image_height,
    )
    checks = result["checks"]
    stats = result["statistics"]
    errors = result["errors"]
    warnings = result.get("warnings", [])

    try:
        frame = pd.read_parquet(parquet_path)
    except (OSError, ValueError, ImportError, pa.ArrowException):
        # The generic report already records the read failure.
        return result

    _validate_wilor_pixel_inverse_transform(
        frame, image_width, image_height, checks, errors
    )
    _validate_wilor_reprojection_consistency(
        frame, wrist_tolerance_px, checks, errors
    )

    run_document: dict | None = None
    try:
        run_document = _load_wilor_run_report(hands_run_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        checks["wilor_provenance"] = "fail"
        checks["failure_records"] = "fail"
        errors.append(f"Cannot read WiLoR hands_run.json: {error}")

    if run_document is not None:
        _validate_wilor_provenance(
            frame,
            run_document,
            expected_model_version,
            checks,
            errors,
        )
        _validate_wilor_failure_records(
            run_document,
            max_failure_ratio,
            checks,
            stats,
            errors,
        )

    _validate_wilor_backend_attribution(frame, checks, errors)

    result["status"] = "fail" if errors else ("warn" if warnings else "pass")
    return result


def _validate_wilor_pixel_inverse_transform(
    frame: pd.DataFrame,
    image_width: int,
    image_height: int,
    checks: dict,
    errors: list,
) -> None:
    invalid_rows: list[int] = []
    for index, keypoints in frame.get("keypoints_2d", pd.Series(dtype=object)).items():
        try:
            points = np.array(list(keypoints), dtype=np.float64)
        except (TypeError, ValueError):
            invalid_rows.append(int(index))
            continue
        if (
            points.shape != (21, 2)
            or not np.isfinite(points).all()
            or (points[:, 0] < 0).any()
            or (points[:, 0] > image_width - 1).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] > image_height - 1).any()
        ):
            invalid_rows.append(int(index))
    checks["pixel_inverse_transform"] = "pass" if not invalid_rows else "fail"
    if invalid_rows:
        errors.append(
            "WiLoR pixel inverse transform invalid in rows: "
            f"{invalid_rows[:20]}..."
        )


def _validate_wilor_reprojection_consistency(
    frame: pd.DataFrame,
    wrist_tolerance_px: float,
    checks: dict,
    errors: list,
) -> None:
    invalid_rows: list[int] = []
    for index, row in frame.iterrows():
        try:
            wrist = np.array(list(row["keypoints_2d"]), dtype=np.float64)[0]
            x1, y1, x2, y2 = (
                float(row["bbox_x1"]),
                float(row["bbox_y1"]),
                float(row["bbox_x2"]),
                float(row["bbox_y2"]),
            )
        except (IndexError, KeyError, TypeError, ValueError):
            invalid_rows.append(int(index))
            continue
        if (
            wrist.shape != (2,)
            or not np.isfinite(wrist).all()
            or wrist[0] < x1 - wrist_tolerance_px
            or wrist[0] > x2 + wrist_tolerance_px
            or wrist[1] < y1 - wrist_tolerance_px
            or wrist[1] > y2 + wrist_tolerance_px
        ):
            invalid_rows.append(int(index))
    checks["reprojection_consistency"] = "pass" if not invalid_rows else "fail"
    if invalid_rows:
        errors.append(
            "WiLoR wrist is outside its bbox in rows: "
            f"{invalid_rows[:20]}..."
        )


def _load_wilor_run_report(path: str) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("WiLoR hands_run.json 顶层必须是对象")
    return document


def _validate_wilor_provenance(
    frame: pd.DataFrame,
    report: dict,
    expected_model_version: str,
    checks: dict,
    errors: list,
) -> None:
    model = report.get("model")
    model = model if isinstance(model, dict) else {}
    wilor_rows = frame[frame.get("model_name", pd.Series(dtype=str)) == "wilor"]
    parquet_complete = (
        not wilor_rows.empty
        and "checkpoint_sha256" in wilor_rows
        and wilor_rows["checkpoint_sha256"].astype(str).ne("").all()
        and "model_version" in wilor_rows
        and wilor_rows["model_version"].eq(expected_model_version).all()
    )
    report_complete = (
        model.get("name") == "wilor"
        and model.get("version") == expected_model_version
        and bool(model.get("checkpoint_sha256"))
        and bool(model.get("device"))
    )
    checks["wilor_provenance"] = "pass" if parquet_complete and report_complete else "fail"
    if checks["wilor_provenance"] == "fail":
        errors.append(
            "WiLoR provenance must include matching model version, checkpoint SHA-256, "
            "and device"
        )


def _validate_wilor_failure_records(
    report: dict,
    max_failure_ratio: float,
    checks: dict,
    stats: dict,
    errors: list,
) -> None:
    coverage = report.get("coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    total = coverage.get("decoded_frames", coverage.get("total_frames"))
    failed = coverage.get("failed_frames", coverage.get("failed"))
    try:
        total_value = int(total)
        failed_value = int(failed)
    except (TypeError, ValueError):
        total_value, failed_value = 0, 0
    ratio = failed_value / total_value if total_value > 0 else float("inf")
    stats["wilor_failure_ratio"] = ratio
    ok = total_value > 0 and 0 <= failed_value <= total_value and ratio < max_failure_ratio
    checks["failure_records"] = "pass" if ok else "fail"
    if not ok:
        errors.append(
            "WiLoR failure ratio must be below "
            f"{max_failure_ratio:.2%}; got {failed_value}/{total_value}"
        )


def _validate_wilor_backend_attribution(
    frame: pd.DataFrame,
    checks: dict,
    errors: list,
) -> None:
    required = {
        "backend_requested",
        "backend_active",
        "backend_fallback_used",
        "backend_fallback_reason",
    }
    if not required.issubset(frame.columns):
        checks["backend_attribution"] = "fail"
        errors.append("WiLoR backend attribution columns are missing")
        return

    requested = frame["backend_requested"].astype(str)
    active = frame["backend_active"].astype(str)
    fallback_used = frame["backend_fallback_used"].astype(bool)
    valid = (
        requested.eq("wilor")
        & active.eq("wilor")
        & ~fallback_used
    )
    checks["backend_attribution"] = "pass" if valid.all() else "fail"
    if not valid.all():
        bad_rows = frame.index[~valid].tolist()
        errors.append(f"Invalid WiLoR backend attribution in rows: {bad_rows[:20]}...")
