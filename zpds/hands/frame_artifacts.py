"""WiLoR 全帧状态与 BBox 资产的写出和覆盖率校验。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zpds.hands.contracts import FrameInferenceRecord

FRAME_STATUS_COLUMNS = [
    "prep_revision",
    "segment_id",
    "video_stream_id",
    "output_frame_index",
    "timestamp_ns",
    "source_frame_index",
    "source_timestamp_ns",
    "inference_status",
    "hand_count",
    "failure_reason",
    "model_name",
    "model_version",
    "checkpoint_sha256",
    "config_sha256",
    "active_backend",
    "device",
    "inference_ms",
]

BBOX_COLUMNS = [
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
    "detection_score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "model_name",
    "model_version",
    "checkpoint_sha256",
    "config_sha256",
]

VALID_FRAME_STATUSES = frozenset(
    {"detected", "no_hand", "failed", "skipped_invalid_input"}
)


@dataclass(frozen=True)
class InferenceArtifactContext:
    """写出两张 WiLoR 全帧资产所需的 run 级来源信息。"""

    prep_revision: str
    segment_id: str
    video_stream_id: str
    model_name: str
    model_version: str
    checkpoint_sha256: str
    config_sha256: str
    device: str

    def __post_init__(self) -> None:
        for field_name in (
            "prep_revision",
            "segment_id",
            "video_stream_id",
            "model_name",
            "model_version",
            "checkpoint_sha256",
            "config_sha256",
            "device",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(
                    f"InferenceArtifactContext.{field_name} 不能为空"
                )


class ParquetFrameStatusWriter:
    """将每个 Prepared 输出帧写成一条 WiLoR 状态记录。"""

    def __init__(
        self,
        output_path: str | Path,
        context: InferenceArtifactContext,
    ) -> None:
        self._output_path = Path(output_path).expanduser().resolve()
        self._context = context
        self._rows: list[dict[str, Any]] = []
        self._closed = False

    def write(self, record: FrameInferenceRecord) -> None:
        self._ensure_open()
        if not isinstance(record, FrameInferenceRecord):
            raise TypeError("frame-status Writer 只接受 FrameInferenceRecord")
        frame = record.frame
        self._rows.append(
            {
                "prep_revision": self._context.prep_revision,
                "segment_id": self._context.segment_id,
                "video_stream_id": self._context.video_stream_id,
                "output_frame_index": frame.output_frame_index,
                "timestamp_ns": frame.timestamp_ns,
                "source_frame_index": frame.source_frame_index,
                "source_timestamp_ns": frame.source_timestamp_ns,
                "inference_status": record.inference_status,
                "hand_count": len(record.raw_hands),
                "failure_reason": record.failure_reason,
                "model_name": self._context.model_name,
                "model_version": self._context.model_version,
                "checkpoint_sha256": self._context.checkpoint_sha256,
                "config_sha256": self._context.config_sha256,
                "active_backend": record.active_backend,
                "device": self._context.device,
                "inference_ms": record.inference_ms,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _write_rows(self._rows, FRAME_STATUS_COLUMNS, self._output_path)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ParquetFrameStatusWriter 已关闭")


class ParquetBBoxWriter:
    """写出 WiLoR 主尝试检测到的原图像素 BBox。"""

    def __init__(
        self,
        output_path: str | Path,
        context: InferenceArtifactContext,
    ) -> None:
        self._output_path = Path(output_path).expanduser().resolve()
        self._context = context
        self._rows: list[dict[str, Any]] = []
        self._closed = False

    def write(self, record: FrameInferenceRecord) -> None:
        self._ensure_open()
        if not isinstance(record, FrameInferenceRecord):
            raise TypeError("BBox Writer 只接受 FrameInferenceRecord")
        frame = record.frame
        for detection_id, hand in enumerate(record.raw_hands):
            self._rows.append(
                {
                    "prep_revision": self._context.prep_revision,
                    "segment_id": self._context.segment_id,
                    "video_stream_id": self._context.video_stream_id,
                    "output_frame_index": frame.output_frame_index,
                    "timestamp_ns": frame.timestamp_ns,
                    "source_frame_index": frame.source_frame_index,
                    "source_timestamp_ns": frame.source_timestamp_ns,
                    "detection_id": detection_id,
                    "handedness": _normalize_handedness(hand.handedness),
                    "handedness_score": float(hand.handedness_score),
                    "detection_score": float(hand.detection_score),
                    "bbox_x1": float(hand.bbox.x1),
                    "bbox_y1": float(hand.bbox.y1),
                    "bbox_x2": float(hand.bbox.x2),
                    "bbox_y2": float(hand.bbox.y2),
                    "model_name": self._context.model_name,
                    "model_version": self._context.model_version,
                    "checkpoint_sha256": self._context.checkpoint_sha256,
                    "config_sha256": self._context.config_sha256,
                }
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _write_rows(self._rows, BBOX_COLUMNS, self._output_path)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ParquetBBoxWriter 已关闭")


def validate_wilor_frame_artifacts(
    frame_status_path: str | Path,
    bbox_path: str | Path,
    sample_map_path: str | Path,
    *,
    expected_frame_count: int | None = None,
    max_failure_ratio: float = 0.02,
) -> dict[str, Any]:
    """校验全帧覆盖、Sample Map 对齐、BBox 数量和 provenance。"""

    errors: list[str] = []
    checks: dict[str, str] = {}
    statistics: dict[str, int] = {}
    try:
        status = pd.read_parquet(frame_status_path)
        bbox = pd.read_parquet(bbox_path)
        sample_map = pd.read_parquet(sample_map_path)
    except (OSError, ValueError, ImportError) as error:
        return {
            "status": "fail",
            "checks": {"parquet_readable": "fail"},
            "statistics": {},
            "errors": [f"Cannot read WiLoR frame artifact: {error}"],
        }

    checks["parquet_readable"] = "pass"
    _require_columns(status, FRAME_STATUS_COLUMNS, "frame-status", errors)
    _require_columns(bbox, BBOX_COLUMNS, "bbox", errors)
    checks["schema"] = "pass" if not errors else "fail"
    if errors:
        return {
            "status": "fail",
            "checks": checks,
            "statistics": statistics,
            "errors": errors,
        }

    if not 0.0 <= max_failure_ratio <= 1.0:
        raise ValueError("max_failure_ratio 必须在 [0, 1] 范围内")

    expected_count = (
        len(sample_map)
        if expected_frame_count is None
        else expected_frame_count
    )
    expected = sample_map.iloc[:expected_count]
    statistics.update(
        {
            "frame_status_rows": len(status),
            "bbox_rows": len(bbox),
            "expected_frame_count": expected_count,
        }
    )

    status_indices = status["output_frame_index"].astype("int64").tolist()
    expected_indices = expected["output_frame_index"].astype("int64").tolist()
    status_timestamps = status["timestamp_ns"].astype("int64").tolist()
    expected_timestamps = (
        expected["output_timestamp_ns"].astype("int64").tolist()
    )
    coverage_ok = (
        len(status) == expected_count
        and status_indices == expected_indices
        and status_timestamps == expected_timestamps
        and status["output_frame_index"].is_unique
    )
    checks["sample_map_alignment"] = "pass" if coverage_ok else "fail"
    if not coverage_ok:
        errors.append(
            "WiLoR frame-status rows must match the processed Sample Map prefix"
        )

    status_values = set(status["inference_status"].astype(str))
    status_ok = status_values <= VALID_FRAME_STATUSES
    failed = status["inference_status"].eq("failed")
    skipped = status["inference_status"].eq("skipped_invalid_input")
    reasons = status["failure_reason"].fillna("").astype(str).str.strip()
    reasons_ok = reasons[failed | skipped].ne("").all()
    hand_counts = status["hand_count"].astype("int64")
    count_semantics_ok = (
        (hand_counts >= 0).all()
        and hand_counts[status["inference_status"].eq("detected")].gt(0).all()
        and hand_counts[~status["inference_status"].eq("detected")].eq(0).all()
    )
    checks["frame_status_semantics"] = (
        "pass"
        if status_ok and reasons_ok and count_semantics_ok
        else "fail"
    )
    if checks["frame_status_semantics"] == "fail":
        errors.append("WiLoR frame-status values or failure semantics are invalid")

    failed_count = int(failed.sum())
    skipped_count = int(skipped.sum())
    failure_ratio = (
        failed_count / expected_count if expected_count > 0 else 1.0
    )
    statistics["failed_frames"] = failed_count
    statistics["skipped_invalid_input_frames"] = skipped_count
    coverage_quality_ok = (
        expected_count > 0
        and failure_ratio < max_failure_ratio
        and skipped_count == 0
    )
    checks["coverage_quality"] = (
        "pass" if coverage_quality_ok else "fail"
    )
    if not coverage_quality_ok:
        errors.append(
            "WiLoR failure ratio must be below "
            f"{max_failure_ratio:.2%} and skipped_invalid_input must be zero; "
            f"got failed={failed_count}/{expected_count}, skipped={skipped_count}"
        )

    bbox_ok = _bbox_rows_are_valid(bbox)
    unique_bbox = not bbox.duplicated(
        ["output_frame_index", "detection_id"]
    ).any()
    bbox_count_ok = len(bbox) == int(hand_counts.sum())
    bbox_frame_counts = (
        bbox.groupby("output_frame_index").size()
        if not bbox.empty
        else pd.Series(dtype="int64")
    )
    expected_frame_counts = status.set_index("output_frame_index")["hand_count"]
    per_frame_count_ok = all(
        int(bbox_frame_counts.get(index, 0)) == int(count)
        for index, count in expected_frame_counts.items()
    )
    checks["bbox_contract"] = (
        "pass"
        if bbox_ok and unique_bbox and bbox_count_ok and per_frame_count_ok
        else "fail"
    )
    if checks["bbox_contract"] == "fail":
        errors.append("WiLoR BBox rows are invalid or do not match hand_count")

    provenance_columns = (
        "model_name",
        "model_version",
        "checkpoint_sha256",
        "config_sha256",
    )
    provenance_ok = all(
        status[column].fillna("").astype(str).str.strip().ne("").all()
        for column in provenance_columns
    ) and all(
        bbox[column].fillna("").astype(str).str.strip().ne("").all()
        for column in provenance_columns
    )
    checks["provenance"] = "pass" if provenance_ok else "fail"
    if not provenance_ok:
        errors.append("WiLoR frame artifacts have incomplete provenance")

    return {
        "status": "fail" if errors else "pass",
        "checks": checks,
        "statistics": statistics,
        "errors": errors,
    }


def _write_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("source_frame_index", "source_timestamp_ns"):
        frame[column] = frame[column].astype("Int64")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)


def _require_columns(
    frame: pd.DataFrame,
    required: list[str],
    label: str,
    errors: list[str],
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        errors.append(f"{label} parquet missing columns: {missing}")


def _bbox_rows_are_valid(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return True
    try:
        boxes = frame[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].astype(
            "float64"
        )
    except (KeyError, TypeError, ValueError):
        return False
    finite = boxes.map(math.isfinite).all(axis=None)
    return bool(
        finite
        and (boxes["bbox_x1"] >= 0).all()
        and (boxes["bbox_y1"] >= 0).all()
        and (boxes["bbox_x2"] > boxes["bbox_x1"]).all()
        and (boxes["bbox_y2"] > boxes["bbox_y1"]).all()
    )


def _normalize_handedness(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in {"left", "right"} else "unknown"


__all__ = [
    "BBOX_COLUMNS",
    "FRAME_STATUS_COLUMNS",
    "InferenceArtifactContext",
    "ParquetBBoxWriter",
    "ParquetFrameStatusWriter",
    "validate_wilor_frame_artifacts",
]
