"""Guida 基础清洗：Source Map、硬质量区间、Prepared 写出与回读。"""

import bisect
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import yaml

from zpds.adapters.common import safe_identifier, sha256_file
from zpds.adapters.guida import GuidaAdapter
from zpds.adapters.video import VideoDecoder
from zpds.config import LoadedConfig
from zpds.core.types import CalibrationDescriptor
from zpds.utils.schema_validator import validate_with_schema

from .validator import PreparedValidator
from .writer import PreparedSegmentWriter


@dataclass(frozen=True)
class GuidaFrameRef:
    source_row: int
    source_seq: int
    timestamp_ns: int
    source_segment: int
    source_frame_index: int
    color_relative_path: str
    depth_relative_path: str


@dataclass(frozen=True)
class GuidaImuSample:
    source_row: int
    timestamp_ns: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    source_relative_path: str


@dataclass(frozen=True)
class BasicQualityIssue:
    code: str
    severity: str
    decision: str
    message: str
    evidence_uri: str
    start_ns: int | None = None
    end_ns: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_segment_issue(self, stage: int) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "uri": self.evidence_uri,
            "kind": "basic_cleaning_check",
            "description": self.message,
        }
        if self.start_ns is not None:
            evidence["start_ns"] = self.start_ns
        if self.end_ns is not None:
            evidence["end_ns"] = self.end_ns
        return {
            "stage": stage,
            "reason_code": self.code,
            "severity": self.severity,
            "decision": self.decision,
            "message": self.message,
            "evidence": [evidence],
        }

    def to_report(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "decision": self.decision,
            "message": self.message,
            "evidence_uri": self.evidence_uri,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "details": self.details,
        }


@dataclass(frozen=True)
class PhysicalProblem:
    start_ns: int
    end_ns: int
    code: str
    evidence_uri: str


@dataclass(frozen=True)
class SourceSpan:
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class GuidaCleaningResult:
    revision_dir: Path
    segment_ids: tuple[str, ...]
    source_frame_count: int
    imu_sample_count: int
    issues: tuple[BasicQualityIssue, ...]
    removed_spans: tuple[PhysicalProblem, ...]


class GuidaBasicCleaner:
    """执行新多源流程中的 Guida 基础清洗参考切片。"""

    def __init__(
        self,
        *,
        pipeline_config: LoadedConfig,
        thresholds_path: str | Path,
        code_version: str,
        config_uri: str = "configs/pipeline/default.yaml",
    ) -> None:
        if len(code_version) < 7:
            raise ValueError("code_version must contain at least 7 characters")
        self.pipeline_config = pipeline_config
        self.thresholds_path = Path(thresholds_path)
        self.thresholds = _load_thresholds(self.thresholds_path)
        self.code_version = code_version
        self.config_uri = config_uri

    def clean(
        self,
        session_path: str | Path,
        output_root: str | Path,
        *,
        raw_session_uri: str,
        prep_revision: str = "r0001",
    ) -> GuidaCleaningResult:
        if not raw_session_uri.startswith("raw://"):
            raise ValueError("raw_session_uri must use raw://")
        root = Path(session_path).resolve()
        raw_root = _raw_root_for_session(root, raw_session_uri)
        adapter = GuidaAdapter()
        structure = adapter.validate(str(root))
        if not structure.passed:
            codes = ", ".join(issue.code for issue in structure.issues)
            raise ValueError(f"Guida structure validation failed: {codes}")
        inventory = adapter.inspect(str(root))
        frames = _read_frame_refs(root)
        imu = _read_imu_samples(root)
        if not frames:
            raise ValueError("Guida index contains no frame records")
        if not imu:
            raise ValueError("Guida session contains no IMU samples")
        issues, physical_problems, media_counts = self._analyze(root, frames, imu)
        source_period_ns = _median_positive_delta(
            [frame.timestamp_ns for frame in frames],
            fallback=33_333_333,
        )
        base_start = max(frames[0].timestamp_ns, imu[0].timestamp_ns)
        base_end = min(frames[-1].timestamp_ns + source_period_ns, imu[-1].timestamp_ns + 1)
        if base_end <= base_start:
            raise ValueError("Guida video and IMU have no common time coverage")
        if frames[0].timestamp_ns < base_start:
            physical_problems.append(
                PhysicalProblem(
                    frames[0].timestamp_ns,
                    base_start,
                    "clock_misalign",
                    f"{raw_session_uri}/index.jsonl",
                )
            )
        if frames[-1].timestamp_ns + source_period_ns > base_end:
            physical_problems.append(
                PhysicalProblem(
                    base_end,
                    frames[-1].timestamp_ns + source_period_ns,
                    "clock_misalign",
                    f"{raw_session_uri}/index.jsonl",
                )
            )
        minimum_ns = int(
            float(self.pipeline_config.section("segmentation")["min_segment_duration_s"])
            * 1_000_000_000
        )
        valid_spans, removed_spans = _subtract_problems(
            SourceSpan(base_start, base_end),
            physical_problems,
            minimum_ns=minimum_ns,
        )
        if not valid_spans:
            raise ValueError("No valid Guida span remains after hard quality checks")
        output = Path(output_root).resolve()
        prepared_root = output / "prepared_segments"
        prepared_root.mkdir(parents=True, exist_ok=True)
        target_revision = prepared_root / prep_revision
        if target_revision.exists():
            raise FileExistsError(f"Prepared revision already exists: {target_revision}")
        temporary_revision = Path(
            tempfile.mkdtemp(
                prefix=f".{prep_revision}.",
                suffix=".tmp",
                dir=prepared_root,
            )
        )
        try:
            selected_assets = _selected_assets(inventory.assets, frames, imu)
            hashed_assets = [
                {
                    "asset_id": asset.asset_id,
                    "relative_path": asset.relative_path,
                    "uri": _join_raw_uri(raw_session_uri, asset.relative_path),
                    "sha256": sha256_file(root / asset.relative_path).removeprefix(
                        "sha256:"
                    ),
                }
                for asset in selected_assets
            ]
            normalized_calibrations = [
                _normalize_calibration(calibration)
                for calibration in inventory.calibrations
            ]
            segment_ids: list[str] = []
            reported_issues = list(issues)
            writer = PreparedSegmentWriter()
            for span_index, span in enumerate(valid_spans, start=1):
                segment_data, files, segment_issues = self._build_segment(
                    inventory.session_id,
                    raw_session_uri,
                    prep_revision,
                    span_index,
                    span,
                    frames,
                    imu,
                    issues,
                    hashed_assets,
                    normalized_calibrations,
                )
                reported_issues.extend(segment_issues)
                segment_id = writer.write(
                    str(temporary_revision),
                    segment_data,
                    files=files,
                )
                PreparedValidator().validate_or_raise(
                    str(temporary_revision / segment_id),
                    raw_root=raw_root,
                )
                segment_ids.append(segment_id)
            revision = self._revision_manifest(prep_revision)
            revision_errors = validate_with_schema(revision, "revision")
            if revision_errors:
                raise ValueError(f"Invalid revision.json: {'; '.join(revision_errors)}")
            (temporary_revision / "revision.json").write_text(
                json.dumps(revision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cleaning_report = {
                "zpds_version": "0.1.0",
                "profile": "guida_ego",
                "source_session": raw_session_uri,
                "source_frames": len(frames),
                "imu_samples": len(imu),
                "media_counts": media_counts,
                "valid_spans": [
                    {"start_ns": span.start_ns, "end_ns": span.end_ns}
                    for span in valid_spans
                ],
                "removed_spans": [
                    {
                        "start_ns": problem.start_ns,
                        "end_ns": problem.end_ns,
                        "reason_code": problem.code,
                        "evidence_uri": problem.evidence_uri,
                    }
                    for problem in removed_spans
                ],
                "issues": [issue.to_report() for issue in issues],
                "segments": segment_ids,
            }
            final_issues = _deduplicate_issues(reported_issues)
            cleaning_report["issues"] = [
                issue.to_report() for issue in final_issues
            ]
            (temporary_revision / "cleaning_report.json").write_text(
                json.dumps(
                    cleaning_report,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _fsync_directory_files(temporary_revision)
            os.replace(temporary_revision, target_revision)
        except BaseException:
            shutil.rmtree(temporary_revision, ignore_errors=True)
            raise
        return GuidaCleaningResult(
            revision_dir=target_revision,
            segment_ids=tuple(segment_ids),
            source_frame_count=len(frames),
            imu_sample_count=len(imu),
            issues=tuple(final_issues),
            removed_spans=tuple(removed_spans),
        )

    def _analyze(
        self,
        root: Path,
        frames: list[GuidaFrameRef],
        imu: list[GuidaImuSample],
    ) -> tuple[list[BasicQualityIssue], list[PhysicalProblem], dict[str, Any]]:
        issues: list[BasicQualityIssue] = []
        problems: list[PhysicalProblem] = []
        visual = self.thresholds["stage3_visual"]
        depth_config = self.thresholds["stage5_depth"]
        grouped = _group_frames_by_segment(frames)
        color_decoded = 0
        depth_decoded = 0
        for segment_frames in grouped.values():
            color_path = root / segment_frames[0].color_relative_path
            depth_path = root / segment_frames[0].depth_relative_path
            color_count, color_issues, color_problems = _scan_color(
                color_path,
                segment_frames,
                raw_uri=f"raw://{segment_frames[0].color_relative_path}",
                black_threshold=float(visual["black_threshold"]),
                pure_stddev=float(visual["pure_stddev_threshold"]),
                sustained_frames=int(visual["sustained_frames"]),
                freeze_frames=int(visual["freeze_frames"]),
            )
            depth_count, depth_issues, depth_problems = _scan_depth(
                depth_path,
                segment_frames,
                raw_uri=f"raw://{segment_frames[0].depth_relative_path}",
                max_invalid_ratio=float(depth_config["max_invalid_ratio"]),
                invalid_values=tuple(int(item) for item in depth_config["invalid_values"]),
                sustained_frames=int(visual["sustained_frames"]),
            )
            color_decoded += color_count
            depth_decoded += depth_count
            issues.extend((*color_issues, *depth_issues))
            problems.extend((*color_problems, *depth_problems))
        imu_issues, imu_problems = _check_imu(
            imu,
            max_gap_ns=int(float(self.thresholds["stage6_imu"]["max_gap_s"]) * 1e9),
        )
        issues.extend(imu_issues)
        problems.extend(imu_problems)
        expected = len(frames)
        if color_decoded != expected or depth_decoded != expected:
            issues.append(
                BasicQualityIssue(
                    code="required_stream_missing",
                    severity="error",
                    decision="quarantine",
                    message=(
                        f"Index has {expected} pairs, decoded color={color_decoded}, "
                        f"depth={depth_decoded}"
                    ),
                    evidence_uri="raw://index.jsonl",
                    details={
                        "index_frames": expected,
                        "color_frames": color_decoded,
                        "depth_frames": depth_decoded,
                    },
                )
            )
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        streams = meta.get("streams")
        depth_meta = streams.get("depth") if isinstance(streams, dict) else None
        if not isinstance(depth_meta, dict) or not depth_meta.get("unit"):
            issues.append(
                BasicQualityIssue(
                    code="depth_unit_unknown",
                    severity="warn",
                    decision="quarantine",
                    message="Depth unit is not explicitly recorded; no unit was inferred",
                    evidence_uri="raw://meta.json#/streams/depth",
                )
            )
        imu_meta = meta.get("imu")
        declared_imu = imu_meta.get("csv") if isinstance(imu_meta, dict) else None
        if isinstance(declared_imu, str) and not (root / declared_imu).is_file():
            issues.append(
                BasicQualityIssue(
                    code="required_stream_missing",
                    severity="warn",
                    decision="keep_with_flag",
                    message=(
                        f"meta.json declares {declared_imu}, but split IMU files were used"
                    ),
                    evidence_uri="raw://meta.json#/imu/csv",
                )
            )
        return (
            issues,
            problems,
            {
                "index_frames": expected,
                "color_decoded_frames": color_decoded,
                "depth_decoded_frames": depth_decoded,
            },
        )

    def _build_segment(
        self,
        session_id: str,
        raw_session_uri: str,
        prep_revision: str,
        span_index: int,
        span: SourceSpan,
        frames: list[GuidaFrameRef],
        imu: list[GuidaImuSample],
        issues: list[BasicQualityIssue],
        assets: list[dict[str, str]],
        calibrations: list[dict[str, Any]],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        list[BasicQualityIssue],
    ]:
        selected_frames = [
            frame for frame in frames if span.start_ns <= frame.timestamp_ns < span.end_ns
        ]
        selected_imu = [
            sample for sample in imu if span.start_ns <= sample.timestamp_ns < span.end_ns
        ]
        if not selected_frames or not selected_imu:
            raise ValueError("Prepared span lacks video frames or IMU samples")
        asset_by_path = {asset["relative_path"]: asset["asset_id"] for asset in assets}
        video_rows = [
            {
                "output_index": index,
                "segment_time_ns": frame.timestamp_ns - span.start_ns,
                "source_timestamp_ns": frame.timestamp_ns,
                "source_row": frame.source_frame_index,
                "source_seq": frame.source_seq,
                "source_segment": frame.source_segment,
                "source_asset_ids": [
                    asset_by_path[frame.color_relative_path],
                    asset_by_path[frame.depth_relative_path],
                ],
                "error_ns": 0,
            }
            for index, frame in enumerate(selected_frames)
        ]
        imu_times = [sample.timestamp_ns for sample in imu]
        max_error_ns = int(
            float(self.thresholds["alignment"]["max_nearest_imu_error_ms"]) * 1e6
        )
        video_imu_rows: list[dict[str, Any]] = []
        for index, frame in enumerate(selected_frames):
            source_row = _nearest_index(imu_times, frame.timestamp_ns)
            sample = imu[source_row]
            error_ns = abs(sample.timestamp_ns - frame.timestamp_ns)
            video_imu_rows.append(
                {
                    "output_index": index,
                    "segment_time_ns": frame.timestamp_ns - span.start_ns,
                    "source_timestamp_ns": sample.timestamp_ns,
                    "source_row": source_row,
                    "error_ns": error_ns,
                }
            )
        imu_source_rows = [
            {
                "output_index": index,
                "segment_time_ns": sample.timestamp_ns - span.start_ns,
                "source_timestamp_ns": sample.timestamp_ns,
                "source_row": sample.source_row,
                "source_asset_ids": [asset_by_path[sample.source_relative_path]],
                "error_ns": 0,
            }
            for index, sample in enumerate(selected_imu)
        ]
        maximum_error = max(row["error_ns"] for row in video_imu_rows)
        applicable_issues = [
            issue
            for issue in issues
            if issue.start_ns is None
            or issue.end_ns is None
            or _ranges_overlap(
                span.start_ns,
                span.end_ns,
                issue.start_ns,
                issue.end_ns,
            )
        ]
        if maximum_error > max_error_ns:
            applicable_issues.append(
                BasicQualityIssue(
                    code="clock_misalign",
                    severity="warn",
                    decision="quarantine",
                    message=(
                        f"Maximum nearest IMU error {maximum_error} ns exceeds "
                        f"{max_error_ns} ns"
                    ),
                    evidence_uri="alignments/video_imu_alignment.json",
                    details={
                        "max_error_ns": maximum_error,
                        "threshold_ns": max_error_ns,
                    },
                )
            )
        decision, status = _segment_quality(applicable_issues)
        segment_id = f"seg_{safe_identifier(session_id)}_{span_index:04d}"
        duration_ns = span.end_ns - span.start_ns
        color_asset_ids = sorted(
            {asset_by_path[frame.color_relative_path] for frame in selected_frames}
        )
        depth_asset_ids = sorted(
            {asset_by_path[frame.depth_relative_path] for frame in selected_frames}
        )
        imu_asset_ids = sorted(
            {asset_by_path[sample.source_relative_path] for sample in selected_imu}
        )
        index_id = asset_by_path["index.jsonl"]
        segment = {
            "zpds_version": "0.1.0",
            "prep_revision": prep_revision,
            "segment_id": segment_id,
            "source_profile": "guida_ego",
            "source_session": {
                "session_id": session_id,
                "session_uri": raw_session_uri,
            },
            "source_assets": [
                {
                    "source_asset_id": asset["asset_id"],
                    "uri": asset["uri"],
                    "sha256": asset["sha256"],
                }
                for asset in assets
            ],
            "timeline": {
                "start_ns": 0,
                "end_ns": duration_ns,
                "continuous": True,
            },
            "source_span": {
                "source_clock_id": "guida_index_timestamp",
                "start_ns": span.start_ns,
                "end_ns": span.end_ns,
            },
            "streams": [
                _source_video_stream(
                    "rgb",
                    "color",
                    "data/color.source.json",
                    duration_ns,
                    color_asset_ids,
                    index_id,
                    "camera_color_optical",
                ),
                _source_video_stream(
                    "depth",
                    "depth",
                    "data/depth.source.json",
                    duration_ns,
                    depth_asset_ids,
                    index_id,
                    "camera_depth_optical",
                ),
                {
                    "stream_id": "imu",
                    "role": "state",
                    "modality": "imu",
                    "uri": "data/imu.csv",
                    "format": "csv",
                    "encoding": "utf-8",
                    "time": {
                        "clock_id": "segment_time",
                        "sampling": "irregular",
                        "timestamp_column": "timestamp_ns",
                        "start_ns": 0,
                        "end_ns": duration_ns,
                    },
                    "fields": [
                        {
                            "name": "linear_acceleration",
                            "shape": [3],
                            "dtype": "float64",
                            "unit": "m/s^2",
                            "frame_id": "imu",
                        },
                        {
                            "name": "angular_velocity",
                            "shape": [3],
                            "dtype": "float64",
                            "unit": "rad/s",
                            "frame_id": "imu",
                        },
                    ],
                    "frame_id": "imu",
                    "origin": {
                        "kind": "deterministic_transform",
                        "producer_id": "zpds_prepare_guida_basic",
                        "source_refs": [
                            *(f"asset://{asset_id}" for asset_id in imu_asset_ids),
                            f"asset://{index_id}",
                        ],
                        "operation": "normalize_timestamp_to_segment_origin",
                        "sample_map_uri": "alignments/imu_source_map.json",
                    },
                },
            ],
            "calibration_uri": "calibration/calibrations.json",
            "alignment_uris": ["alignments/video_imu_alignment.json"],
            "quality": {
                "status": status,
                "decision": decision,
                "issues": [
                    issue.to_segment_issue(_issue_stage(issue.code))
                    for issue in applicable_issues
                ],
            },
            "producer": {
                "producer_id": "zpds_prepare_guida_basic",
                "name": "zpds",
                "version": "0.1.0",
                "code_commit": self.code_version,
                "config_version": self.pipeline_config.version,
                "config_uri": self.config_uri,
                "config_hash": self.pipeline_config.config_hash,
            },
        }
        video_map = {
            "zpds_version": "0.1.0",
            "map_type": "video_source_map",
            "source_clock_id": "guida_index_timestamp",
            "target_clock_id": "segment_time",
            "method": "identity",
            "rows": video_rows,
        }
        imu_source_map = {
            "zpds_version": "0.1.0",
            "map_type": "imu_source_map",
            "source_clock_id": "guida_imu_timestamp",
            "target_clock_id": "segment_time",
            "method": "identity",
            "rows": imu_source_rows,
        }
        video_imu_alignment = {
            "zpds_version": "0.1.0",
            "map_type": "video_imu_alignment",
            "source_clock_id": "guida_imu_timestamp",
            "target_clock_id": "segment_time",
            "method": "nearest",
            "rows": video_imu_rows,
        }
        files: dict[str, Any] = {
            "data/color.source.json": _source_selection(
                selected_frames,
                color_asset_ids,
                "color",
                span,
            ),
            "data/depth.source.json": _source_selection(
                selected_frames,
                depth_asset_ids,
                "depth",
                span,
            ),
            "data/imu.csv": _imu_csv(selected_imu, span.start_ns),
            "alignments/video_source_map.json": video_map,
            "alignments/imu_source_map.json": imu_source_map,
            "alignments/video_imu_alignment.json": video_imu_alignment,
            "calibration/calibrations.json": {
                "zpds_version": "0.1.0",
                "items": calibrations,
            },
            "quality/report.json": {
                "zpds_version": "0.1.0",
                "decision": decision,
                "status": status,
                "issues": [issue.to_report() for issue in applicable_issues],
                "max_nearest_imu_error_ns": maximum_error,
            },
        }
        return segment, files, applicable_issues

    def _revision_manifest(self, prep_revision: str) -> dict[str, Any]:
        return {
            "zpds_version": "0.1.0",
            "prep_revision": prep_revision,
            "parent_revision": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": {
                "name": "zpds",
                "version": "0.1.0",
                "code_commit": self.code_version,
                "config_version": self.pipeline_config.version,
                "config_uri": self.config_uri,
                "config_hash": self.pipeline_config.config_hash,
            },
            "conventions": {
                "time_unit": "ns",
                "time_interval": "[start_ns,end_ns)",
                "segment_time_origin_ns": 0,
                "length_unit": "m",
                "angle_unit": "rad",
                "quaternion_order": "xyzw",
                "pose_notation": "T_parent_child",
                "coordinate_system": "right-handed",
            },
            "changes": [
                "Guida basic hard-quality cleaning",
                "Source frame and nearest-IMU sample maps",
                "Manifest-first Prepared source selections without Raw transcoding",
            ],
        }


def _read_frame_refs(root: Path) -> list[GuidaFrameRef]:
    frames: list[GuidaFrameRef] = []
    segments: dict[int, tuple[str, str]] = {}
    local_indexes: dict[int, int] = {}
    with (root / "index.jsonl").open(encoding="utf-8") as file:
        for source_row, line in enumerate(file):
            value = json.loads(line)
            if not isinstance(value, dict):
                continue
            if value.get("type") == "segment_start":
                segment = int(value.get("segment", len(segments)))
                color = value.get("color_video")
                depth = value.get("depth_video")
                if not isinstance(color, str) or not isinstance(depth, str):
                    raise ValueError("segment_start must name color_video and depth_video")
                segments[segment] = (color, depth)
                local_indexes.setdefault(segment, 0)
                continue
            if value.get("type") != "frame":
                continue
            segment = int(value.get("segment", 0))
            if segment not in segments:
                raise ValueError(f"Frame references unknown segment {segment}")
            timestamp = value.get("timestamp_ns")
            sequence = value.get("seq")
            if not isinstance(timestamp, int) or not isinstance(sequence, int):
                raise TypeError(f"Invalid frame record at index row {source_row}")
            color, depth = segments[segment]
            local = local_indexes[segment]
            frames.append(
                GuidaFrameRef(
                    source_row=source_row,
                    source_seq=sequence,
                    timestamp_ns=timestamp,
                    source_segment=segment,
                    source_frame_index=local,
                    color_relative_path=color,
                    depth_relative_path=depth,
                )
            )
            local_indexes[segment] = local + 1
    if any(
        current.timestamp_ns <= previous.timestamp_ns
        for previous, current in pairwise(frames)
    ):
        raise ValueError("Guida frame timestamps must be strictly increasing")
    return frames


def _read_imu_samples(root: Path) -> list[GuidaImuSample]:
    samples: list[GuidaImuSample] = []
    source_row = 0
    for path in sorted(root.glob("imu/imu_*.csv")):
        relative = path.relative_to(root).as_posix()
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                samples.append(
                    GuidaImuSample(
                        source_row=source_row,
                        timestamp_ns=int(row["timestamp_ns"]),
                        ax=float(row["ax"]),
                        ay=float(row["ay"]),
                        az=float(row["az"]),
                        gx=float(row["gx"]),
                        gy=float(row["gy"]),
                        gz=float(row["gz"]),
                        source_relative_path=relative,
                    )
                )
                source_row += 1
    return samples


def _scan_color(
    path: Path,
    frames: list[GuidaFrameRef],
    *,
    raw_uri: str,
    black_threshold: float,
    pure_stddev: float,
    sustained_frames: int,
    freeze_frames: int,
) -> tuple[int, list[BasicQualityIssue], list[PhysicalProblem]]:
    means: list[float] = []
    stddevs: list[float] = []
    hashes: list[str] = []
    count = 0
    expected_shape: tuple[int, ...] | None = None
    shape_changes: list[int] = []
    for index, _, frame in VideoDecoder(str(path)).iter_frames():
        count += 1
        if expected_shape is None:
            expected_shape = tuple(frame.shape)
        elif tuple(frame.shape) != expected_shape:
            shape_changes.append(index)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(float(np.mean(gray)))
        stddevs.append(float(np.std(gray)))
        thumbnail = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
        hashes.append(hashlib.sha256(thumbnail.tobytes()).hexdigest())
    issues: list[BasicQualityIssue] = []
    problems: list[PhysicalProblem] = []
    black_mask = [
        mean <= black_threshold and std <= pure_stddev
        for mean, std in zip(means, stddevs)
    ]
    pure_mask = [std <= pure_stddev for std in stddevs]
    for mask in (black_mask, pure_mask):
        for start, stop in _true_runs(mask, minimum=sustained_frames):
            issue, problem = _frame_problem(
                "black_frame",
                "Sustained black or pure-color frames",
                raw_uri,
                frames,
                start,
                stop,
            )
            issues.append(issue)
            problems.append(problem)
    freeze_mask = [False] * len(hashes)
    for index in range(1, len(hashes)):
        freeze_mask[index] = hashes[index] == hashes[index - 1]
    for start, stop in _true_runs(freeze_mask, minimum=freeze_frames):
        issue, problem = _frame_problem(
            "frozen_frame",
            "Sustained exactly repeated video frames",
            raw_uri,
            frames,
            max(start - 1, 0),
            stop,
        )
        issues.append(issue)
        problems.append(problem)
    for index in shape_changes:
        if index >= len(frames):
            continue
        issue, problem = _frame_problem(
            "container_corrupt",
            "Video resolution changed within a source segment",
            raw_uri,
            frames,
            index,
            min(index + 1, len(frames)),
        )
        issues.append(issue)
        problems.append(problem)
    return count, _deduplicate_issues(issues), _merge_problem_list(problems)


def _scan_depth(
    path: Path,
    frames: list[GuidaFrameRef],
    *,
    raw_uri: str,
    max_invalid_ratio: float,
    invalid_values: tuple[int, ...],
    sustained_frames: int,
) -> tuple[int, list[BasicQualityIssue], list[PhysicalProblem]]:
    invalid_ratios: list[float] = []
    count = 0
    dtype_issue = False
    expected_shape: tuple[int, ...] | None = None
    shape_changes: list[int] = []
    for index, _, frame in VideoDecoder(str(path), convert_rgb=False).iter_frames():
        count += 1
        if frame.dtype != np.uint16:
            dtype_issue = True
        if expected_shape is None:
            expected_shape = tuple(frame.shape)
        elif tuple(frame.shape) != expected_shape:
            shape_changes.append(index)
        invalid = np.zeros(frame.shape, dtype=bool)
        for value in invalid_values:
            invalid |= frame == value
        invalid_ratios.append(float(np.mean(invalid)))
    issues: list[BasicQualityIssue] = []
    problems: list[PhysicalProblem] = []
    if dtype_issue:
        issues.append(
            BasicQualityIssue(
                code="depth_unit_unknown",
                severity="error",
                decision="quarantine",
                message="Depth decoder did not produce uint16 samples",
                evidence_uri=raw_uri,
            )
        )
    mask = [ratio > max_invalid_ratio for ratio in invalid_ratios]
    for start, stop in _true_runs(mask, minimum=sustained_frames):
        issue, problem = _frame_problem(
            "depth_invalid_ratio",
            "Sustained depth invalid ratio exceeds threshold",
            raw_uri,
            frames,
            start,
            stop,
            details={"max_invalid_ratio": max(invalid_ratios[start:stop])},
        )
        issues.append(issue)
        problems.append(problem)
    for index in shape_changes:
        if index >= len(frames):
            continue
        issue, problem = _frame_problem(
            "container_corrupt",
            "Depth resolution changed within a source segment",
            raw_uri,
            frames,
            index,
            min(index + 1, len(frames)),
        )
        issues.append(issue)
        problems.append(problem)
    return count, issues, _merge_problem_list(problems)


def _check_imu(
    samples: list[GuidaImuSample],
    *,
    max_gap_ns: int,
) -> tuple[list[BasicQualityIssue], list[PhysicalProblem]]:
    issues: list[BasicQualityIssue] = []
    problems: list[PhysicalProblem] = []
    for previous, current in pairwise(samples):
        if current.timestamp_ns < previous.timestamp_ns:
            raise ValueError("IMU timestamps must be monotonically non-decreasing")
        gap = current.timestamp_ns - previous.timestamp_ns
        if gap > max_gap_ns:
            message = f"IMU gap {gap} ns exceeds {max_gap_ns} ns"
            issues.append(
                BasicQualityIssue(
                    code="imu_gap",
                    severity="error",
                    decision="split",
                    message=message,
                    evidence_uri=f"raw://{current.source_relative_path}",
                    start_ns=previous.timestamp_ns + 1,
                    end_ns=current.timestamp_ns,
                    details={"gap_ns": gap, "threshold_ns": max_gap_ns},
                )
            )
            problems.append(
                PhysicalProblem(
                    previous.timestamp_ns + 1,
                    current.timestamp_ns,
                    "imu_gap",
                    f"raw://{current.source_relative_path}",
                )
            )
    for sample in samples:
        values = (sample.ax, sample.ay, sample.az, sample.gx, sample.gy, sample.gz)
        if not all(math.isfinite(value) for value in values):
            issues.append(
                BasicQualityIssue(
                    code="imu_saturation",
                    severity="error",
                    decision="quarantine",
                    message=f"IMU row {sample.source_row} contains NaN or Inf",
                    evidence_uri=f"raw://{sample.source_relative_path}",
                    start_ns=sample.timestamp_ns,
                    end_ns=sample.timestamp_ns + 1,
                )
            )
    return issues, problems


def _frame_problem(
    code: str,
    message: str,
    uri: str,
    frames: list[GuidaFrameRef],
    start: int,
    stop: int,
    *,
    details: dict[str, Any] | None = None,
) -> tuple[BasicQualityIssue, PhysicalProblem]:
    start_ns = frames[start].timestamp_ns
    period = _median_positive_delta(
        [frame.timestamp_ns for frame in frames],
        fallback=33_333_333,
    )
    end_index = min(max(stop - 1, start), len(frames) - 1)
    end_ns = frames[end_index].timestamp_ns + period
    return (
        BasicQualityIssue(
            code=code,
            severity="error",
            decision="split",
            message=message,
            evidence_uri=uri,
            start_ns=start_ns,
            end_ns=end_ns,
            details=details or {},
        ),
        PhysicalProblem(start_ns, end_ns, code, uri),
    )


def _true_runs(mask: list[bool], *, minimum: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate((*mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum:
                runs.append((start, index))
            start = None
    return runs


def _subtract_problems(
    base: SourceSpan,
    problems: list[PhysicalProblem],
    *,
    minimum_ns: int,
) -> tuple[list[SourceSpan], list[PhysicalProblem]]:
    clipped = [
        PhysicalProblem(
            max(base.start_ns, problem.start_ns),
            min(base.end_ns, problem.end_ns),
            problem.code,
            problem.evidence_uri,
        )
        for problem in problems
        if problem.end_ns > base.start_ns and problem.start_ns < base.end_ns
    ]
    merged = _merge_problem_list(clipped)
    valid: list[SourceSpan] = []
    cursor = base.start_ns
    for problem in merged:
        if problem.start_ns > cursor and problem.start_ns - cursor >= minimum_ns:
            valid.append(SourceSpan(cursor, problem.start_ns))
        cursor = max(cursor, problem.end_ns)
    if base.end_ns > cursor and base.end_ns - cursor >= minimum_ns:
        valid.append(SourceSpan(cursor, base.end_ns))
    return valid, merged


def _merge_problem_list(problems: list[PhysicalProblem]) -> list[PhysicalProblem]:
    if not problems:
        return []
    ordered = sorted(problems, key=lambda item: (item.start_ns, item.end_ns))
    merged = [ordered[0]]
    for problem in ordered[1:]:
        previous = merged[-1]
        if problem.start_ns <= previous.end_ns:
            merged[-1] = PhysicalProblem(
                previous.start_ns,
                max(previous.end_ns, problem.end_ns),
                previous.code if previous.code == problem.code else "multiple_hard_failures",
                previous.evidence_uri,
            )
        else:
            merged.append(problem)
    return merged


def _group_frames_by_segment(
    frames: list[GuidaFrameRef],
) -> dict[int, list[GuidaFrameRef]]:
    result: dict[int, list[GuidaFrameRef]] = {}
    for frame in frames:
        result.setdefault(frame.source_segment, []).append(frame)
    return result


def _nearest_index(values: list[int], target: int) -> int:
    index = bisect.bisect_left(values, target)
    if index <= 0:
        return 0
    if index >= len(values):
        return len(values) - 1
    before = values[index - 1]
    after = values[index]
    return index - 1 if target - before <= after - target else index


def _median_positive_delta(values: list[int], *, fallback: int) -> int:
    deltas = sorted(
        current - previous
        for previous, current in pairwise(values)
        if current > previous
    )
    return deltas[len(deltas) // 2] if deltas else fallback


def _selected_assets(
    assets: list[Any],
    frames: list[GuidaFrameRef],
    imu: list[GuidaImuSample],
) -> list[Any]:
    selected_paths = {
        "meta.json",
        "index.jsonl",
        *(frame.color_relative_path for frame in frames),
        *(frame.depth_relative_path for frame in frames),
        *(sample.source_relative_path for sample in imu),
    }
    selected = [asset for asset in assets if asset.relative_path in selected_paths]
    missing = selected_paths - {asset.relative_path for asset in selected}
    if missing:
        raise ValueError(f"Inventory lacks selected source assets: {sorted(missing)}")
    return selected


def _normalize_calibration(calibration: CalibrationDescriptor) -> dict[str, Any]:
    metadata = json.loads(json.dumps(calibration.metadata))
    source_metadata = json.loads(json.dumps(calibration.metadata))
    if (
        isinstance(metadata, dict)
        and metadata.get("translation_unit") == "mm"
        and isinstance(metadata.get("translation"), list)
    ):
        metadata["translation"] = [
            float(value) / 1000.0 for value in metadata["translation"]
        ]
        metadata["translation_unit"] = "m"
        metadata["source_value"] = source_metadata
        metadata["conversion"] = "mm_to_m"
    return {
        "calibration_id": calibration.calibration_id,
        "kind": calibration.kind,
        "uri": calibration.uri,
        "parent_frame": calibration.parent_frame,
        "child_frame": calibration.child_frame,
        "format": calibration.format,
        "source_recorded": calibration.source_recorded,
        "metadata": metadata,
    }


def _source_video_stream(
    stream_id: str,
    modality: str,
    uri: str,
    duration_ns: int,
    asset_ids: list[str],
    index_id: str,
    frame_id: str,
) -> dict[str, Any]:
    return {
        "stream_id": stream_id,
        "role": "observation",
        "modality": modality,
        "uri": uri,
        "format": "source_selection_v1",
        "time": {
            "clock_id": "segment_time",
            "sampling": "irregular",
            "timestamp_column": "segment_time_ns",
            "start_ns": 0,
            "end_ns": duration_ns,
        },
        "frame_id": frame_id,
        "origin": {
            "kind": "source_recorded",
            "producer_id": "source_guida",
            "source_refs": [
                *(f"asset://{asset_id}" for asset_id in asset_ids),
                f"asset://{index_id}",
            ],
            "sample_map_uri": "alignments/video_source_map.json",
        },
    }


def _source_selection(
    frames: list[GuidaFrameRef],
    asset_ids: list[str],
    modality: str,
    span: SourceSpan,
) -> dict[str, Any]:
    return {
        "zpds_version": "0.1.0",
        "materialized": False,
        "modality": modality,
        "source_asset_ids": asset_ids,
        "source_start_ns": span.start_ns,
        "source_end_ns": span.end_ns,
        "selected_frame_count": len(frames),
        "mapping": "alignments/video_source_map.json",
        "note": "Raw is immutable; media transcoding is deferred and not implied",
    }


def _imu_csv(samples: list[GuidaImuSample], start_ns: int) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "timestamp_ns",
            "source_timestamp_ns",
            "source_row",
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ]
    )
    for sample in samples:
        writer.writerow(
            [
                sample.timestamp_ns - start_ns,
                sample.timestamp_ns,
                sample.source_row,
                sample.ax,
                sample.ay,
                sample.az,
                sample.gx,
                sample.gy,
                sample.gz,
            ]
        )
    return output.getvalue()


def _segment_quality(
    issues: list[BasicQualityIssue],
) -> tuple[str, str]:
    if any(issue.decision in {"reject", "quarantine"} for issue in issues):
        return "quarantine", "quarantine"
    if issues:
        return "keep_with_flag", "pass"
    return "keep", "pass"


def _issue_stage(code: str) -> int:
    if code in {"black_frame", "frozen_frame", "container_corrupt"}:
        return 3
    if code in {"depth_invalid_ratio", "depth_unit_unknown"}:
        return 5
    if code in {"imu_gap", "imu_saturation"}:
        return 6
    if code == "clock_misalign":
        return 2
    return 1


def _ranges_overlap(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    return first_start < second_end and second_start < first_end


def _join_raw_uri(session_uri: str, relative: str) -> str:
    return f"{session_uri.rstrip('/')}/{relative}"


def _raw_root_for_session(session_path: Path, session_uri: str) -> Path:
    logical = PurePosixPath(session_uri.removeprefix("raw://"))
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise ValueError(f"Invalid raw session URI: {session_uri!r}")
    raw_root = session_path
    for _ in logical.parts:
        raw_root = raw_root.parent
    return raw_root


def _load_thresholds(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict) or value.get("profile") != "guida_ego":
        raise ValueError(f"Invalid Guida threshold config: {path}")
    required = {
        "stage3_visual",
        "stage5_depth",
        "stage6_imu",
        "alignment",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Guida threshold config lacks sections: {sorted(missing)}")
    return value


def _deduplicate_issues(
    issues: list[BasicQualityIssue],
) -> list[BasicQualityIssue]:
    result: list[BasicQualityIssue] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for issue in issues:
        key = (issue.code, issue.start_ns, issue.end_ns)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _fsync_directory_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            # Windows 的 CRT 不允许对只读文件描述符调用 fsync。
            with path.open("r+b") as file:
                os.fsync(file.fileno())
