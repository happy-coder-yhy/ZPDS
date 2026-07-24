"""墨现 Guida V2 只读 Adapter。"""

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zpds.core.types import (
    CalibrationDescriptor,
    ClockDescriptor,
    ClockDomain,
    SessionInventory,
    SourceStream,
    StreamKind,
)

from .base import BaseAdapter
from .common import require_directory, source_asset
from .contracts import IssueLevel, ValidationIssue, ValidationReport
from .video import VideoInspector


@dataclass(frozen=True)
class GuidaIndexStats:
    frame_count: int
    drop_count: int
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None
    max_gap_ns: int
    regressions: int
    malformed_lines: int
    segment_files: tuple[str, ...]


class GuidaAdapter(BaseAdapter):
    """以 index.jsonl 为权威时间轴，不把预览 MP4 当作真值。"""

    profile_name = "guida_ego"

    def inspect(self, path: str) -> SessionInventory:
        root = require_directory(path)
        meta = _load_json_object(root / "meta.json")
        stats = self.index_stats(root)
        streams = self._streams(meta, root)
        asset_paths = self._asset_paths(root)
        return SessionInventory(
            session_id=_session_id(meta, root),
            source_profile=self.profile_name,
            session_uri=str(root),
            assets=[source_asset(item, root) for item in asset_paths],
            streams=streams,
            clocks=self.read_clock_catalog(str(root)),
            calibrations=self._calibrations(meta),
            total_frames=stats.frame_count,
            duration_s=(
                (stats.last_timestamp_ns - stats.first_timestamp_ns) / 1_000_000_000
                if stats.first_timestamp_ns is not None
                and stats.last_timestamp_ns is not None
                else 0.0
            ),
            clock_domain=ClockDomain.UNIX_UTC,
            metadata={
                "index_authoritative": True,
                "frame_drops": stats.drop_count,
                "max_gap_ns": stats.max_gap_ns,
                "timestamp_regressions": stats.regressions,
                "malformed_index_lines": stats.malformed_lines,
                "declared_imu_path": _nested(meta, "imu", "csv"),
                "actual_imu_files": [
                    item.relative_to(root).as_posix()
                    for item in sorted(root.glob("imu/imu_*.csv"))
                ],
            },
        )

    def validate(self, path: str) -> ValidationReport:
        root = Path(path)
        issues: list[ValidationIssue] = []
        if not root.is_dir():
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="guida_session_missing",
                        level=IssueLevel.FATAL,
                        message=f"Guida session directory not found: {root}",
                        path=str(root),
                    ),
                )
            )
        meta_path = root / "meta.json"
        index_path = root / "index.jsonl"
        for required in (meta_path, index_path):
            if not required.is_file():
                issues.append(
                    ValidationIssue(
                        code="required_file_missing",
                        level=IssueLevel.ERROR,
                        message=f"Required Guida file is missing: {required.name}",
                        path=str(required),
                    )
                )
        if issues:
            return ValidationReport(issues=tuple(issues))
        try:
            meta = _load_json_object(meta_path)
            stats = self.index_stats(root)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="guida_metadata_invalid",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(root),
                    ),
                ),
                checked_assets=2,
            )
        if stats.malformed_lines:
            issues.append(
                ValidationIssue(
                    code="index_record_invalid",
                    level=IssueLevel.ERROR,
                    message=f"{stats.malformed_lines} malformed index records",
                    path=str(index_path),
                )
            )
        if stats.regressions:
            issues.append(
                ValidationIssue(
                    code="timestamp_regression",
                    level=IssueLevel.ERROR,
                    message=f"{stats.regressions} timestamp regressions in index.jsonl",
                    path=str(index_path),
                )
            )
        for relative in stats.segment_files:
            if not (root / relative).is_file():
                issues.append(
                    ValidationIssue(
                        code="segment_container_missing",
                        level=IssueLevel.ERROR,
                        message=f"Indexed video container is missing: {relative}",
                        path=str(root / relative),
                    )
                )
        declared_imu = _nested(meta, "imu", "csv")
        actual_imu = tuple(sorted(root.glob("imu/imu_*.csv")))
        if isinstance(declared_imu, str) and not (root / declared_imu).is_file():
            issues.append(
                ValidationIssue(
                    code="imu_declared_path_missing",
                    level=IssueLevel.WARN if actual_imu else IssueLevel.ERROR,
                    message=(
                        f"meta.json declares {declared_imu!r}; "
                        f"discovered {len(actual_imu)} imu/imu_*.csv file(s)"
                    ),
                    path=str(meta_path),
                )
            )
        if not actual_imu:
            issues.append(
                ValidationIssue(
                    code="imu_file_missing",
                    level=IssueLevel.ERROR,
                    message="No imu/imu_*.csv file was discovered",
                    path=str(root / "imu"),
                )
            )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=len(self._asset_paths(root)),
            checked_records=stats.frame_count + stats.drop_count,
            metadata={
                "frame_count": stats.frame_count,
                "max_gap_ns": stats.max_gap_ns,
            },
        )

    def scan(self, path: str) -> ValidationReport:
        root = require_directory(path)
        base = self.validate(str(root))
        issues = list(base.issues)
        checked = base.checked_records
        decoded = 0
        for csv_path in sorted(root.glob("imu/imu_*.csv")):
            try:
                for _ in self.iter_imu(csv_path):
                    checked += 1
                    decoded += 1
            except (OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="imu_csv_read_failed",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(csv_path),
                    )
                )
        stats = self.index_stats(root)
        for relative in stats.segment_files:
            video = root / relative
            if not video.is_file():
                continue
            report = VideoInspector().scan(str(video))
            checked += report.checked_records
            decoded += report.decoded_records
            issues.extend(report.issues)
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=base.checked_assets,
            checked_records=checked,
            decoded_records=decoded,
            metadata=base.metadata,
        )

    def analyze_time(self, path: str, *, max_gap_ns: int = 40_000_000) -> ValidationReport:
        root = require_directory(path)
        stats = self.index_stats(root)
        issues: list[ValidationIssue] = []
        if stats.regressions:
            issues.append(
                ValidationIssue(
                    code="timestamp_regression",
                    level=IssueLevel.ERROR,
                    message=f"{stats.regressions} index timestamp regressions",
                    path=str(root / "index.jsonl"),
                )
            )
        if stats.max_gap_ns > max_gap_ns:
            issues.append(
                ValidationIssue(
                    code="timestamp_gap",
                    level=IssueLevel.WARN,
                    message=(
                        f"maximum index gap {stats.max_gap_ns} ns exceeds "
                        f"{max_gap_ns} ns"
                    ),
                    path=str(root / "index.jsonl"),
                    details={"max_gap_ns": stats.max_gap_ns, "threshold_ns": max_gap_ns},
                )
            )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=1,
            checked_records=stats.frame_count,
            metadata={
                "clock_id": "guida_index_timestamp",
                "max_gap_ns": stats.max_gap_ns,
                "frame_count": stats.frame_count,
            },
        )

    def read_clock_catalog(self, path: str) -> list[ClockDescriptor]:
        del path
        return [
            ClockDescriptor(
                clock_id="guida_index_timestamp",
                domain=ClockDomain.UNIX_UTC,
                source="index.jsonl frame.timestamp_ns",
                authoritative=True,
            ),
            ClockDescriptor(
                clock_id="guida_imu_timestamp",
                domain=ClockDomain.UNIX_UTC,
                source="imu/imu_*.csv timestamp_ns",
                notes="Preserved independently until alignment produces a sample map",
            ),
            ClockDescriptor(
                clock_id="video_container_time",
                domain=ClockDomain.CUSTOM_EPOCH,
                source="MKV/MP4 presentation timestamp",
                notes="QA only; index.jsonl remains authoritative",
            ),
        ]

    def index_stats(self, path: str | Path) -> GuidaIndexStats:
        root = require_directory(path)
        frame_count = 0
        drop_count = 0
        first: int | None = None
        previous: int | None = None
        last: int | None = None
        max_gap = 0
        regressions = 0
        malformed = 0
        segment_files: set[str] = set()
        with (root / "index.jsonl").open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError("record is not an object")
                except (json.JSONDecodeError, TypeError):
                    malformed += 1
                    continue
                record_type = record.get("type")
                if line_number == 1 and record_type is None:
                    continue
                if record_type == "segment_start":
                    for field in ("color_video", "depth_video"):
                        value = record.get(field)
                        if isinstance(value, str):
                            segment_files.add(value)
                    continue
                if record_type == "frame_drop":
                    drop_count += 1
                    continue
                if record_type != "frame":
                    continue
                timestamp = record.get("timestamp_ns")
                if not isinstance(timestamp, int):
                    malformed += 1
                    continue
                frame_count += 1
                if first is None:
                    first = timestamp
                if previous is not None:
                    if timestamp < previous:
                        regressions += 1
                    else:
                        max_gap = max(max_gap, timestamp - previous)
                previous = timestamp
                last = timestamp
        return GuidaIndexStats(
            frame_count=frame_count,
            drop_count=drop_count,
            first_timestamp_ns=first,
            last_timestamp_ns=last,
            max_gap_ns=max_gap,
            regressions=regressions,
            malformed_lines=malformed,
            segment_files=tuple(sorted(segment_files)),
        )

    def iter_imu(self, path: str | Path) -> Iterator[dict[str, float | int]]:
        with Path(path).open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            required = {"timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError(f"Invalid Guida IMU columns: {reader.fieldnames}")
            for row in reader:
                yield {
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "ax": float(row["ax"]),
                    "ay": float(row["ay"]),
                    "az": float(row["az"]),
                    "gx": float(row["gx"]),
                    "gy": float(row["gy"]),
                    "gz": float(row["gz"]),
                }

    @staticmethod
    def _streams(meta: dict[str, Any], root: Path) -> list[SourceStream]:
        streams: list[SourceStream] = []
        stream_config = meta.get("streams", {})
        if isinstance(stream_config, dict):
            for name, kind in (("color", StreamKind.COLOR), ("depth", StreamKind.DEPTH)):
                value = stream_config.get(name, {})
                if not isinstance(value, dict) or value.get("enabled") is False:
                    continue
                streams.append(
                    SourceStream(
                        kind=kind,
                        stream_id=name,
                        role="observation",
                        clock_id="guida_index_timestamp",
                        width=_optional_int(value.get("width")),
                        height=_optional_int(value.get("height")),
                        fps=_optional_float(value.get("fps")),
                        codec=str(value.get("format", "")) or None,
                        container="mkv",
                        frame_id=f"camera_{name}_optical",
                    )
                )
        imu = meta.get("imu", {})
        if isinstance(imu, dict) and tuple(root.glob("imu/imu_*.csv")):
            streams.append(
                SourceStream(
                    kind=StreamKind.IMU,
                    stream_id="imu",
                    role="state",
                    clock_id="guida_imu_timestamp",
                    sample_rate_hz=_optional_float(imu.get("sample_rate_hz")),
                    container="csv",
                    encoding="float",
                    frame_id="imu",
                    metadata={
                        "accel_unit": imu.get("accel_unit"),
                        "gyro_unit": imu.get("gyro_unit"),
                    },
                )
            )
        return streams

    @staticmethod
    def _asset_paths(root: Path) -> tuple[Path, ...]:
        patterns = (
            "meta.json",
            "index.jsonl",
            "color_*.mkv",
            "depth_*.mkv",
            "color*.mp4",
            "depth*.mp4",
            "imu/imu_*.csv",
            "log/*",
        )
        return tuple(
            sorted(
                {
                    item
                    for pattern in patterns
                    for item in root.glob(pattern)
                    if item.is_file()
                }
            )
        )

    @staticmethod
    def _calibrations(meta: dict[str, Any]) -> list[CalibrationDescriptor]:
        calibrations: list[CalibrationDescriptor] = []
        streams = meta.get("streams")
        if isinstance(streams, dict):
            for name in ("color", "depth"):
                stream = streams.get(name)
                if not isinstance(stream, dict):
                    continue
                intrinsics = stream.get("intrinsics")
                if isinstance(intrinsics, dict):
                    calibrations.append(
                        CalibrationDescriptor(
                            calibration_id=f"{name}_intrinsics",
                            kind="camera_intrinsics",
                            uri=f"raw://meta.json#/streams/{name}/intrinsics",
                            child_frame=f"camera_{name}_optical",
                            format="json_fragment",
                            metadata=dict(intrinsics),
                        )
                    )
            depth = streams.get("depth")
            if isinstance(depth, dict) and isinstance(
                depth.get("extrinsics_to_color"), dict
            ):
                calibrations.append(
                    CalibrationDescriptor(
                        calibration_id="depth_to_color",
                        kind="rigid_extrinsics",
                        uri="raw://meta.json#/streams/depth/extrinsics_to_color",
                        parent_frame="camera_color_optical",
                        child_frame="camera_depth_optical",
                        format="rotation_matrix_translation",
                        metadata=dict(depth["extrinsics_to_color"]),
                    )
                )
        imu = meta.get("imu")
        if isinstance(imu, dict) and isinstance(imu.get("extrinsics_to_depth"), dict):
            calibrations.append(
                CalibrationDescriptor(
                    calibration_id="imu_to_depth",
                    kind="rigid_extrinsics",
                    uri="raw://meta.json#/imu/extrinsics_to_depth",
                    parent_frame="camera_depth_optical",
                    child_frame="imu",
                    format="rotation_matrix_translation",
                    metadata=dict(imu["extrinsics_to_depth"]),
                )
            )
        return calibrations


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _nested(value: dict[str, Any], first: str, second: str) -> Any:
    child = value.get(first)
    return child.get(second) if isinstance(child, dict) else None


def _session_id(meta: dict[str, Any], root: Path) -> str:
    session = meta.get("session")
    if isinstance(session, dict):
        name = session.get("output_folder_name")
        if isinstance(name, str) and name:
            return name
    return root.name


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
