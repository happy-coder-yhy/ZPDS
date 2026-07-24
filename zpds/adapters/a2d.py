"""A2D 真机目录只读 Adapter。"""

import json
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
from .hdf5 import Hdf5Inspector
from .log import LogParser
from .mcap import McapInspector, McapReader, Ros2McapReader

CAMERA_FILES = (
    "head_color.jpg",
    "head_depth.png",
    "hand_left_color.jpg",
    "hand_left_depth.png",
    "hand_right_color.jpg",
    "hand_right_depth.png",
)


class A2DAdapter(BaseAdapter):
    profile_name = "a2d_robot"

    def inspect(self, path: str) -> SessionInventory:
        root = require_directory(path)
        meta = _load_json(root / "meta_info.json")
        frame_dirs = self._frame_directories(root)
        assets = [source_asset(item, root) for item in self._asset_paths(root)]
        streams = [
            SourceStream(
                kind=StreamKind.DEPTH if "depth" in name else StreamKind.COLOR,
                stream_id=name.rsplit(".", 1)[0],
                role="observation",
                clock_id="camera_frame_index",
                fps=self._fps(meta, name),
                container=Path(name).suffix.lstrip("."),
                frame_id=name.rsplit("_", 1)[0],
                metadata={"mapping": "frame_index_inferred"},
            )
            for name in CAMERA_FILES
        ]
        for hdf5 in sorted(root.rglob("*.h5")):
            relative = hdf5.relative_to(root).as_posix()
            hdf5_inventory = Hdf5Inspector().inspect(str(hdf5))
            streams.extend(
                SourceStream(
                    kind=stream.kind,
                    stream_id=f"{relative}:{stream.stream_id}",
                    role="state",
                    clock_id="hdf5_row_index",
                    dtype=stream.dtype,
                    container="hdf5",
                    metadata={
                        **stream.metadata,
                        "source_file": relative,
                        "row_to_frame_mapping": "unproven",
                    },
                )
                for stream in hdf5_inventory.streams
            )
        for mcap in sorted(root.rglob("*.mcap")):
            relative = mcap.relative_to(root).as_posix()
            mcap_inventory = McapInspector().inspect(str(mcap))
            streams.extend(
                SourceStream(
                    kind=stream.kind,
                    stream_id=f"{relative}:{stream.stream_id}",
                    role=stream.role,
                    clock_id="ros_mcap_log_time",
                    topic=stream.topic,
                    encoding=stream.encoding,
                    container="mcap",
                    metadata={**stream.metadata, "source_file": relative},
                )
                for stream in mcap_inventory.streams
            )
        calibrations = [
            CalibrationDescriptor(
                calibration_id=path.stem,
                kind="camera_intrinsics",
                uri=f"raw://{path.relative_to(root).as_posix()}",
                child_frame=path.stem.removesuffix("_intrinsic_params"),
                format="json",
            )
            for path in sorted(root.glob("parameters/camera/*_intrinsic_params.json"))
        ]
        for mcap in sorted(root.rglob("*.mcap")):
            relative = mcap.relative_to(root).as_posix()
            calibrations.extend(
                CalibrationDescriptor(
                    calibration_id=f"{mcap.stem}_{calibration.calibration_id}",
                    kind=calibration.kind,
                    uri=f"raw://{relative}#{calibration.metadata.get('topic', '')}",
                    parent_frame=calibration.parent_frame,
                    child_frame=calibration.child_frame,
                    format=calibration.format,
                    metadata=calibration.metadata,
                )
                for calibration in McapInspector().inspect(str(mcap)).calibrations
            )
        return SessionInventory(
            session_id=str(meta.get("episode_token") or meta.get("episode_id") or root.name),
            source_profile=self.profile_name,
            session_uri=str(root),
            assets=assets,
            streams=streams,
            clocks=self.read_clock_catalog(str(root)),
            calibrations=calibrations,
            total_frames=len(frame_dirs),
            duration_s=float(meta.get("duration", 0.0)),
            clock_domain=ClockDomain.CUSTOM_EPOCH,
            metadata={
                "frame_directory_count": len(frame_dirs),
                "camera_file_count": sum(
                    1 for directory in frame_dirs for item in directory.iterdir() if item.is_file()
                ),
                "frame_mapping": "inferred_not_ground_truth",
            },
        )

    def validate(self, path: str) -> ValidationReport:
        root = Path(path)
        if not root.is_dir():
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="a2d_session_missing",
                        level=IssueLevel.FATAL,
                        message=f"A2D session directory not found: {root}",
                    ),
                )
            )
        issues: list[ValidationIssue] = []
        meta = root / "meta_info.json"
        if not meta.is_file():
            issues.append(
                ValidationIssue(
                    code="a2d_meta_missing",
                    level=IssueLevel.ERROR,
                    message="meta_info.json is missing",
                    path=str(meta),
                )
            )
        frame_dirs = self._frame_directories(root)
        if not frame_dirs:
            issues.append(
                ValidationIssue(
                    code="a2d_camera_frames_missing",
                    level=IssueLevel.ERROR,
                    message="No numeric camera frame directories were found",
                    path=str(root / "camera"),
                )
            )
        incomplete: list[dict[str, Any]] = []
        for directory in frame_dirs:
            missing = [name for name in CAMERA_FILES if not (directory / name).is_file()]
            if missing:
                incomplete.append({"frame": int(directory.name), "missing": missing})
        if incomplete:
            issues.append(
                ValidationIssue(
                    code="a2d_camera_tuple_incomplete",
                    level=IssueLevel.WARN,
                    message=f"{len(incomplete)} frame(s) lack the complete 3×(color,depth) tuple",
                    path=str(root / "camera"),
                    details={
                        "incomplete_count": len(incomplete),
                        "examples": incomplete[:20],
                    },
                )
            )
        hdf5_files = tuple(sorted(root.rglob("*.h5")))
        mcap_files = tuple(sorted(root.rglob("*.mcap")))
        for hdf5 in hdf5_files:
            try:
                issues.extend(Hdf5Inspector().validate(str(hdf5)).issues)
            except (ImportError, OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="a2d_hdf5_validation_failed",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(hdf5),
                    )
                )
        for mcap in mcap_files:
            try:
                issues.extend(McapInspector().validate(str(mcap)).issues)
            except (ImportError, OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="a2d_mcap_validation_failed",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(mcap),
                    )
                )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=len(self._asset_paths(root)),
            checked_records=len(frame_dirs),
            metadata={
                "complete_frames": len(frame_dirs) - len(incomplete),
                "incomplete_frames": len(incomplete),
                "hdf5_files": len(hdf5_files),
                "mcap_files": len(mcap_files),
            },
        )

    def scan(self, path: str) -> ValidationReport:
        root = require_directory(path)
        report = self.validate(str(root))
        checked = 0
        issues = list(report.issues)
        for directory in self._frame_directories(root):
            for filename in CAMERA_FILES:
                source = directory / filename
                if not source.is_file():
                    continue
                checked += 1
                try:
                    with source.open("rb") as file:
                        header = file.read(8)
                    valid = (
                        header.startswith(b"\xff\xd8\xff")
                        if source.suffix.lower() == ".jpg"
                        else header == b"\x89PNG\r\n\x1a\n"
                    )
                    if not valid:
                        raise ValueError("image magic is invalid")
                except (OSError, ValueError) as error:
                    issues.append(
                        ValidationIssue(
                            code="a2d_image_invalid",
                            level=IssueLevel.ERROR,
                            message=str(error),
                            path=str(source),
                        )
                    )
        for hdf5 in sorted(root.rglob("*.h5")):
            try:
                hdf5_report = Hdf5Inspector().scan(str(hdf5))
                checked += hdf5_report.checked_records
                issues.extend(hdf5_report.issues)
            except (ImportError, OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="a2d_hdf5_read_failed",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(hdf5),
                    )
                )
        decoded = checked - sum(issue.code == "a2d_image_invalid" for issue in issues)
        for mcap in sorted(root.rglob("*.mcap")):
            try:
                raw_count = sum(1 for _ in McapReader(str(mcap)).iter_messages())
                decoded_count = sum(1 for _ in Ros2McapReader(str(mcap)).iter_decoded())
                checked += raw_count
                decoded += decoded_count
                if raw_count != decoded_count:
                    issues.append(
                        ValidationIssue(
                            code="a2d_mcap_decode_count_mismatch",
                            level=IssueLevel.ERROR,
                            message=(
                                f"Read {raw_count} raw messages but decoded "
                                f"{decoded_count} ROS2 messages"
                            ),
                            path=str(mcap),
                        )
                    )
            except (ImportError, OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="a2d_mcap_read_failed",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(mcap),
                    )
                )
        for log_path in sorted(root.glob("logs/*.log")):
            try:
                summary = LogParser().parse(str(log_path))
                event_count = int(summary["event_count"])
                checked += event_count
                decoded += event_count
            except (OSError, TypeError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="a2d_log_read_failed",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(log_path),
                    )
                )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=report.checked_assets,
            checked_records=checked,
            decoded_records=decoded,
            metadata=report.metadata,
        )

    def read_clock_catalog(self, path: str) -> list[ClockDescriptor]:
        del path
        return [
            ClockDescriptor(
                clock_id="camera_frame_index",
                domain=ClockDomain.CUSTOM_EPOCH,
                source="camera/<frame_index> directory",
                notes="Index is not a proven timestamp or HDF5 row mapping",
            ),
            ClockDescriptor(
                clock_id="hdf5_row_index",
                domain=ClockDomain.CUSTOM_EPOCH,
                source="HDF5 dataset row",
                notes="Must not be equated with camera frame index without sample map",
            ),
            ClockDescriptor(
                clock_id="ros_mcap_log_time",
                domain=ClockDomain.ROS_TIME,
                source="record MCAP message.log_time",
                notes="Preserved independently from message header timestamps",
            ),
            ClockDescriptor(
                clock_id="ros_message_header_time",
                domain=ClockDomain.ROS_TIME,
                source="decoded ROS message header timestamp when present",
                notes="No equality with camera frame index or HDF5 row is assumed",
            ),
        ]

    @staticmethod
    def _frame_directories(root: Path) -> tuple[Path, ...]:
        camera = root / "camera"
        if not camera.is_dir():
            return ()
        return tuple(
            sorted(
                (item for item in camera.iterdir() if item.is_dir() and item.name.isdigit()),
                key=lambda item: int(item.name),
            )
        )

    @staticmethod
    def _asset_paths(root: Path) -> tuple[Path, ...]:
        return tuple(sorted(item for item in root.rglob("*") if item.is_file()))

    @staticmethod
    def _fps(meta: dict[str, Any], filename: str) -> float | None:
        fps = meta.get("fps")
        if not isinstance(fps, dict):
            return None
        role = "head" if filename.startswith("head") else filename.split("_color")[0].split("_depth")[0]
        value = fps.get(role)
        if not isinstance(value, (str, int, float)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value
