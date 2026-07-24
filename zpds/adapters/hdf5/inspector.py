"""HDF5 文件探测器。"""

from pathlib import Path
from typing import Any

from zpds.adapters.base import BaseAdapter
from zpds.adapters.common import (
    infer_stream_kind,
    require_file,
    require_optional_module,
    source_asset,
)
from zpds.adapters.contracts import IssueLevel, ValidationIssue, ValidationReport
from zpds.core.types import (
    ClockDescriptor,
    ClockDomain,
    SessionInventory,
    SourceStream,
)


class Hdf5Inspector(BaseAdapter):
    """HDF5 文件探测器。"""

    def inspect(self, path: str) -> SessionInventory:
        source = require_file(path)
        h5py = require_optional_module("h5py", "hdf5")
        streams: list[SourceStream] = []
        with h5py.File(source, "r") as file:
            def collect(name: str, value: Any) -> None:
                if not hasattr(value, "shape"):
                    return
                shape = tuple(int(item) for item in value.shape)
                streams.append(
                    SourceStream(
                        kind=infer_stream_kind(name),
                        stream_id=name,
                        role="state",
                        clock_id="hdf5_row_index",
                        dtype=str(value.dtype),
                        container="hdf5",
                        metadata={"shape": shape, "chunks": value.chunks},
                    )
                )

            file.visititems(collect)
        return SessionInventory(
            session_id=source.stem,
            source_profile="hdf5",
            session_uri=str(source),
            assets=[source_asset(source, source.parent)],
            streams=streams,
            clocks=[
                ClockDescriptor(
                    clock_id="hdf5_row_index",
                    domain=ClockDomain.CUSTOM_EPOCH,
                    source="dataset row index; timestamp dataset required for physical time",
                )
            ],
            metadata={"dataset_count": len(streams)},
        )

    def validate(self, path: str) -> ValidationReport:
        source = Path(path)
        if not source.is_file():
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="hdf5_missing",
                        level=IssueLevel.FATAL,
                        message=f"HDF5 file not found: {source}",
                        path=str(source),
                    ),
                )
            )
        with source.open("rb") as file:
            magic = file.read(8)
        if magic != b"\x89HDF\r\n\x1a\n":
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="hdf5_magic_invalid",
                        level=IssueLevel.ERROR,
                        message="HDF5 magic header is invalid",
                        path=str(source),
                    ),
                ),
                checked_assets=1,
            )
        try:
            inventory = self.inspect(str(source))
        except ImportError as error:
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="hdf5_dependency_missing",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(source),
                    ),
                ),
                checked_assets=1,
            )
        return ValidationReport(
            checked_assets=1,
            checked_records=len(inventory.streams),
        )

    def scan(self, path: str) -> ValidationReport:
        source = require_file(path)
        h5py = require_optional_module("h5py", "hdf5")
        checked = 0
        dataset_count = 0
        element_count = 0
        issues: list[ValidationIssue] = []
        with h5py.File(source, "r") as file:
            def scan_dataset(name: str, value: Any) -> None:
                nonlocal checked, dataset_count, element_count
                if not hasattr(value, "shape"):
                    return
                dataset_count += 1
                try:
                    if value.ndim == 0:
                        value[()]
                        checked += 1
                        element_count += 1
                    elif value.shape[0] > 0:
                        chunk_rows = int(value.chunks[0]) if value.chunks else 1024
                        for start in range(0, int(value.shape[0]), max(chunk_rows, 1)):
                            stop = min(start + max(chunk_rows, 1), int(value.shape[0]))
                            chunk = value[start:stop]
                            checked += stop - start
                            element_count += int(chunk.size)
                except (OSError, ValueError) as error:
                    issues.append(
                        ValidationIssue(
                            code="hdf5_dataset_read_failed",
                            level=IssueLevel.ERROR,
                            message=str(error),
                            path=str(source),
                            stream_id=name,
                        )
                    )

            file.visititems(scan_dataset)
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=1,
            checked_records=checked,
            decoded_records=checked if not issues else 0,
            metadata={
                "dataset_count": dataset_count,
                "rows_or_scalars_read": checked,
                "elements_read": element_count,
                "full_dataset_scan": True,
            },
        )
