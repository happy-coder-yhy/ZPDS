"""Prepared Segment 的结构、时间、映射和源追溯校验。"""

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from zpds.utils.schema_validator import load_json, validate_with_schema


class PreparedValidator:
    """验证机器 Schema、内部文件、时间单调性和可选 Raw hash。"""

    def validate(
        self,
        segment_dir: str,
        *,
        raw_root: str | Path | None = None,
    ) -> list[str]:
        root = Path(segment_dir).resolve()
        segment_path = root / "segment.json"
        if not segment_path.is_file():
            return [f"segment.json is missing: {segment_path}"]
        try:
            segment = load_json(segment_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return [f"segment.json cannot be read: {error}"]
        errors = validate_with_schema(segment, "segment")
        for relative in _referenced_files(segment):
            try:
                path = _safe_existing_child(root, relative)
            except ValueError as error:
                errors.append(str(error))
                continue
            if not path.is_file():
                errors.append(f"referenced Prepared file is missing: {relative}")
        for relative in _sample_map_files(segment):
            try:
                path = _safe_existing_child(root, relative)
            except ValueError:
                continue
            if not path.is_file():
                continue
            try:
                value = load_json(path)
                errors.extend(
                    f"{relative}: {error}"
                    for error in validate_with_schema(value, "sample_map")
                )
                errors.extend(_sample_map_semantic_errors(value, relative))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{relative}: cannot read sample map: {error}")
        for stream in segment.get("streams", []):
            if not isinstance(stream, dict) or stream.get("format") != "csv":
                continue
            stream_uri = stream.get("uri")
            if isinstance(stream_uri, str):
                try:
                    csv_path = _safe_existing_child(root, stream_uri)
                except ValueError:
                    continue
                errors.extend(_csv_time_errors(csv_path, stream_uri))
        if raw_root is not None:
            errors.extend(_source_hash_errors(segment, Path(raw_root).resolve()))
        return errors

    def validate_or_raise(
        self,
        segment_dir: str,
        *,
        raw_root: str | Path | None = None,
    ) -> None:
        errors = self.validate(segment_dir, raw_root=raw_root)
        if errors:
            raise ValueError(f"Invalid Prepared Segment: {'; '.join(errors)}")


def _referenced_files(segment: dict[str, Any]) -> set[str]:
    references = {
        stream["uri"]
        for stream in segment.get("streams", [])
        if isinstance(stream, dict) and isinstance(stream.get("uri"), str)
    }
    calibration = segment.get("calibration_uri")
    if isinstance(calibration, str):
        references.add(calibration)
    references.update(_sample_map_files(segment))
    return references


def _sample_map_files(segment: dict[str, Any]) -> set[str]:
    result = {
        item
        for item in segment.get("alignment_uris", [])
        if isinstance(item, str)
    }
    for stream in segment.get("streams", []):
        if not isinstance(stream, dict):
            continue
        origin = stream.get("origin")
        if isinstance(origin, dict) and isinstance(origin.get("sample_map_uri"), str):
            result.add(origin["sample_map_uri"])
    return result


def _safe_existing_child(root: Path, relative: str) -> Path:
    logical = PurePosixPath(relative)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"unsafe Prepared reference: {relative!r}")
    target = (root / Path(*logical.parts)).resolve(strict=False)
    if not target.is_relative_to(root):
        raise ValueError(f"Prepared reference escapes segment: {relative!r}")
    return target


def _sample_map_semantic_errors(value: dict[str, Any], relative: str) -> list[str]:
    errors: list[str] = []
    rows = value.get("rows")
    if not isinstance(rows, list):
        return errors
    previous_time: int | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if row.get("output_index") != index:
            errors.append(f"{relative}: output_index is not contiguous at row {index}")
        timestamp = row.get("segment_time_ns")
        if isinstance(timestamp, int):
            if previous_time is not None and timestamp < previous_time:
                errors.append(f"{relative}: segment_time_ns regressed at row {index}")
            previous_time = timestamp
    return errors


def _csv_time_errors(path: Path, relative: str) -> list[str]:
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "timestamp_ns" not in reader.fieldnames:
                return [f"{relative}: timestamp_ns column is missing"]
            previous: int | None = None
            for row_index, row in enumerate(reader):
                timestamp = int(row["timestamp_ns"])
                if timestamp < 0:
                    errors.append(f"{relative}: negative timestamp at row {row_index}")
                if previous is not None and timestamp < previous:
                    errors.append(f"{relative}: timestamp regressed at row {row_index}")
                previous = timestamp
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"{relative}: cannot validate CSV: {error}")
    return errors


def _source_hash_errors(segment: dict[str, Any], raw_root: Path) -> list[str]:
    errors: list[str] = []
    for asset in segment.get("source_assets", []):
        if not isinstance(asset, dict):
            continue
        uri = asset.get("uri")
        expected = asset.get("sha256")
        if not isinstance(uri, str) or not uri.startswith("raw://") or not isinstance(
            expected, str
        ):
            continue
        relative = uri.removeprefix("raw://")
        try:
            path = _safe_existing_child(raw_root, relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"source asset is missing: {uri}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            errors.append(f"source asset hash mismatch: {uri}")
    return errors
