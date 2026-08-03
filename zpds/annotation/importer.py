"""Import existing Prepared annotation streams into a versioned Experience."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pyarrow.parquet as pq

ANNOTATION_MANIFEST_KEY = "source_annotations_v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def import_segment_annotations(
    segment_dir: str | Path,
    experience_dir: str | Path,
    *,
    experience_version: str | None = None,
) -> str | None:
    """Copy declared existing annotation streams from one Prepared Segment.

    Only ``role: annotation`` Parquet streams declared in ``segment.json`` are
    accepted.  This keeps untrusted source formats, particularly EPIC pickles,
    outside the Experience layer: they must first be parsed and normalized by the
    adapter/Prepared stage.

    Returns the updated manifest path, or ``None`` when the segment has no
    declared annotation streams.
    """
    source_root = Path(segment_dir).expanduser().resolve()
    root = Path(experience_dir).expanduser().resolve()
    segment = _load_segment(source_root)
    entries = _collect_annotation_entries(source_root, segment)
    if not entries:
        return None

    segment_id = _require_identifier(segment.get("segment_id"), "segment_id")
    source_session = _source_session_id(segment)
    prep_revision = str(segment.get("record_revision") or "r0001")
    version = experience_version or root.name
    if not version.strip():
        raise ValueError("experience_version 不能为空")

    manifest_path = root / "experience_manifest.json"
    document = _load_manifest(manifest_path, version, prep_revision)
    annotation_group = _annotation_group(document)

    # Validate every input before copying, so a bad stream never leaves a partly
    # populated Experience directory or a partial manifest.
    planned = [
        _plan_asset(
            entry,
            root=root,
            segment_id=segment_id,
            source_session_id=source_session,
            prep_revision=prep_revision,
        )
        for entry in entries
    ]
    _verify_existing_assets(annotation_group, planned)

    for asset in planned:
        destination = asset.pop("_destination")
        _copy_immutable(asset.pop("_source"), destination, asset["sha256"])

    existing = {
        item["annotation_id"]: item
        for item in annotation_group["assets"]
        if isinstance(item, dict) and "annotation_id" in item
    }
    for asset in planned:
        existing[asset["annotation_id"]] = asset
    annotation_group["assets"] = [existing[key] for key in sorted(existing)]
    annotation_group["asset_count"] = len(annotation_group["assets"])

    _write_json_atomic(manifest_path, document)
    return str(manifest_path.resolve())


def _load_segment(segment_root: Path) -> dict[str, Any]:
    path = segment_root / "segment.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared Segment 缺少 segment.json: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("segment.json 顶层必须是对象")
    streams = document.get("streams")
    if not isinstance(streams, list):
        raise TypeError("segment.json 的 streams 必须是数组")
    return document


def _collect_annotation_entries(
    segment_root: Path,
    segment: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for stream in segment["streams"]:
        if not isinstance(stream, dict) or stream.get("role") != "annotation":
            continue
        stream_id = _require_identifier(stream.get("stream_id"), "annotation stream_id")
        if stream_id in seen:
            raise ValueError(f"segment.json 包含重复 annotation stream_id: {stream_id}")
        seen.add(stream_id)
        if stream.get("format") != "parquet":
            raise ValueError(
                f"暂不支持导入非 Parquet 标注流: {stream_id} ({stream.get('format')!r})"
            )
        uri = stream.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError(f"annotation stream {stream_id} 缺少 uri")
        source = _resolve_segment_file(segment_root, uri)
        if source.suffix.lower() != ".parquet":
            raise ValueError(f"annotation stream {stream_id} 必须指向 .parquet 文件")
        if not source.is_file():
            raise FileNotFoundError(f"annotation stream 文件不存在: {source}")
        try:
            row_count = pq.ParquetFile(source).metadata.num_rows
        except Exception as error:  # pyarrow has several version-specific exception types.
            raise ValueError(f"annotation stream 不是可读的 Parquet: {source}: {error}") from error
        entries.append({"stream": stream, "source": source, "rows": int(row_count)})
    return sorted(entries, key=lambda item: item["stream"]["stream_id"])


def _plan_asset(
    entry: dict[str, Any],
    *,
    root: Path,
    segment_id: str,
    source_session_id: str,
    prep_revision: str,
) -> dict[str, Any]:
    stream = entry["stream"]
    source = entry["source"]
    stream_id = stream["stream_id"]
    segment_key = f"{source_session_id}__{segment_id}"
    destination = root / "assets" / "annotations" / segment_key / f"{stream_id}.parquet"
    return {
        "annotation_id": f"{source_session_id}:{segment_id}:{stream_id}",
        "segment_id": segment_id,
        "source_session_id": source_session_id,
        "prep_revision": prep_revision,
        "stream_id": stream_id,
        "modality": str(stream.get("modality") or "unknown"),
        "format": "parquet",
        "ground_truth_status": str(stream.get("ground_truth_status") or "unknown"),
        "rows": entry["rows"],
        "uri": _relative_uri(destination, root),
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
        "origin": dict(stream.get("origin") or {}),
        "source_stream_uri": str(stream["uri"]),
        "_source": source,
        "_destination": destination,
    }


def _load_manifest(path: Path, version: str, prep_revision: str) -> dict[str, Any]:
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TypeError("experience_manifest.json 顶层必须是对象")
        existing_version = document.get("experience_version")
        if existing_version not in {None, version}:
            raise ValueError(
                f"Experience 版本不一致: {existing_version!r} != {version!r}"
            )
        existing_revision = document.get("prep_revision")
        if existing_revision not in {None, prep_revision}:
            raise ValueError(
                f"Prepared revision 不一致: {existing_revision!r} != {prep_revision!r}"
            )
    else:
        document = {}
    document.setdefault("schema_version", 1)
    document["experience_version"] = version
    document.setdefault("prep_revision", prep_revision)
    return document


def _annotation_group(document: dict[str, Any]) -> dict[str, Any]:
    annotations = document.setdefault("annotations", {})
    if not isinstance(annotations, dict):
        raise TypeError("experience_manifest.json 的 annotations 必须是对象")
    group = annotations.setdefault(
        ANNOTATION_MANIFEST_KEY,
        {"schema_version": 1, "assets": []},
    )
    if not isinstance(group, dict) or not isinstance(group.get("assets"), list):
        raise TypeError(f"annotations.{ANNOTATION_MANIFEST_KEY} 必须包含 assets 数组")
    group.setdefault("schema_version", 1)
    return group


def _verify_existing_assets(group: dict[str, Any], planned: list[dict[str, Any]]) -> None:
    existing = {
        item.get("annotation_id"): item
        for item in group["assets"]
        if isinstance(item, dict)
    }
    for asset in planned:
        current = existing.get(asset["annotation_id"])
        if current is not None and current.get("sha256") != asset["sha256"]:
            raise ValueError(
                "Experience 已包含同一 annotation_id 但内容哈希不同: "
                f"{asset['annotation_id']}"
            )


def _copy_immutable(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != expected_sha256:
            raise ValueError(f"Experience 目标资产已存在且内容不同: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256(destination) != expected_sha256:
        raise RuntimeError(f"复制后的 annotation 资产哈希不匹配: {destination}")


def _resolve_segment_file(segment_root: Path, uri: str) -> Path:
    # ``Path`` only understands the current host's path syntax. A manifest
    # produced on another OS must still reject absolute paths and traversal,
    # so validate the URI using both path flavours before resolving it.
    posix_candidate = PurePosixPath(uri)
    windows_candidate = PureWindowsPath(uri)
    if (
        posix_candidate.is_absolute()
        or windows_candidate.is_absolute()
        or bool(windows_candidate.drive)
        or ".." in posix_candidate.parts
        or ".." in windows_candidate.parts
    ):
        raise ValueError(f"annotation uri 必须是 Segment 内相对路径: {uri!r}")
    # Treat either slash style as a portable manifest separator.
    candidate = Path(*PurePosixPath(uri.replace("\\", "/")).parts)
    resolved = (segment_root / candidate).resolve()
    if not resolved.is_relative_to(segment_root):
        raise ValueError(f"annotation uri 超出 Segment 目录: {uri!r}")
    return resolved


def _source_session_id(segment: dict[str, Any]) -> str:
    session = segment.get("source_session")
    session_id = session.get("session_id") if isinstance(session, dict) else None
    return _require_identifier(session_id or "unknown_session", "source_session.session_id")


def _require_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field_name} 必须是安全的非空标识符")
    return value


def _relative_uri(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["ANNOTATION_MANIFEST_KEY", "import_segment_annotations"]
