"""Hands V1 产物登记到 Experience 层。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from zpds.hands.config import HandsOutputPaths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_uri(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_hands_experience_manifest(
    *,
    experience_dir: str | Path,
    experience_version: str,
    segment_id: str,
    video_stream_id: str,
    outputs: HandsOutputPaths,
    prep_revision: str,
    config_sha256: str,
    checkpoint_sha256: str,
    validation_status: str | None,
) -> str:
    """创建或更新当前 Experience 的 Hands V1 资产登记。"""
    root = Path(experience_dir).expanduser().resolve()
    manifest_path = outputs.experience_manifest
    if manifest_path is None:
        raise ValueError("Experience 输出路径缺少 experience_manifest.json")
    if not experience_version.strip():
        raise ValueError("experience_version 不能为空")
    if not outputs.parquet.is_file():
        raise FileNotFoundError(f"Hands Parquet 不存在: {outputs.parquet}")

    frame = pd.read_parquet(outputs.parquet)
    files: dict[str, dict[str, Any]] = {}
    for role, path in (
        ("hands_2d", outputs.parquet),
        ("hands_validation", outputs.validation_report),
        ("hands_preview", outputs.preview),
        ("hands_run", outputs.run_manifest),
    ):
        if path.is_file():
            files[role] = {
                "uri": _relative_uri(path, root),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }

    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError("现有 experience_manifest.json 顶层必须是对象")
        document: dict[str, Any] = loaded
        existing_version = document.get("experience_version")
        if existing_version not in {None, experience_version}:
            raise ValueError(
                "现有 Experience 版本不一致: "
                f"{existing_version!r} != {experience_version!r}"
            )
    else:
        document = {}

    document.update(
        {
            "schema_version": document.get("schema_version", 1),
            "experience_version": experience_version,
            "prep_revision": prep_revision,
            "segment_id": segment_id,
            "video_stream_id": video_stream_id,
        }
    )
    annotations = document.setdefault("annotations", {})
    if not isinstance(annotations, dict):
        raise TypeError("experience_manifest.json 的 annotations 必须是对象")
    annotations["hands_v1"] = {
        "rows": len(frame),
        "annotated_frames": int(
            frame["output_frame_index"].nunique()
        ),
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_status": validation_status,
        "files": files,
    }
    _write_json_atomic(manifest_path, document)
    return str(manifest_path)


__all__ = ["write_hands_experience_manifest"]
