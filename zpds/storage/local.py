"""安全的本地文件系统 Storage Adapter。"""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .base import (
    ArtifactExistsError,
    InvalidStorageReference,
    StorageError,
    StoredFile,
)

PathWriter = Callable[[Path], None]
PathValidator = Callable[[Path], None]


class LocalStorage:
    """将 raw:// 与 artifact:// 引用限制在两个独立根目录中。"""

    def __init__(
        self,
        raw_root: str | Path | None,
        artifact_root: str | Path,
    ) -> None:
        self.raw_root = Path(raw_root).resolve() if raw_root is not None else None
        self.artifact_root = Path(artifact_root).resolve()
        if self.raw_root is not None and not self.raw_root.is_dir():
            raise StorageError(f"Raw root is not a directory: {self.raw_root}")
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        if not self.artifact_root.is_dir():
            raise StorageError(f"Artifact root is not a directory: {self.artifact_root}")

    def open_read(self, reference: str) -> BinaryIO:
        path = self._resolve(reference)
        if not path.is_file():
            raise FileNotFoundError(f"Storage file not found: {reference}")
        return path.open("rb")

    def exists(self, reference: str) -> bool:
        return self._resolve(reference).exists()

    def sha256(self, reference: str, chunk_size: int = 1024 * 1024) -> str:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        digest = hashlib.sha256()
        with self.open_read(reference) as file:
            while chunk := file.read(chunk_size):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def read_json(self, reference: str) -> dict[str, Any]:
        with self.open_read(reference) as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise StorageError(f"Expected JSON object: {reference}")
        return value

    def atomic_write_json(
        self,
        reference: str,
        value: dict[str, Any],
        *,
        validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> StoredFile:
        if validator is not None:
            validator(value)

        def write(path: Path) -> None:
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return self.atomic_write_file(reference, write)

    def atomic_write_file(
        self,
        reference: str,
        writer: PathWriter,
        *,
        validator: PathValidator | None = None,
    ) -> StoredFile:
        target = self._resolve_artifact(reference)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            writer(temporary_path)
            if not temporary_path.is_file():
                raise StorageError(f"Writer did not produce a file: {reference}")
            if validator is not None:
                validator(temporary_path)
            with temporary_path.open("rb+") as file:
                os.fsync(file.fileno())
            os.replace(temporary_path, target)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return StoredFile(
            reference=reference,
            path=target,
            size_bytes=target.stat().st_size,
            sha256=self.sha256(reference),
        )

    def atomic_write_directory(
        self,
        reference: str,
        writer: PathWriter,
        *,
        validator: PathValidator | None = None,
    ) -> Path:
        target = self._resolve_artifact(reference)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ArtifactExistsError(f"Artifact directory already exists: {reference}")
        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        )
        try:
            writer(temporary_path)
            if validator is not None:
                validator(temporary_path)
            os.replace(temporary_path, target)
        except BaseException:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise
        return target

    def raw_path(self, reference: str) -> Path:
        """返回只读 Raw 路径；调用方不得修改该路径。"""
        return self._resolve_in_namespace(reference, "raw", self._require_raw_root())

    def artifact_path(self, reference: str) -> Path:
        return self._resolve_artifact(reference)

    def _resolve_artifact(self, reference: str) -> Path:
        return self._resolve_in_namespace(reference, "artifact", self.artifact_root)

    def _resolve(self, reference: str) -> Path:
        namespace, _ = _split_reference(reference)
        if namespace == "raw":
            return self._resolve_in_namespace(
                reference,
                namespace,
                self._require_raw_root(),
            )
        if namespace == "artifact":
            return self._resolve_in_namespace(reference, namespace, self.artifact_root)
        raise InvalidStorageReference(
            f"Unsupported storage namespace {namespace!r}; expected raw or artifact"
        )

    def _require_raw_root(self) -> Path:
        if self.raw_root is None:
            raise StorageError("Raw root is not configured for this storage instance")
        return self.raw_root

    @staticmethod
    def _resolve_in_namespace(reference: str, expected: str, root: Path) -> Path:
        namespace, logical_path = _split_reference(reference)
        if namespace != expected:
            raise InvalidStorageReference(
                f"Expected {expected}:// reference, received {reference!r}"
            )
        candidate = (root / Path(*logical_path.parts)).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise InvalidStorageReference(f"Storage reference escapes root: {reference}")
        return candidate


def _split_reference(reference: str) -> tuple[str, PurePosixPath]:
    if not isinstance(reference, str) or "://" not in reference:
        raise InvalidStorageReference(
            f"Storage reference must use namespace://path: {reference!r}"
        )
    namespace, raw_path = reference.split("://", 1)
    if not namespace or not raw_path:
        raise InvalidStorageReference(f"Storage reference is incomplete: {reference!r}")
    if "\\" in raw_path or ":" in raw_path:
        raise InvalidStorageReference(
            f"Storage reference path must use relative POSIX syntax: {reference!r}"
        )
    parts = raw_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise InvalidStorageReference(f"Unsafe storage reference path: {reference!r}")
    logical_path = PurePosixPath(*parts)
    if logical_path.is_absolute():
        raise InvalidStorageReference(f"Storage reference path must be relative: {reference!r}")
    return namespace, logical_path
