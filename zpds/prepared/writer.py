"""Prepared Segment 的原子写入。"""

import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from zpds.utils.schema_validator import validate_with_schema

from .validator import PreparedValidator


class PreparedSegmentWriter:
    """先写临时目录，通过完整校验后再原子落盘。"""

    def write(
        self,
        output_dir: str,
        segment_data: dict[str, Any],
        *,
        files: dict[str, bytes | str | dict[str, Any] | list[Any]] | None = None,
    ) -> str:
        errors = validate_with_schema(segment_data, "segment")
        if errors:
            raise ValueError(f"Invalid Prepared Segment: {'; '.join(errors)}")
        segment_id = str(segment_data["segment_id"])
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = root / segment_id
        if target.exists():
            raise FileExistsError(f"Prepared Segment already exists: {target}")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{segment_id}.", suffix=".tmp", dir=root)
        )
        try:
            for relative, value in (files or {}).items():
                path = _safe_child(temporary, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(value, bytes):
                    path.write_bytes(value)
                elif isinstance(value, str):
                    path.write_text(value, encoding="utf-8")
                else:
                    path.write_text(
                        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    )
            (temporary / "segment.json").write_text(
                json.dumps(
                    segment_data,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            validation_errors = PreparedValidator().validate(str(temporary))
            if validation_errors:
                raise ValueError(
                    "Prepared Segment write validation failed: "
                    + "; ".join(validation_errors)
                )
            _fsync_tree(temporary)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return segment_id


def _safe_child(root: Path, relative: str) -> Path:
    logical = PurePosixPath(relative)
    if logical.is_absolute() or not logical.parts or any(
        part in {"", ".", ".."} for part in logical.parts
    ):
        raise ValueError(f"Unsafe Prepared relative path: {relative!r}")
    target = (root / Path(*logical.parts)).resolve(strict=False)
    if not target.is_relative_to(root):
        raise ValueError(f"Prepared path escapes segment directory: {relative!r}")
    return target


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Windows 的 CRT 不允许对只读文件描述符调用 fsync。
        with path.open("r+b") as file:
            os.fsync(file.fileno())
