"""Adapter 共用的只读文件、哈希与类型推断工具。"""

import hashlib
import importlib
import mimetypes
import re
from pathlib import Path

from zpds.core.types import SourceAsset, StreamKind

from .contracts import OptionalDependencyError


def require_file(path: str | Path) -> Path:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")
    return source


def require_directory(path: str | Path) -> Path:
    source = Path(path).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Session directory not found: {source}")
    return source


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with require_file(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def source_asset(
    path: Path,
    root: Path,
    *,
    asset_id: str | None = None,
    include_hash: bool = False,
    required: bool = True,
) -> SourceAsset:
    relative = path.relative_to(root).as_posix()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return SourceAsset(
        asset_id=asset_id or safe_identifier(relative),
        uri=f"raw://{relative}",
        relative_path=relative,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path) if include_hash else None,
        media_type=media_type,
        required=required,
    )


def safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return normalized or "asset"


def infer_stream_kind(name: str) -> StreamKind:
    lowered = name.lower()
    if "depth" in lowered:
        return StreamKind.DEPTH
    if "imu" in lowered or "gyro" in lowered or "accel" in lowered:
        return StreamKind.IMU
    if "command" in lowered or "action" in lowered:
        return StreamKind.ROBOT_COMMAND
    if "joint" in lowered or "state" in lowered:
        return StreamKind.ROBOT_STATE
    if "mag" in lowered or "encoder" in lowered or "gripper" in lowered:
        return StreamKind.MAGNETIC_ENCODER
    if "vio" in lowered or "pose" in lowered or "odom" in lowered:
        return StreamKind.VIO_POSE
    return StreamKind.COLOR


def require_optional_module(module_name: str, extra: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise OptionalDependencyError(
            f"{module_name} is required; install with "
            f'python -m pip install -e ".[{extra}]"'
        ) from error
