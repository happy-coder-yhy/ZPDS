"""人员 A 使用的 WiLoR 轻量启动前检查。

本模块只依赖标准库和配置对象，不导入 PyTorch、CUDA、WiLoR 或 MANO
运行时代码。它用于在创建真实后端前检查模型来源、路径、大小和 SHA-256。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zpds.hands.config import WilorConfig


@dataclass(frozen=True)
class WilorAssetCheck:
    """单个 WiLoR 运行资产的完整性检查结果。"""

    name: str
    path: str
    exists: bool
    size_bytes: int | None
    expected_size_bytes: int | None
    sha256: str | None
    expected_sha256: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.exists
            and self.size_bytes == self.expected_size_bytes
            and self.sha256 == self.expected_sha256
        )


@dataclass(frozen=True)
class WilorPreflightReport:
    """可写入日志或交付记录的 WiLoR 资产预检报告。"""

    ready: bool
    model_revision: str
    manifest_path: str
    assets: tuple[WilorAssetCheck, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_revision": self.model_revision,
            "manifest_path": self.manifest_path,
            "assets": [
                {**asdict(asset), "ok": asset.ok} for asset in self.assets
            ],
            "errors": list(self.errors),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(document: dict[str, Any], name: str) -> dict[str, Any]:
    files = document.get("files")
    if not isinstance(files, list):
        raise TypeError("资产清单缺少 files 列表")
    matches = [
        item
        for item in files
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"资产清单中 {name!r} 应恰好出现一次")
    return matches[0]


def _nested_entry(
    document: dict[str, Any],
    *keys: str,
) -> dict[str, Any]:
    value: Any = document
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"资产清单缺少字段: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, dict):
        raise TypeError(f"资产清单字段必须是对象: {'.'.join(keys)}")
    return value


def _check_asset(
    name: str,
    configured_path: str,
    metadata: dict[str, Any],
) -> WilorAssetCheck:
    expected_size = metadata.get("size_bytes")
    expected_sha256 = str(metadata.get("sha256", "")).lower()
    if not configured_path:
        return WilorAssetCheck(
            name=name,
            path="",
            exists=False,
            size_bytes=None,
            expected_size_bytes=expected_size,
            sha256=None,
            expected_sha256=expected_sha256 or None,
            error=f"hands.wilor.{name}_path 未配置",
        )
    if not isinstance(expected_size, int) or expected_size < 0:
        return WilorAssetCheck(
            name=name,
            path=configured_path,
            exists=Path(configured_path).is_file(),
            size_bytes=None,
            expected_size_bytes=None,
            sha256=None,
            expected_sha256=expected_sha256 or None,
            error=f"{name} 的清单 size_bytes 非法",
        )
    if len(expected_sha256) != 64:
        return WilorAssetCheck(
            name=name,
            path=configured_path,
            exists=Path(configured_path).is_file(),
            size_bytes=None,
            expected_size_bytes=expected_size,
            sha256=None,
            expected_sha256=None,
            error=f"{name} 的清单 sha256 非法",
        )

    path = Path(configured_path)
    if not path.is_file():
        return WilorAssetCheck(
            name=name,
            path=str(path),
            exists=False,
            size_bytes=None,
            expected_size_bytes=expected_size,
            sha256=None,
            expected_sha256=expected_sha256,
            error=f"{name} 文件不存在",
        )
    size = path.stat().st_size
    if size != expected_size:
        return WilorAssetCheck(
            name=name,
            path=str(path),
            exists=True,
            size_bytes=size,
            expected_size_bytes=expected_size,
            sha256=None,
            expected_sha256=expected_sha256,
            error=f"{name} 文件大小不匹配",
        )
    actual_sha256 = _sha256(path)
    return WilorAssetCheck(
        name=name,
        path=str(path),
        exists=True,
        size_bytes=size,
        expected_size_bytes=expected_size,
        sha256=actual_sha256,
        expected_sha256=expected_sha256,
        error=(
            None
            if actual_sha256 == expected_sha256
            else f"{name} SHA-256 不匹配"
        ),
    )


def check_wilor_assets(config: WilorConfig) -> WilorPreflightReport:
    """校验真实 WiLoR 后端启动所需的全部本地资产。"""

    configured_model_revision = (
        config.model_revision or config.upstream_commit
    )
    manifest_path = Path(config.asset_manifest_path) if config.asset_manifest_path else None
    if manifest_path is None or not manifest_path.is_file():
        message = "hands.wilor.asset_manifest_path 不存在或未配置"
        return WilorPreflightReport(
            ready=False,
            model_revision=configured_model_revision,
            manifest_path=str(manifest_path or ""),
            assets=(),
            errors=(message,),
        )

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TypeError("资产清单顶层必须是对象")
        repository = _nested_entry(document, "model_repository")
        manifest_revision = str(repository.get("revision", ""))
        expectations = (
            (
                "detector",
                config.detector_path,
                _file_entry(document, "detector.pt"),
            ),
            (
                "checkpoint",
                config.checkpoint_path,
                _file_entry(document, "wilor_final.ckpt"),
            ),
            (
                "model_config",
                config.model_config_path,
                _file_entry(document, "model_config.yaml"),
            ),
            (
                "mano_model",
                config.mano_model_path,
                _nested_entry(document, "mano", "right_hand_model"),
            ),
            (
                "mano_mean_params",
                config.mano_mean_params_path,
                _nested_entry(document, "mano", "mean_parameters"),
            ),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return WilorPreflightReport(
            ready=False,
            model_revision=configured_model_revision,
            manifest_path=str(manifest_path),
            assets=(),
            errors=(f"无法解析 WiLoR 资产清单: {error}",),
        )

    errors: list[str] = []
    if not configured_model_revision:
        errors.append("hands.wilor.upstream_commit 未配置")
    elif manifest_revision != configured_model_revision:
        errors.append("配置的 upstream_commit 与资产清单 revision 不一致")

    manifest_checkpoint_sha = str(expectations[1][2].get("sha256", "")).lower()
    if not config.checkpoint_sha256:
        errors.append("hands.wilor.checkpoint_sha256 未配置")
    elif manifest_checkpoint_sha != config.checkpoint_sha256.lower():
        errors.append("配置的 checkpoint_sha256 与资产清单不一致")

    assets = tuple(
        _check_asset(name, configured_path, metadata)
        for name, configured_path, metadata in expectations
    )
    errors.extend(asset.error for asset in assets if asset.error is not None)
    return WilorPreflightReport(
        ready=not errors and all(asset.ok for asset in assets),
        model_revision=manifest_revision,
        manifest_path=str(manifest_path),
        assets=assets,
        errors=tuple(errors),
    )


__all__ = [
    "WilorAssetCheck",
    "WilorPreflightReport",
    "check_wilor_assets",
]
