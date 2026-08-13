"""Hands Pipeline 的统一配置和输出路径约定。"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WilorConfig:
    """WiLoR 运行配置；阶段一只解析和校验，不导入模型依赖。"""

    enabled: bool = False
    ego_bbox_every_frame: bool = True
    bbox_fps: float = 30.0
    write_frame_status: bool = True
    upstream_repository: str = "https://github.com/rolpotamias/WiLoR.git"
    upstream_commit: str = ""
    model_revision: str = ""
    source_path: str = ""
    model_version: str = "wilor_cvpr2025"
    upstream_license_checked: bool = False
    detector_path: str = ""
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    model_config_path: str = ""
    mano_model_path: str = ""
    mano_mean_params_path: str = ""
    asset_manifest_path: str = ""
    device: str = "cuda:0"
    precision: str = "fp16"
    bbox_padding_ratio: float = 0.15
    candidate_context_s: float = 0.75
    require_joint_mapping_version: str | None = None
    require_calibration_for_metric_3d: bool = True

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any] | None,
        *,
        config_path: Path,
    ) -> WilorConfig:
        document = value or {}

        def resolve_path(field_name: str) -> str:
            raw_value = str(document.get(field_name, ""))
            if not raw_value:
                return ""
            path = Path(raw_value).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            return str(path.resolve())

        config = cls(
            enabled=bool(document.get("enabled", False)),
            ego_bbox_every_frame=bool(
                document.get("ego_bbox_every_frame", True)
            ),
            bbox_fps=float(document.get("bbox_fps", 30.0)),
            write_frame_status=bool(document.get("write_frame_status", True)),
            upstream_repository=str(
                document.get(
                    "upstream_repository",
                    "https://github.com/rolpotamias/WiLoR.git",
                )
            ),
            upstream_commit=str(document.get("upstream_commit", "")),
            model_revision=str(document.get("model_revision", "")),
            source_path=resolve_path("source_path"),
            model_version=str(
                document.get("model_version", "wilor_cvpr2025")
            ),
            upstream_license_checked=bool(
                document.get("upstream_license_checked", False)
            ),
            detector_path=resolve_path("detector_path"),
            checkpoint_path=resolve_path("checkpoint_path"),
            checkpoint_sha256=str(document.get("checkpoint_sha256", "")),
            model_config_path=resolve_path("model_config_path"),
            mano_model_path=resolve_path("mano_model_path"),
            mano_mean_params_path=resolve_path("mano_mean_params_path"),
            asset_manifest_path=resolve_path("asset_manifest_path"),
            device=str(document.get("device", "cuda:0")),
            precision=str(document.get("precision", "fp16")),
            bbox_padding_ratio=float(
                document.get("bbox_padding_ratio", 0.15)
            ),
            candidate_context_s=float(
                document.get("candidate_context_s", 0.75)
            ),
            require_joint_mapping_version=document.get(
                "require_joint_mapping_version"
            ),
            require_calibration_for_metric_3d=bool(
                document.get("require_calibration_for_metric_3d", True)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.bbox_fps <= 0:
            raise ValueError("hands.wilor.bbox_fps 必须大于 0")
        if self.bbox_padding_ratio < 0:
            raise ValueError("hands.wilor.bbox_padding_ratio 不能为负数")
        if self.candidate_context_s < 0:
            raise ValueError("hands.wilor.candidate_context_s 不能为负数")
        if self.precision not in {"fp16", "fp32", "bf16"}:
            raise ValueError(
                "hands.wilor.precision 必须是 fp16、fp32 或 bf16"
            )
        if not self.device.strip():
            raise ValueError("hands.wilor.device 不能为空")
        if self.enabled and not self.model_version.strip():
            raise ValueError("hands.wilor.model_version must not be empty")


def _config_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HandsPipelineConfig:
    """经过校验、可追溯的 Hands 运行配置（单后端：恒 WiLoR）。"""

    path: Path
    document: dict[str, Any]
    wilor: WilorConfig
    config_sha256: str
    checkpoint_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> HandsPipelineConfig:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Hands 配置文件不存在: {config_path}")

        with config_path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if not isinstance(loaded, dict):
            raise TypeError(f"配置文件顶层必须是对象: {config_path}")

        document: dict[str, Any] = copy.deepcopy(loaded)
        hands_document = document.setdefault("hands", {})
        if not isinstance(hands_document, dict):
            raise TypeError("配置中的 hands 必须是对象")
        wilor_document = hands_document.get("wilor", {})
        if not isinstance(wilor_document, dict):
            raise TypeError("配置中的 hands.wilor 必须是对象")
        wilor = WilorConfig.from_mapping(
            wilor_document,
            config_path=config_path,
        )
        # 单后端模式：Hands 检测恒走 WiLoR，不再有 MediaPipe 回退。
        if not wilor.enabled:
            raise ValueError("单后端模式要求 hands.wilor.enabled 必须为 true")
        # ego_bbox_every_frame=False 时按 bbox_fps 抽帧（estimator 时间窗），
        # 由 WiLoREstimatorConfig.ego_bbox_every_frame / bbox_fps 承接；
        # bbox_fps<=0 已在上方 validate() 拦截。
        if not wilor.write_frame_status:
            raise ValueError("WiLoR 配置必须启用 write_frame_status")

        checkpoint_path = Path(wilor.checkpoint_path)
        if not wilor.checkpoint_path:
            raise ValueError("hands.wilor.checkpoint_path 不能为空")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"WiLoR checkpoint 不存在: {checkpoint_path}"
            )
        checkpoint_sha256 = _sha256_file(checkpoint_path)

        return cls(
            path=config_path,
            document=document,
            wilor=wilor,
            config_sha256=_config_sha256(document),
            checkpoint_sha256=checkpoint_sha256,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HandsOutputPaths:
    """一次 Hands 运行的标准产物路径。"""

    parquet: Path
    validation_report: Path
    preview: Path
    run_manifest: Path
    experience_manifest: Path | None = None
    frame_status: Path | None = None
    bbox: Path | None = None

    @classmethod
    def standard(
        cls,
        output_root: str | Path,
        segment_id: str,
        video_stream_id: str,
    ) -> HandsOutputPaths:
        directory = (
            Path(output_root).expanduser().resolve()
            / segment_id
            / video_stream_id
        )
        return cls(
            parquet=directory / "hands_2d.parquet",
            validation_report=directory / "hands_validation.json",
            preview=directory / "hands_preview.mp4",
            run_manifest=directory / "hands_run.json",
            frame_status=directory / "wilor_frame_status.parquet",
            bbox=directory / "wilor_hands_bbox.parquet",
        )

    @classmethod
    def experience(
        cls,
        experience_dir: str | Path,
    ) -> HandsOutputPaths:
        root = Path(experience_dir).expanduser().resolve()
        return cls(
            parquet=root / "assets" / "poses" / "hands_2d.parquet",
            validation_report=root / "reports" / "hands_validation.json",
            preview=root / "previews" / "hands_preview.mp4",
            run_manifest=root / "reports" / "hands_run.json",
            experience_manifest=root / "experience_manifest.json",
            frame_status=(
                root / "assets" / "poses" / "wilor_frame_status.parquet"
            ),
            bbox=root / "assets" / "poses" / "wilor_hands_bbox.parquet",
        )


__all__ = [
    "HandsOutputPaths",
    "HandsPipelineConfig",
    "WilorConfig",
]
