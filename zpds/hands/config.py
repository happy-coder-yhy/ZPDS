"""Hands Pipeline 的统一配置和输出路径约定。"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from zpds.hands.mediapipe_adapter import HandEstimatorConfig
from zpds.hands.schemas import ModelInfo
from zpds.hands.wilor_schema import WiLoRConfig

VALID_BACKENDS = frozenset(
    {"auto", "tasks_hand_landmarker", "solutions_hands", "wilor"}
)


def _config_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HandsPipelineConfig:
    """经过校验、可追溯的 Hands 运行配置。"""

    path: Path
    document: dict[str, Any]
    estimator: HandEstimatorConfig
    config_sha256: str
    checkpoint_sha256: str
    wilor: WiLoRConfig | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend_override: str | None = None,
    ) -> HandsPipelineConfig:
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
        if backend_override is not None:
            hands_document["backend"] = backend_override

        estimator = HandEstimatorConfig.from_yaml(config_path)
        if backend_override is not None:
            estimator.backend = backend_override
        cls._validate_estimator(estimator)

        model_path = Path(estimator.tasks.model_path).expanduser()
        if not model_path.is_absolute():
            model_path = config_path.parent / model_path
        estimator.tasks.model_path = str(model_path.resolve())
        model_info = ModelInfo.from_file(estimator.tasks.model_path)
        wilor = cls._load_wilor_config(config_path, hands_document)
        checkpoint_sha256 = (
            _sha256_file(Path(wilor.checkpoint_path)) if wilor is not None else model_info.sha256
        )

        return cls(
            path=config_path,
            document=document,
            estimator=estimator,
            config_sha256=_config_sha256(document),
            checkpoint_sha256=checkpoint_sha256,
            wilor=wilor,
        )

    @staticmethod
    def _load_wilor_config(
        config_path: Path,
        hands_document: dict[str, Any],
    ) -> WiLoRConfig | None:
        if hands_document.get("backend") != "wilor":
            return None
        raw = hands_document.get("wilor")
        if not isinstance(raw, dict):
            raise TypeError("hands.backend=wilor 时 hands.wilor 必须是对象")

        def resolve_path(name: str) -> str:
            value = raw.get(name, "")
            if not value:
                return ""
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            return str(path.resolve())

        checkpoint_path = resolve_path("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("hands.wilor.checkpoint_path 不能为空")
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"WiLoR checkpoint 不存在: {checkpoint_path}")
        return WiLoRConfig(
            checkpoint_path=checkpoint_path,
            expected_sha256=str(raw.get("checkpoint_sha256", raw.get("expected_sha256", ""))),
            wilor_source_path=resolve_path("wilor_source_path"),
            detector_path=resolve_path("detector_path"),
            model_config_path=resolve_path("model_config_path"),
            device=str(raw.get("device", "cpu")),
            precision=str(raw.get("precision", "float32")),
            model_version=str(raw.get("model_version", "wilor_cvpr2025")),
            upstream_repository=str(raw.get("upstream_repository", "")),
            upstream_git_commit=str(raw.get("upstream_git_commit", "")),
            upstream_license_checked=bool(raw.get("upstream_license_checked", False)),
        )

    @staticmethod
    def _validate_estimator(config: HandEstimatorConfig) -> None:
        if config.backend not in VALID_BACKENDS:
            raise ValueError(
                f"hands.backend 必须是 {sorted(VALID_BACKENDS)}，实际为 {config.backend!r}"
            )
        if config.fallback_backend != "solutions_hands":
            raise ValueError("hands.fallback_backend 当前仅支持 solutions_hands")
        if config.num_hands <= 0:
            raise ValueError("hands.num_hands 必须大于 0")
        for field_name in (
            "min_hand_detection_confidence",
            "min_hand_presence_confidence",
            "min_tracking_confidence",
        ):
            value = float(getattr(config, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"hands.{field_name} 必须在 [0, 1] 范围内")
        if config.bbox_padding_ratio < 0:
            raise ValueError("hands.bbox_padding_ratio 不能为负数")
        if config.tasks.delegate not in {"cpu", "gpu"}:
            raise ValueError("hands.tasks.delegate 必须是 cpu 或 gpu")
        if config.solutions.model_complexity not in {0, 1}:
            raise ValueError("hands.solutions.model_complexity 必须是 0 或 1")


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
        )


__all__ = [
    "VALID_BACKENDS",
    "HandsOutputPaths",
    "HandsPipelineConfig",
]
