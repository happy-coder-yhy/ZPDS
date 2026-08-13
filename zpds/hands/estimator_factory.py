"""Hands 模型工厂和运行时元数据（单后端：恒 WiLoR）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from zpds.hands.config import HandsPipelineConfig
from zpds.hands.contracts import HandEstimator


class EstimatorUnavailableError(RuntimeError):
    """配置选择的模型尚未在当前运行环境中可用。"""


@dataclass
class EstimatorRuntime:
    """模型实例及写入 Manifest/Parquet 所需的来源信息。"""

    estimator: HandEstimator
    model_name: str
    model_version: str
    checkpoint_sha256: str
    active_backend: str
    upstream_git_commit: str = ""
    run_meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.estimator, HandEstimator):
            raise TypeError("estimator 必须实现 estimate() 和 close()")
        for field_name in ("model_name", "active_backend"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"EstimatorRuntime.{field_name} 不能为空")
        if not isinstance(self.run_meta, dict):
            raise TypeError("EstimatorRuntime.run_meta 必须是字典")


def validate_estimator_runtime(
    runtime: EstimatorRuntime,
    config: HandsPipelineConfig,
) -> None:
    """在 Pipeline 接管前验证运行时元数据（单后端恒为 WiLoR）。"""

    if runtime.model_name != "wilor":
        raise ValueError(
            f"EstimatorRuntime.model_name 必须为 'wilor'，实际为 {runtime.model_name!r}"
        )

    expected_sha256 = config.wilor.checkpoint_sha256.lower()
    if not expected_sha256:
        raise ValueError("hands.wilor.checkpoint_sha256 未配置")
    if runtime.checkpoint_sha256.lower() != expected_sha256:
        raise ValueError("WiLoR runtime checkpoint_sha256 与配置不一致")

    expected_commit = config.wilor.upstream_commit
    if not expected_commit:
        raise ValueError("hands.wilor.upstream_commit 未配置")
    if runtime.upstream_git_commit != expected_commit:
        raise ValueError("WiLoR runtime upstream_git_commit 与配置不一致")
    if runtime.active_backend != "wilor":
        raise ValueError("WiLoR runtime active_backend 必须为 'wilor'")


def create_hand_estimator(config: HandsPipelineConfig) -> EstimatorRuntime:
    """创建 Hands 检测器（单后端：恒 WiLoR，无跨模型回退）。"""
    return _create_wilor_estimator(config)


def _read_source_commit(source_path: Path) -> str:
    git_dir = source_path / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return ""
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_name = head.removeprefix("ref: ").strip()
    ref_path = git_dir / Path(ref_name)
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")):
                continue
            commit, _, name = line.partition(" ")
            if name == ref_name:
                return commit
    return ""


def _validate_wilor_joint_mapping_contract(
    *,
    required_version: str | None,
    mapping_version: str,
    mapping: tuple[int, ...],
) -> None:
    """Validate Person B's versioned module-level joint mapping."""
    if required_version not in {None, mapping_version}:
        raise EstimatorUnavailableError(
            "Unsupported WiLoR joint mapping version: "
            f"required={required_version!r}, provided={mapping_version!r}"
        )
    if len(mapping) != 21:
        raise EstimatorUnavailableError(
            "WiLoR joint mapping must contain exactly 21 indices"
        )
    if any(
        not isinstance(index, int) or isinstance(index, bool)
        for index in mapping
    ):
        raise EstimatorUnavailableError(
            "WiLoR joint mapping indices must be integers"
        )
    if len(set(mapping)) != 21:
        raise EstimatorUnavailableError(
            "WiLoR joint mapping must not contain duplicate indices"
        )
    if any(index < 0 or index >= 21 for index in mapping):
        raise EstimatorUnavailableError(
            "WiLoR joint mapping indices must be in the range [0, 20]"
        )


def _create_wilor_estimator(config: HandsPipelineConfig) -> EstimatorRuntime:
    """Create the real WiLoR stack through its lazy-loading modules."""
    wilor = config.wilor
    source_path = Path(wilor.source_path)
    if not wilor.source_path or not (source_path / "wilor").is_dir():
        raise EstimatorUnavailableError(
            "WiLoR source package is unavailable: "
            f"{source_path if wilor.source_path else '<not configured>'}; "
            "单后端模式下 WiLoR 不可用即 Hands 检测不可用"
        )
    if not wilor.upstream_commit:
        raise EstimatorUnavailableError(
            "hands.wilor.upstream_commit is not configured"
        )
    source_commit = _read_source_commit(source_path)
    if source_commit != wilor.upstream_commit:
        raise EstimatorUnavailableError(
            "WiLoR source commit does not match config: "
            f"expected={wilor.upstream_commit}, actual={source_commit or 'unknown'}"
        )
    if not wilor.upstream_license_checked:
        raise EstimatorUnavailableError(
            "WiLoR upstream license has not been acknowledged in config"
        )

    from zpds.hands.backends.wilor import WiLoRBackend
    from zpds.hands.wilor_adapter import WiLoRAdapter
    from zpds.hands.wilor_estimator import (
        WiLoREstimatorConfig,
        WiLoRHandEstimator,
    )
    from zpds.hands.wilor_joint_mapping import (
        MAPPING_VERSION,
        WILOR_TO_HANDS_V1_V1,
    )
    from zpds.hands.wilor_schema import (
        WiLoRConfig as BackendWiLoRConfig,
    )

    _validate_wilor_joint_mapping_contract(
        required_version=wilor.require_joint_mapping_version,
        mapping_version=MAPPING_VERSION,
        mapping=WILOR_TO_HANDS_V1_V1,
    )
    precision = {
        "fp16": "float16",
        "fp32": "float32",
    }.get(wilor.precision)
    if precision is None:
        raise EstimatorUnavailableError(
            f"WiLoR backend does not support precision={wilor.precision!r}"
        )

    backend_config = BackendWiLoRConfig(
        checkpoint_path=wilor.checkpoint_path,
        expected_sha256=wilor.checkpoint_sha256.lower(),
        wilor_source_path=str(source_path),
        detector_path=wilor.detector_path,
        model_config_path=wilor.model_config_path,
        device=wilor.device,
        precision=precision,
        model_version=wilor.model_version,
        upstream_repository=wilor.upstream_repository,
        upstream_git_commit=wilor.upstream_commit,
        upstream_license_checked=wilor.upstream_license_checked,
    )
    backend = WiLoRBackend(backend_config)
    adapter = WiLoRAdapter(backend)
    estimator = WiLoRHandEstimator(
        adapter=adapter,
        model_info=backend.model_info,
        config=WiLoREstimatorConfig(
            model_name="wilor",
            model_version=wilor.model_version,
            # 抽帧：ego_bbox_every_frame=False 时按 bbox_fps 时间窗推理，
            # 中间帧复用上一推理帧结果（WiLoRHandEstimator 内部实现）
            ego_bbox_every_frame=wilor.ego_bbox_every_frame,
            bbox_fps=wilor.bbox_fps,
        ),
    )
    return EstimatorRuntime(
        estimator=estimator,
        model_name="wilor",
        model_version=backend.model_info.model_version,
        checkpoint_sha256=backend.model_info.checkpoint_sha256,
        active_backend="wilor",
        upstream_git_commit=backend.model_info.upstream_git_commit,
        run_meta={
            "backend_requested": "wilor",
            "backend_active": "wilor",
            "backend_fallback_used": False,
            "backend_fallback_reason": "",
            "device": backend.model_info.device,
            "precision": backend.model_info.precision,
            "source_path": str(source_path),
            "upstream_repository": wilor.upstream_repository,
            "upstream_git_commit": wilor.upstream_commit,
            "model_revision": wilor.model_revision,
            "joint_mapping_version": MAPPING_VERSION,
        },
    )


__all__ = [
    "EstimatorRuntime",
    "EstimatorUnavailableError",
    "create_hand_estimator",
    "validate_estimator_runtime",
]
