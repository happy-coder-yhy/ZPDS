"""场景分割统一 YAML 配置、校验及可追溯哈希。"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _unit(value: float, field_name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} 必须在 [0, 1] 范围内")
    return number


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{field_name} 必须大于 0")
    return number


def _positive_int(value: int, field_name: str) -> int:
    number = int(value)
    if isinstance(value, bool) or number <= 0:
        raise ValueError(f"{field_name} 必须是正整数")
    return number


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} 必须是对象")
    return value


def _config_hash(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass(frozen=True)
class HistogramConfig:
    enabled: bool = True
    h_bins: int = 32
    s_bins: int = 32
    method: str = "bhattacharyya"
    threshold: float = 0.45

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> HistogramConfig:
        config = cls(
            enabled=bool(value.get("enabled", True)),
            h_bins=_positive_int(value.get("h_bins", 32), "scene.stage_a.histogram.h_bins"),
            s_bins=_positive_int(value.get("s_bins", 32), "scene.stage_a.histogram.s_bins"),
            method=str(value.get("method", "bhattacharyya")),
            threshold=_unit(value.get("threshold", 0.45), "scene.stage_a.histogram.threshold"),
        )
        if config.method not in {"bhattacharyya", "chi_square"}:
            raise ValueError("scene.stage_a.histogram.method 必须是 bhattacharyya 或 chi_square")
        return config


@dataclass(frozen=True)
class SSIMConfig:
    enabled: bool = True
    gaussian_window_size: int = 11
    gaussian_sigma: float = 1.5
    hard_cut_similarity: float = 0.45
    gradual_similarity: float = 0.82
    gradual_window_frames: int = 5
    gradual_min_frames: int = 3

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SSIMConfig:
        window = _positive_int(
            value.get("gaussian_window_size", 11),
            "scene.stage_a.ssim.gaussian_window_size",
        )
        if window % 2 == 0:
            raise ValueError("scene.stage_a.ssim.gaussian_window_size 必须是奇数")
        config = cls(
            enabled=bool(value.get("enabled", True)),
            gaussian_window_size=window,
            gaussian_sigma=_positive(value.get("gaussian_sigma", 1.5), "scene.stage_a.ssim.gaussian_sigma"),
            hard_cut_similarity=_unit(value.get("hard_cut_similarity", 0.45), "scene.stage_a.ssim.hard_cut_similarity"),
            gradual_similarity=_unit(value.get("gradual_similarity", 0.82), "scene.stage_a.ssim.gradual_similarity"),
            gradual_window_frames=_positive_int(value.get("gradual_window_frames", 5), "scene.stage_a.ssim.gradual_window_frames"),
            gradual_min_frames=_positive_int(value.get("gradual_min_frames", 3), "scene.stage_a.ssim.gradual_min_frames"),
        )
        if config.hard_cut_similarity >= config.gradual_similarity:
            raise ValueError("scene.stage_a.ssim.hard_cut_similarity 必须小于 gradual_similarity")
        return config


@dataclass(frozen=True)
class OpticalFlowConfig:
    enabled: bool = True
    analysis_max_dimension: int = 320
    pyr_scale: float = 0.5
    levels: int = 3
    window_size: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    grid_step: int = 8
    min_correspondences: int = 16
    ransac_reproj_threshold: float = 2.0
    residual_threshold_px: float = 1.8
    residual_hard_scale_px: float = 6.0
    freeze_motion_threshold_px: float = 0.08
    freeze_min_frames: int = 5

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> OpticalFlowConfig:
        config = cls(
            enabled=bool(value.get("enabled", True)),
            analysis_max_dimension=_positive_int(
                value.get("analysis_max_dimension", 320),
                "scene.stage_a.optical_flow.analysis_max_dimension",
            ),
            pyr_scale=_unit(value.get("pyr_scale", 0.5), "scene.stage_a.optical_flow.pyr_scale"),
            levels=_positive_int(value.get("levels", 3), "scene.stage_a.optical_flow.levels"),
            window_size=_positive_int(value.get("window_size", 15), "scene.stage_a.optical_flow.window_size"),
            iterations=_positive_int(value.get("iterations", 3), "scene.stage_a.optical_flow.iterations"),
            poly_n=_positive_int(value.get("poly_n", 5), "scene.stage_a.optical_flow.poly_n"),
            poly_sigma=_positive(value.get("poly_sigma", 1.2), "scene.stage_a.optical_flow.poly_sigma"),
            grid_step=_positive_int(value.get("grid_step", 8), "scene.stage_a.optical_flow.grid_step"),
            min_correspondences=_positive_int(value.get("min_correspondences", 16), "scene.stage_a.optical_flow.min_correspondences"),
            ransac_reproj_threshold=_positive(value.get("ransac_reproj_threshold", 2.0), "scene.stage_a.optical_flow.ransac_reproj_threshold"),
            residual_threshold_px=_positive(value.get("residual_threshold_px", 1.8), "scene.stage_a.optical_flow.residual_threshold_px"),
            residual_hard_scale_px=_positive(value.get("residual_hard_scale_px", 6.0), "scene.stage_a.optical_flow.residual_hard_scale_px"),
            freeze_motion_threshold_px=_positive(value.get("freeze_motion_threshold_px", 0.08), "scene.stage_a.optical_flow.freeze_motion_threshold_px"),
            freeze_min_frames=_positive_int(value.get("freeze_min_frames", 5), "scene.stage_a.optical_flow.freeze_min_frames"),
        )
        if not 0.0 < config.pyr_scale < 1.0:
            raise ValueError("scene.stage_a.optical_flow.pyr_scale 必须在 (0, 1) 范围内")
        if config.residual_hard_scale_px <= config.residual_threshold_px:
            raise ValueError("residual_hard_scale_px 必须大于 residual_threshold_px")
        if config.poly_n not in {5, 7}:
            raise ValueError("scene.stage_a.optical_flow.poly_n 必须是 5 或 7")
        return config


@dataclass(frozen=True)
class BrightnessConfig:
    enabled: bool = True
    mean_jump_threshold: float = 0.22
    black_pixel_value: int = 16
    black_ratio_threshold: float = 0.95

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> BrightnessConfig:
        black_value = int(value.get("black_pixel_value", 16))
        if not 0 <= black_value <= 255:
            raise ValueError("scene.stage_a.brightness.black_pixel_value 必须在 [0, 255]")
        return cls(
            enabled=bool(value.get("enabled", True)),
            mean_jump_threshold=_unit(value.get("mean_jump_threshold", 0.22), "scene.stage_a.brightness.mean_jump_threshold"),
            black_pixel_value=black_value,
            black_ratio_threshold=_unit(value.get("black_ratio_threshold", 0.95), "scene.stage_a.brightness.black_ratio_threshold"),
        )


@dataclass(frozen=True)
class StageAConfig:
    smoothing_window_frames: int
    merge_window_s: float
    weights: dict[str, float]
    histogram: HistogramConfig
    ssim: SSIMConfig
    optical_flow: OpticalFlowConfig
    brightness: BrightnessConfig

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> StageAConfig:
        raw_weights = _mapping(value.get("weights", {}), "scene.stage_a.weights")
        defaults = {"histogram": 0.25, "ssim": 0.30, "optical_flow": 0.30, "brightness": 0.15}
        weights = {name: float(raw_weights.get(name, weight)) for name, weight in defaults.items()}
        if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("scene.stage_a.weights 必须为非负数且总和大于 0")
        unknown = set(raw_weights) - set(defaults)
        if unknown:
            raise ValueError(f"scene.stage_a.weights 包含未知检测器: {sorted(unknown)}")
        smoothing_window = _positive_int(
            value.get("smoothing_window_frames", 5),
            "scene.stage_a.smoothing_window_frames",
        )
        if smoothing_window % 2 == 0:
            raise ValueError("scene.stage_a.smoothing_window_frames 必须是奇数")
        return cls(
            smoothing_window_frames=smoothing_window,
            merge_window_s=_positive(value.get("merge_window_s", 0.5), "scene.stage_a.merge_window_s"),
            weights=weights,
            histogram=HistogramConfig.from_mapping(_mapping(value.get("histogram", {}), "scene.stage_a.histogram")),
            ssim=SSIMConfig.from_mapping(_mapping(value.get("ssim", {}), "scene.stage_a.ssim")),
            optical_flow=OpticalFlowConfig.from_mapping(_mapping(value.get("optical_flow", {}), "scene.stage_a.optical_flow")),
            brightness=BrightnessConfig.from_mapping(_mapping(value.get("brightness", {}), "scene.stage_a.brightness")),
        )


@dataclass(frozen=True)
class DinoConfig:
    enabled: bool = True
    model: str = "facebook/dinov2-small"
    sample_fps: float = 1.0
    candidate_context_s: float = 2.0
    z_score_window: int = 30
    z_score_threshold: float = 2.0
    min_z_score_samples: int = 3
    batch_size: int = 8
    device: str = "cpu"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DinoConfig:
        config = cls(
            enabled=bool(value.get("enabled", True)),
            model=str(value.get("model", "facebook/dinov2-small")),
            sample_fps=_positive(value.get("sample_fps", 1.0), "scene.stage_b.sample_fps"),
            candidate_context_s=_positive(value.get("candidate_context_s", 2.0), "scene.stage_b.candidate_context_s"),
            z_score_window=_positive_int(value.get("z_score_window", 30), "scene.stage_b.z_score_window"),
            z_score_threshold=_positive(value.get("z_score_threshold", 2.0), "scene.stage_b.z_score_threshold"),
            min_z_score_samples=_positive_int(value.get("min_z_score_samples", 3), "scene.stage_b.min_z_score_samples"),
            batch_size=_positive_int(value.get("batch_size", 8), "scene.stage_b.batch_size"),
            device=str(value.get("device", "cpu")),
        )
        if not config.model.strip() or not config.device.strip():
            raise ValueError("scene.stage_b.model 和 device 不能为空")
        if config.model != "facebook/dinov2-small":
            raise ValueError(
                "scene.stage_b.model v1 仅支持 facebook/dinov2-small"
            )
        return config


@dataclass(frozen=True)
class FusionConfig:
    min_scene_duration_s: float = 3.0
    hysteresis_frames: int = 2
    same_scene_similarity: float = 0.95
    low_confidence_threshold: float = 0.6

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> FusionConfig:
        return cls(
            min_scene_duration_s=_positive(value.get("min_scene_duration_s", 3.0), "scene.fusion.min_scene_duration_s"),
            hysteresis_frames=_positive_int(value.get("hysteresis_frames", 2), "scene.fusion.hysteresis_frames"),
            same_scene_similarity=_unit(value.get("same_scene_similarity", 0.95), "scene.fusion.same_scene_similarity"),
            low_confidence_threshold=_unit(value.get("low_confidence_threshold", 0.6), "scene.fusion.low_confidence_threshold"),
        )


@dataclass(frozen=True)
class VLMConfig:
    enabled: bool = True
    base_url: str = ""
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 60.0
    review_confidence_threshold: float = 0.6

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> VLMConfig:
        config = cls(
            enabled=bool(value.get("enabled", True)),
            base_url=str(value.get("base_url", "")),
            model=str(value.get("model", "")),
            api_key_env=str(value.get("api_key_env", "OPENAI_API_KEY")),
            timeout_s=_positive(value.get("timeout_s", 60.0), "scene.vlm.timeout_s"),
            review_confidence_threshold=_unit(value.get("review_confidence_threshold", 0.6), "scene.vlm.review_confidence_threshold"),
        )
        if not config.api_key_env.strip():
            raise ValueError("scene.vlm.api_key_env 不能为空")
        return config


@dataclass(frozen=True)
class SceneGovernanceConfig:
    calibration_status: str = "provisional"
    soft_boundary_action: str = "quarantine"
    auto_finalize_soft_boundaries: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SceneGovernanceConfig:
        config = cls(
            calibration_status=str(
                value.get("calibration_status", "provisional")
            ),
            soft_boundary_action=str(
                value.get("soft_boundary_action", "quarantine")
            ),
            auto_finalize_soft_boundaries=bool(
                value.get("auto_finalize_soft_boundaries", False)
            ),
        )
        if config.calibration_status not in {"provisional", "calibrated"}:
            raise ValueError(
                "scene.governance.calibration_status 必须是 provisional 或 calibrated"
            )
        if config.soft_boundary_action not in {"quarantine", "accept"}:
            raise ValueError(
                "scene.governance.soft_boundary_action 必须是 quarantine 或 accept"
            )
        if config.calibration_status == "provisional":
            if config.soft_boundary_action != "quarantine":
                raise ValueError("未经金标校准的软边界必须进入 quarantine")
            if config.auto_finalize_soft_boundaries:
                raise ValueError("未经金标校准时禁止自动定稿软边界")
        return config


@dataclass(frozen=True)
class SceneConfig:
    path: Path
    document: dict[str, Any]
    profile: str | None
    enabled: bool
    output_dir: Path
    stage_a: StageAConfig
    stage_b: DinoConfig
    fusion: FusionConfig
    vlm: VLMConfig
    governance: SceneGovernanceConfig
    config_hash: str

    @classmethod
    def load(cls, path: str | Path) -> SceneConfig:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Scene 配置文件不存在: {config_path}")
        with config_path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        document = copy.deepcopy(_mapping(loaded, "配置文件顶层"))
        return cls._from_document(document, config_path=config_path)

    @classmethod
    def load_with_profile(
        cls,
        default_path: str | Path,
        profile_path: str | Path,
    ) -> SceneConfig:
        default_config_path = Path(default_path).expanduser().resolve()
        profile_config_path = Path(profile_path).expanduser().resolve()
        if not default_config_path.is_file():
            raise FileNotFoundError(f"Scene 默认配置文件不存在: {default_config_path}")
        if not profile_config_path.is_file():
            raise FileNotFoundError(f"Scene Profile 配置文件不存在: {profile_config_path}")
        with default_config_path.open(encoding="utf-8") as file:
            default_document = _mapping(yaml.safe_load(file), "Scene 默认配置顶层")
        with profile_config_path.open(encoding="utf-8") as file:
            profile_document = _mapping(yaml.safe_load(file), "Profile 配置顶层")
        profile_name = str(profile_document.get("profile", "")).strip()
        if not profile_name:
            raise ValueError("Profile 配置必须包含非空 profile")
        scene_override = _mapping(profile_document.get("scene"), "Profile.scene")
        merged_document = copy.deepcopy(default_document)
        merged_document["profile"] = profile_name
        merged_document["scene"] = _deep_merge(
            _mapping(default_document.get("scene"), "scene"),
            scene_override,
        )
        return cls._from_document(
            merged_document,
            config_path=default_config_path,
        )

    @classmethod
    def _from_document(
        cls,
        document: dict[str, Any],
        *,
        config_path: Path,
    ) -> SceneConfig:
        scene = _mapping(document.get("scene"), "scene")
        output_raw = str(scene.get("output_dir", "../../output/scene"))
        output_dir = Path(output_raw).expanduser()
        if not output_dir.is_absolute():
            output_dir = config_path.parent / output_dir
        return cls(
            path=config_path,
            document=document,
            profile=(
                str(document["profile"]).strip()
                if document.get("profile") is not None
                else None
            ),
            enabled=bool(scene.get("enabled", True)),
            output_dir=output_dir.resolve(),
            stage_a=StageAConfig.from_mapping(_mapping(scene.get("stage_a", {}), "scene.stage_a")),
            stage_b=DinoConfig.from_mapping(_mapping(scene.get("stage_b", {}), "scene.stage_b")),
            fusion=FusionConfig.from_mapping(_mapping(scene.get("fusion", {}), "scene.fusion")),
            vlm=VLMConfig.from_mapping(_mapping(scene.get("vlm", {}), "scene.vlm")),
            governance=SceneGovernanceConfig.from_mapping(
                _mapping(scene.get("governance", {}), "scene.governance")
            ),
            config_hash=_config_hash(document),
        )


__all__ = [
    "BrightnessConfig",
    "DinoConfig",
    "FusionConfig",
    "HistogramConfig",
    "OpticalFlowConfig",
    "SSIMConfig",
    "SceneConfig",
    "SceneGovernanceConfig",
    "StageAConfig",
    "VLMConfig",
]
