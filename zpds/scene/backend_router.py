"""场景检测后端选择策略；模型实现保持惰性导入。"""

from __future__ import annotations

from dataclasses import dataclass

from zpds.scene.config import SceneConfig

STAGE_A_BACKENDS = ("histogram", "ssim", "optical_flow", "brightness")
SEMANTIC_BACKENDS = ("dino",)


@dataclass(frozen=True)
class SceneBackendPolicy:
    enabled: bool
    stage_a_backends: tuple[str, ...]
    semantic_backend: str | None


class SceneBackendRouter:
    """仅返回配置选择，不在路由阶段导入 torch/transformers。"""

    @classmethod
    def from_config(cls, config: SceneConfig) -> SceneBackendRouter:
        enabled = tuple(
            name
            for name in STAGE_A_BACKENDS
            if getattr(config.stage_a, name).enabled
        )
        semantic = "dino" if config.stage_b.enabled else None
        return cls(SceneBackendPolicy(config.enabled, enabled, semantic))

    def __init__(self, policy: SceneBackendPolicy) -> None:
        unknown = set(policy.stage_a_backends) - set(STAGE_A_BACKENDS)
        if unknown:
            raise ValueError(f"未知 Stage A 后端: {sorted(unknown)}")
        if policy.semantic_backend not in {*SEMANTIC_BACKENDS, None}:
            raise ValueError(f"未知语义后端: {policy.semantic_backend!r}")
        self.policy = policy


__all__ = [
    "SEMANTIC_BACKENDS",
    "STAGE_A_BACKENDS",
    "SceneBackendPolicy",
    "SceneBackendRouter",
]
