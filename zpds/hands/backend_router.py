"""按数据源类型选择 Hands 2D 主后端。"""

from __future__ import annotations

from dataclasses import dataclass

VALID_MODEL_BACKENDS = frozenset({"wilor", "mediapipe"})


@dataclass(frozen=True)
class HandsBackendPolicy:
    """ego、非 ego 和 2D 回退模型的选择策略。"""

    ego_bbox_backend: str = "mediapipe"
    non_ego_bbox_backend: str = "mediapipe"
    fallback_2d_backend: str = "mediapipe"

    def __post_init__(self) -> None:
        for field_name in (
            "ego_bbox_backend",
            "non_ego_bbox_backend",
            "fallback_2d_backend",
        ):
            value = getattr(self, field_name)
            if value not in VALID_MODEL_BACKENDS:
                raise ValueError(
                    f"hands.{field_name} 必须是 "
                    f"{sorted(VALID_MODEL_BACKENDS)}，实际为 {value!r}"
                )


class HandsBackendRouter:
    """由 Profile 明确提供的 ``is_ego`` 决定主模型。

    Router 不根据 stream 名称猜测 ego 属性，避免把 ``ego_rgb`` 等命名约定
    当成数据语义。来源 Profile 负责提供可靠的 ``is_ego``。
    """

    def __init__(self, policy: HandsBackendPolicy) -> None:
        self._policy = policy

    @property
    def fallback_2d_backend(self) -> str:
        return self._policy.fallback_2d_backend

    def select_backend(self, *, is_ego: bool) -> str:
        if is_ego:
            return self._policy.ego_bbox_backend
        return self._policy.non_ego_bbox_backend


__all__ = [
    "VALID_MODEL_BACKENDS",
    "HandsBackendPolicy",
    "HandsBackendRouter",
]
