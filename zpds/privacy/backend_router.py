"""按 Profile 适用性选择隐私脱敏后端。

Router 不根据 stream 名称猜测是否为人脸源，Profile 负责提供可靠的
applicable/not_applicable/unavailable 语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Applicability = Literal["applicable", "not_applicable", "unavailable"]

VALID_APPLICABILITY = frozenset({"applicable", "not_applicable", "unavailable"})
VALID_FACE_BACKENDS = frozenset({"yolo11n_face"})
VALID_TEXT_BACKENDS = frozenset({"easyocr"})
VALID_PII_BACKENDS = frozenset({"llm"})


@dataclass(frozen=True)
class PrivacyBackendPolicy:
    """各 Profile 对隐私子系统的适用性声明。

    face/text 单独控制：
    - ``applicable``: 运行检测
    - ``not_applicable``: 跳过且不产生误报（如机器人相机无操作者人脸）
    - ``unavailable``: 数据未取回，暂不运行（如 EPIC 原视频）
    """

    face_applicability: Applicability = "applicable"
    text_applicability: Applicability = "applicable"

    face_backend: str = "yolo11n_face"
    text_backend: str = "easyocr"
    pii_backend: str = "llm"

    def __post_init__(self) -> None:
        if self.face_applicability not in VALID_APPLICABILITY:
            raise ValueError(
                f"face_applicability 必须是 {sorted(VALID_APPLICABILITY)}"
            )
        if self.text_applicability not in VALID_APPLICABILITY:
            raise ValueError(
                f"text_applicability 必须是 {sorted(VALID_APPLICABILITY)}"
            )
        if self.face_backend not in VALID_FACE_BACKENDS:
            raise ValueError(
                f"face_backend 必须是 {sorted(VALID_FACE_BACKENDS)}"
            )
        if self.text_backend not in VALID_TEXT_BACKENDS:
            raise ValueError(
                f"text_backend 必须是 {sorted(VALID_TEXT_BACKENDS)}"
            )
        if self.pii_backend not in VALID_PII_BACKENDS:
            raise ValueError(
                f"pii_backend 必须是 {sorted(VALID_PII_BACKENDS)}"
            )

    @classmethod
    def from_profile(cls, profile: str) -> PrivacyBackendPolicy:
        """根据 profile 名返回默认适用性。

        人脸：仅墨现/EPIC 可能拍到操作者面部；机器人/夹爪相机不适用。
        文本：所有场景都可能拍到标签/文档/屏幕。
        """
        face: Applicability
        text: Applicability = "applicable"

        if profile in ("guida_ego", "guida"):
            face = "applicable"
        elif profile in ("dunjia_ego", "dunjia", "jianzhi_umi", "umi", "a2d_robot", "a2d"):
            face = "not_applicable"
        elif profile in ("epic100", "epic"):
            face = "unavailable"
        else:
            face = "applicable"

        return cls(
            face_applicability=face,
            text_applicability=text,
        )

    @property
    def face_enabled(self) -> bool:
        return self.face_applicability == "applicable"

    @property
    def text_enabled(self) -> bool:
        return self.text_applicability == "applicable"


class PrivacyBackendRouter:
    """根据 Profile 声明路由各检测器的启用/跳过。"""

    def __init__(self, policy: PrivacyBackendPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> PrivacyBackendPolicy:
        return self._policy

    @property
    def face_enabled(self) -> bool:
        return self._policy.face_enabled

    @property
    def text_enabled(self) -> bool:
        return self._policy.text_enabled

    def should_run_face(self) -> bool:
        return self._policy.face_enabled

    def should_run_text(self) -> bool:
        return self._policy.text_enabled


__all__ = [
    "VALID_APPLICABILITY",
    "VALID_FACE_BACKENDS",
    "VALID_PII_BACKENDS",
    "VALID_TEXT_BACKENDS",
    "Applicability",
    "PrivacyBackendPolicy",
    "PrivacyBackendRouter",
]
