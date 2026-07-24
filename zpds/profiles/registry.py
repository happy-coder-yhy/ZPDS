"""Profile 注册表。"""

from .a2d_robot import A2DRobotProfile
from .base import BaseProfile
from .dunjia_ego import DunjiaEgoProfile
from .epic100 import Epic100Profile
from .guida_ego import GuidaEgoProfile
from .jianzhi_umi import JianzhiUmiProfile

_PROFILES: dict[str, BaseProfile] = {}
_ALIASES = {"epic100_auto_annotation": "epic100"}


def register(profile: BaseProfile) -> None:
    if profile.name in _PROFILES:
        raise ValueError(f"Profile already registered: {profile.name}")
    _PROFILES[profile.name] = profile


def get(name: str) -> BaseProfile | None:
    return _PROFILES.get(_ALIASES.get(name, name))


def list_all() -> list[str]:
    return sorted(_PROFILES)


# 自动注册所有内置 profile
for _cls in [
    GuidaEgoProfile,
    DunjiaEgoProfile,
    JianzhiUmiProfile,
    A2DRobotProfile,
    Epic100Profile,
]:
    register(_cls())
