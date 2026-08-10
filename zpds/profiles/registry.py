"""Profile 注册表。"""

from .base import BaseProfile
from .guida_ego import GuidaEgoProfile
from .dunjia_ego import DunjiaEgoProfile
from .jianzhi_umi import JianzhiUmiProfile
from .a2d_robot import A2DRobotProfile
from .epic100 import Epic100Profile


_PROFILES: dict[str, BaseProfile] = {}

_PROFILE_ALIASES = {
    "guida": "guida_ego",
    "dunjia": "dunjia_ego",
    "umi": "jianzhi_umi",
    "epic": "epic100",
    "a2d": "a2d_robot",
}


def register(profile: BaseProfile) -> None:
    _PROFILES[profile.name] = profile


def get(name: str) -> BaseProfile | None:
    return _PROFILES.get(resolve_name(name))


def resolve_name(name: str) -> str:
    """Return the canonical profile name used by configs and the registry."""
    return _PROFILE_ALIASES.get(name, name)


def list_all() -> list[str]:
    return list(_PROFILES.keys())


# 自动注册所有内置 profile
for _cls in [GuidaEgoProfile, DunjiaEgoProfile, JianzhiUmiProfile, A2DRobotProfile, Epic100Profile]:
    register(_cls())
