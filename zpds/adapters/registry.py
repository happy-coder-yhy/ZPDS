"""Profile 名称到只读 Adapter 工厂的显式注册表。"""

from collections.abc import Callable

from .a2d import A2DAdapter
from .base import BaseAdapter
from .epic import Epic100Adapter
from .guida import GuidaAdapter
from .profiled_mcap import ProfiledMcapAdapter

AdapterFactory = Callable[[], BaseAdapter]


_FACTORIES: dict[str, AdapterFactory] = {
    "guida_ego": GuidaAdapter,
    "dunjia_ego": lambda: ProfiledMcapAdapter("dunjia_ego"),
    "jianzhi_umi": lambda: ProfiledMcapAdapter("jianzhi_umi"),
    "a2d_robot": A2DAdapter,
    "epic100": Epic100Adapter,
    "epic100_auto_annotation": Epic100Adapter,
}


def create_adapter(profile_name: str) -> BaseAdapter:
    try:
        return _FACTORIES[profile_name]()
    except KeyError as error:
        raise KeyError(f"Unknown adapter profile: {profile_name}") from error


def list_adapter_profiles() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))
