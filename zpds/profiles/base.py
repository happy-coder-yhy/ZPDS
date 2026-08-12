"""BaseProfile — profile 基类。"""

from dataclasses import dataclass, field


@dataclass
class BaseProfile:
    """采集源 profile 基类。"""

    name: str
    description: str = ""
    modalities: dict[str, str] = field(default_factory=dict)
    # 兼容仅有一个默认主视角的旧配置。
    primary_stream_id: str | None = None
    # 没有天然主副关系时可以同时声明多个主摄。
    primary_stream_ids: tuple[str, ...] = ()

    def applicability_for(self, modality: str) -> str:
        """Return the declared modality applicability, or ``unavailable``."""
        return self.modalities.get(modality, "unavailable")
