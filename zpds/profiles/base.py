"""BaseProfile — profile 基类。"""

from dataclasses import dataclass, field


@dataclass
class BaseProfile:
    """采集源 profile 基类。"""

    name: str
    description: str = ""
    modalities: dict[str, str] = field(default_factory=dict)
    # 多摄像头来源的主相机 stream_id；单相机来源为 None（segment.json 不写 is_primary）
    primary_stream_id: str | None = None

    def applicability_for(self, modality: str) -> str:
        """Return the declared modality applicability, or ``unavailable``."""
        return self.modalities.get(modality, "unavailable")
