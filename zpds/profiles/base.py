"""BaseProfile — profile 基类。"""

from dataclasses import dataclass, field


@dataclass
class BaseProfile:
    """采集源 profile 基类。"""

    name: str
    description: str = ""
    modalities: dict[str, str] = field(default_factory=dict)

    def applicability_for(self, modality: str) -> str:
        """Return the declared modality applicability, or ``unavailable``."""
        return self.modalities.get(modality, "unavailable")
