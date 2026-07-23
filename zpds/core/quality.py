"""质量指标与报告模型。"""

from dataclasses import dataclass, field
from .decisions import Decision


@dataclass
class QualityMetric:
    """单项质量指标（0.0–1.0 归一化）。"""
    name: str
    value: float
    threshold: float = 0.8
    pass_: bool = True

    def __post_init__(self):
        self.pass_ = self.value >= self.threshold


@dataclass
class QualityReport:
    """完整质量报告。"""
    session_id: str
    segment_id: str = ""
    decisions: list[Decision] = field(default_factory=list)
    metrics: list[QualityMetric] = field(default_factory=list)
    overall_pass: bool = True

    @property
    def fatal_count(self) -> int:
        return sum(1 for d in self.decisions if d.severity == "fatal")

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.decisions if d.severity == "error")

    @property
    def warn_count(self) -> int:
        return sum(1 for d in self.decisions if d.severity == "warn")
