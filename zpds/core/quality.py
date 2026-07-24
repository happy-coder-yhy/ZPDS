"""质量指标与报告模型。"""

from dataclasses import dataclass, field
from enum import Enum

from .decisions import Decision, DecisionType, Severity


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    IN_RANGE = "in_range"


@dataclass
class QualityMetric:
    """单项质量指标，支持不同阈值方向。"""

    name: str
    value: float
    unit: str = "1"
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    threshold: float | None = 0.8
    lower_bound: float | None = None
    upper_bound: float | None = None
    applicable: bool = True
    pass_: bool = field(init=False)

    def __post_init__(self) -> None:
        if not self.applicable:
            self.pass_ = True
        elif self.direction == MetricDirection.HIGHER_IS_BETTER:
            if self.threshold is None:
                raise ValueError("threshold is required for higher_is_better")
            self.pass_ = self.value >= self.threshold
        elif self.direction == MetricDirection.LOWER_IS_BETTER:
            if self.threshold is None:
                raise ValueError("threshold is required for lower_is_better")
            self.pass_ = self.value <= self.threshold
        else:
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("lower_bound and upper_bound are required for in_range")
            self.pass_ = self.lower_bound <= self.value <= self.upper_bound


@dataclass(frozen=True)
class QualityView:
    """面向一种下游用途的多维质量视图。"""

    view_id: str
    version: str
    required_metrics: tuple[str, ...]
    optional_metrics: tuple[str, ...] = ()


@dataclass
class QualityReport:
    """完整质量报告。"""
    session_id: str
    segment_id: str = ""
    decisions: list[Decision] = field(default_factory=list)
    metrics: list[QualityMetric] = field(default_factory=list)

    @property
    def overall_pass(self) -> bool:
        """始终根据当前指标和决策计算，避免列表更新后状态过期。"""

        blocking = {DecisionType.QUARANTINE, DecisionType.REJECT}
        return all(metric.pass_ for metric in self.metrics if metric.applicable) and not any(
            decision.decision in blocking for decision in self.decisions
        )

    @property
    def fatal_count(self) -> int:
        return sum(1 for d in self.decisions if d.severity == Severity.FATAL)

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.decisions if d.severity == Severity.ERROR)

    @property
    def warn_count(self) -> int:
        return sum(1 for d in self.decisions if d.severity == Severity.WARN)
