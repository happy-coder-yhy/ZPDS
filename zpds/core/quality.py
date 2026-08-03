"""质量指标与报告模型。"""

from dataclasses import dataclass, field
from typing import Any

from .decisions import Decision, Disposition, Severity


@dataclass(frozen=True)
class QualityView:
    """面向下游用途的独立质量视图，不能由 ``overall_pass`` 推导。"""

    name: str
    ready: bool
    applicability: str = "applicable"
    disposition: Disposition = Disposition.KEEP
    reasons: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    evidence_uris: tuple[str, ...] = ()
    producer: str = "zpds"
    version: str = "v1"
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.applicability not in {"applicable", "not_applicable", "unavailable"}:
            raise ValueError(f"invalid applicability: {self.applicability}")


@dataclass
class QualityMetric:
    """单项质量指标及其可复现的检测上下文。

    原有四个字段保留，以兼容已实现的 Stage 3/5/6/11 调用方。数值不
    强制归一化：源检测器可保留帧数、纳秒和事件计数等原始单位。
    """
    name: str
    value: float | int | str | bool | None
    threshold: float = 0.8
    pass_: bool = True
    unit: str = "ratio"
    applicability: str = "applicable"
    severity: Severity = Severity.INFO
    disposition: Disposition = Disposition.KEEP
    reason_code: str = "measurement"
    start_ns: int | None = None
    end_ns: int | None = None
    evidence_uri: str = ""
    producer: str = "zpds"
    version: str = "v1"
    config_hash: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.applicability not in {"applicable", "not_applicable", "unavailable"}:
            raise ValueError(f"invalid applicability: {self.applicability}")
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            self.pass_ = self.value >= self.threshold

    @property
    def metric_name(self) -> str:
        """Formal-contract alias for the legacy ``name`` field."""
        return self.name


@dataclass
class QualityReport:
    """完整质量报告。"""
    session_id: str
    segment_id: str = ""
    decisions: list[Decision] = field(default_factory=list)
    metrics: list[QualityMetric] = field(default_factory=list)
    quality_views: dict[str, QualityView] = field(default_factory=dict)
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
