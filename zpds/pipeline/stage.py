"""可供 Adapter、QC 和 Prepared 阶段共同使用的 Stage 契约。"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from zpds.config import LoadedConfig
from zpds.core.decisions import Decision, Evidence
from zpds.core.quality import QualityMetric

SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
CONFIG_HASH_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StageDescriptor:
    """Stage 的稳定身份。"""

    stage_id: int
    name: str
    version: str

    def __post_init__(self) -> None:
        if not 0 <= self.stage_id <= 12:
            raise ValueError("stage_id must be between 0 and 12")
        if not self.name or not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("stage name must use lower_snake_case")
        if not SEMVER_PATTERN.fullmatch(self.version):
            raise ValueError("stage version must use SemVer")


@dataclass(frozen=True)
class StageContext:
    """一次 Stage 执行所需的不可变上下文。"""

    run_id: str
    session_id: str
    input_refs: tuple[str, ...]
    config: LoadedConfig
    code_version: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.input_refs or any(not reference for reference in self.input_refs):
            raise ValueError("input_refs must contain non-empty references")
        if not self.code_version:
            raise ValueError("code_version must not be empty")


@dataclass(frozen=True)
class StageResult:
    """Stage 的终态结果；运行中状态由后续 Run Ledger 管理。"""

    descriptor: StageDescriptor
    status: StageStatus
    input_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    config_hash: str
    started_at: datetime
    finished_at: datetime
    metrics: tuple[QualityMetric, ...] = ()
    decisions: tuple[Decision, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status in {StageStatus.PENDING, StageStatus.RUNNING}:
            raise ValueError("StageResult status must be terminal")
        if not self.input_refs or any(not reference for reference in self.input_refs):
            raise ValueError("input_refs must contain non-empty references")
        if any(not reference for reference in self.output_refs):
            raise ValueError("output_refs must not contain empty references")
        if not CONFIG_HASH_PATTERN.fullmatch(self.config_hash):
            raise ValueError("config_hash must be sha256:<64 lowercase hex characters>")
        _require_timezone(self.started_at, "started_at")
        _require_timezone(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be earlier than started_at")
        if self.status == StageStatus.FAILED and not self.error:
            raise ValueError("failed StageResult requires error")
        if self.status != StageStatus.FAILED and self.error is not None:
            raise ValueError("only failed StageResult may contain error")
        mismatched_decisions = [
            decision.stage
            for decision in self.decisions
            if decision.stage != self.descriptor.stage_id
        ]
        if mismatched_decisions:
            raise ValueError(
                "all decisions must use the StageResult descriptor stage_id"
            )

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


@runtime_checkable
class PipelineStage(Protocol):
    """Runner 后续依赖的最小 Stage 接口。"""

    @property
    def descriptor(self) -> StageDescriptor:
        ...

    def execute(self, context: StageContext) -> StageResult:
        ...


def validate_stage_contract(stage: object) -> StageDescriptor:
    """尽早验证插件对象是否满足 Stage 接口和元数据规则。"""

    if not isinstance(stage, PipelineStage):
        raise TypeError("stage must provide descriptor and execute(context)")
    descriptor = stage.descriptor
    if not isinstance(descriptor, StageDescriptor):
        raise TypeError("stage descriptor must be a StageDescriptor")
    return descriptor


def _require_timezone(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
