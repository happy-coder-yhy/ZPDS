"""Adapter 的验证、消息与扫描契约。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OptionalDependencyError(ImportError):
    """Adapter 所需 optional extra 未安装。"""


class IssueLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    level: IssueLevel
    message: str
    path: str = ""
    stream_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("issue code and message must not be empty")


@dataclass(frozen=True)
class ValidationReport:
    """快速或全量验证结果；bool(report) 等价于 report.passed。"""

    issues: tuple[ValidationIssue, ...] = ()
    checked_assets: int = 0
    checked_records: int = 0
    decoded_records: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        blocking = {IssueLevel.ERROR, IssueLevel.FATAL}
        return not any(issue.level in blocking for issue in self.issues)

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class ContainerMessage:
    """容器无关的消息信封；payload 保持原始 bytes。"""

    stream_id: str
    log_time_ns: int
    publish_time_ns: int | None
    sequence: int
    payload: bytes
    schema_name: str = ""
    encoding: str = ""
