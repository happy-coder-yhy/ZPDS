"""
核心数据类型定义。

SessionInventory — 采集会话清单
SourceStream   — 单个传感器流描述
ClockDomain    — 时钟域（纳秒基准）
SpanProposal   — 有效区间提议
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StreamKind(str, Enum):
    COLOR = "color"
    DEPTH = "depth"
    IMU = "imu"
    ROBOT_STATE = "robot_state"
    ROBOT_COMMAND = "robot_command"
    MAGNETIC_ENCODER = "magnetic_encoder"
    VIO_POSE = "vio_pose"


class ClockDomain(str, Enum):
    DEVICE_MONOTONIC = "device_monotonic"
    ROS_TIME = "ros_time"
    UNIX_UTC = "unix_utc"
    CUSTOM_EPOCH = "custom_epoch"


@dataclass(frozen=True)
class ClockDescriptor:
    """一个可解释的时间域及其权威性。"""

    clock_id: str
    domain: ClockDomain
    source: str
    unit: str = "ns"
    authoritative: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.clock_id or not self.source:
            raise ValueError("clock_id and source must not be empty")
        if self.unit != "ns":
            raise ValueError("ZPDS clock unit must be ns")


@dataclass(frozen=True)
class CalibrationDescriptor:
    """来源记录的标定引用；仅登记事实，不在 Adapter 中推断新标定。"""

    calibration_id: str
    kind: str
    uri: str
    parent_frame: str = ""
    child_frame: str = ""
    format: str = ""
    source_recorded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.calibration_id or not self.kind or not self.uri:
            raise ValueError("calibration_id, kind and uri must not be empty")


@dataclass(frozen=True)
class SourceAsset:
    """Raw Session 中一个不可变源文件。"""

    asset_id: str
    uri: str
    relative_path: str
    size_bytes: int
    sha256: str | None = None
    media_type: str = "application/octet-stream"
    required: bool = True

    def __post_init__(self) -> None:
        if not self.asset_id or not self.uri or not self.relative_path:
            raise ValueError("asset identity fields must not be empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.sha256 is not None and (
            not self.sha256.startswith("sha256:") or len(self.sha256) != 71
        ):
            raise ValueError("sha256 must use sha256:<64 hex characters>")


class BoundaryKind(str, Enum):
    """时间边界类型；只有 PHYSICAL 会切分 Prepared Segment。"""

    PHYSICAL = "physical"
    SCENE = "scene"
    ACTION = "action"
    CEU = "ceu"


@dataclass(frozen=True)
class TimeRange:
    """半开时间区间 ``[start_ns, end_ns)``。"""

    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if self.start_ns < 0:
            raise ValueError("start_ns must be non-negative")
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass
class SourceStream:
    """单个传感器流描述。"""
    kind: StreamKind
    stream_id: str = ""
    role: str = ""
    clock_id: str = ""
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate_hz: float | None = None
    codec: str | None = None
    container: str | None = None
    topic: str | None = None
    encoding: str | None = None
    dtype: str | None = None
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInventory:
    """采集会话清单。"""
    session_id: str
    source_profile: str
    session_uri: str = ""
    assets: list[SourceAsset] = field(default_factory=list)
    streams: list[SourceStream] = field(default_factory=list)
    clocks: list[ClockDescriptor] = field(default_factory=list)
    calibrations: list[CalibrationDescriptor] = field(default_factory=list)
    total_frames: int = 0
    duration_s: float = 0.0
    clock_domain: ClockDomain = ClockDomain.DEVICE_MONOTONIC
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanProposal:
    """有效区间提议（起止纳秒时间戳）。"""
    start_ns: int
    end_ns: int
    confidence: float = 1.0
    reason: str = ""
    boundary_kind: BoundaryKind = BoundaryKind.PHYSICAL

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
