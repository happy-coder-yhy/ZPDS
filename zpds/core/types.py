"""
核心数据类型定义。

SessionInventory — 采集会话清单
SourceStream   — 单个传感器流描述
ClockDomain    — 时钟域（纳秒基准）
SpanProposal   — 有效区间提议
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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


@dataclass
class SourceStream:
    """单个传感器流描述。"""
    kind: StreamKind
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    sample_rate_hz: Optional[float] = None
    codec: Optional[str] = None
    container: Optional[str] = None
    topic: Optional[str] = None


@dataclass
class SessionInventory:
    """采集会话清单。"""
    session_id: str
    source_profile: str
    streams: list[SourceStream] = field(default_factory=list)
    total_frames: int = 0
    duration_s: float = 0.0
    clock_domain: ClockDomain = ClockDomain.DEVICE_MONOTONIC


@dataclass
class SpanProposal:
    """有效区间提议（起止纳秒时间戳）。"""
    start_ns: int
    end_ns: int
    confidence: float = 1.0
    reason: str = ""
