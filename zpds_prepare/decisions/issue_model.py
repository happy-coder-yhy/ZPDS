"""
统一异常记录模型。

所有检测器（黑屏、视频缺口、IMU 缺口）都输出 QualityIssue，
而不是直接 print()。下游统一消费这个结构。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityIssue:
    """一次数据质量异常的完整记录。

    Attributes:
        issue_type: 异常类型标识 (continuous_black_frames, timestamp_gap, imu_gap)
        stream_id: 所属数据流 (ego_rgb, ego_imu)
        start_ns: 异常起始时间 (纳秒，设备时钟)
        end_ns: 异常结束时间 (纳秒，设备时钟)
        severity: 严重等级 (warning, error, critical)
        decision: 处理建议 (trim, split, keep_with_flag, quarantine)
        details: 附加信息 (帧数、间隔、阈值等)
    """

    issue_type: str
    stream_id: str
    start_ns: int
    end_ns: int
    severity: str
    decision: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "issue_type": self.issue_type,
            "stream_id": self.stream_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.end_ns - self.start_ns,
            "severity": self.severity,
            "decision": self.decision,
            "details": self.details,
        }
