"""BaseAdapter — 所有适配器的抽象基类。"""

from abc import ABC, abstractmethod

from zpds.core.types import (
    CalibrationDescriptor,
    ClockDescriptor,
    SessionInventory,
    SourceStream,
)

from .contracts import ValidationReport


class BaseAdapter(ABC):
    """容器适配器基类。"""

    @abstractmethod
    def inspect(self, path: str) -> SessionInventory:
        """扫描路径/文件，返回会话清单。"""
        ...

    @abstractmethod
    def validate(self, path: str) -> ValidationReport:
        """快速校验容器完整性（header/magic/schema）。"""
        ...

    def read_stream_catalog(self, path: str) -> list[SourceStream]:
        """读取流目录；默认复用轻量 inspect 结果。"""
        return self.inspect(path).streams

    def read_clock_catalog(self, path: str) -> list[ClockDescriptor]:
        """读取时钟目录；默认复用轻量 inspect 结果。"""
        return self.inspect(path).clocks

    def read_calibration_catalog(self, path: str) -> list[CalibrationDescriptor]:
        """读取来源中明确记录的标定目录，不在 Adapter 层估计新标定。"""
        return self.inspect(path).calibrations

    def scan(self, path: str) -> ValidationReport:
        """全量读取或解码扫描；默认退化为快速 validate。"""
        report = self.validate(path)
        if isinstance(report, ValidationReport):
            return report
        return ValidationReport(
            checked_assets=1,
            metadata={"legacy_validate": bool(report)},
        )

    def analyze_time(self, path: str) -> ValidationReport:
        """Stage 2 时间检查；通用默认仅报告已声明时钟数量。"""
        clocks = self.read_clock_catalog(path)
        return ValidationReport(
            checked_records=len(clocks),
            metadata={"clock_count": len(clocks)},
        )
