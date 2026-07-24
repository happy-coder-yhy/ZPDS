"""容器适配器：MP4/MKV/MOV、MCAP、ROS BAG、HDF5、设备日志、仿真日志。"""

from .a2d import A2DAdapter
from .base import BaseAdapter
from .contracts import (
    ContainerMessage,
    IssueLevel,
    OptionalDependencyError,
    ValidationIssue,
    ValidationReport,
)
from .epic import Epic100Adapter
from .guida import GuidaAdapter
from .profiled_mcap import ProfiledMcapAdapter
from .registry import create_adapter, list_adapter_profiles

__all__ = [
    "A2DAdapter",
    "BaseAdapter",
    "ContainerMessage",
    "Epic100Adapter",
    "GuidaAdapter",
    "IssueLevel",
    "OptionalDependencyError",
    "ProfiledMcapAdapter",
    "ValidationIssue",
    "ValidationReport",
    "create_adapter",
    "list_adapter_profiles",
]
