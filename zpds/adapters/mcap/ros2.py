"""ROS2 CDR 编码 MCAP 专用读取。"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from zpds.adapters.common import require_file, require_optional_module


class Ros2McapReader:
    """ROS2 CDR MCAP 读取器。"""

    def __init__(self, path: str):
        self.path = require_file(path)

    def iter_decoded(self, topic: str | None = None) -> Iterator[Any]:
        ros2_reader = require_optional_module("mcap_ros2.reader", "mcap")
        topics = [topic] if topic is not None else None
        yield from ros2_reader.read_ros2_messages(
            Path(self.path),
            topics=topics,
        )
