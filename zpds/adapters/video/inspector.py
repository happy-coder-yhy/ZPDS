"""ffprobe 解析：帧数、码流、元数据提取。"""

from zpds.adapters.base import BaseAdapter
from zpds.core.types import SessionInventory


class VideoInspector(BaseAdapter):
    """视频容器探测器。"""

    def inspect(self, path: str) -> SessionInventory:
        raise NotImplementedError

    def validate(self, path: str) -> bool:
        raise NotImplementedError
