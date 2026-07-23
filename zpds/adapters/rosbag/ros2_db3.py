"""ROS2 DB3 (sqlite3) 适配器。"""

from zpds.adapters.base import BaseAdapter
from zpds.core.types import SessionInventory


class Ros2Db3Adapter(BaseAdapter):
    """ROS2 DB3 适配器。"""

    def inspect(self, path: str) -> SessionInventory:
        raise NotImplementedError

    def validate(self, path: str) -> bool:
        raise NotImplementedError
