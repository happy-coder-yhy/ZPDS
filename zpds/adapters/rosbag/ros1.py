"""ROS1 BAG 适配器。"""

from zpds.adapters.base import BaseAdapter
from zpds.core.types import SessionInventory


class Ros1BagAdapter(BaseAdapter):
    """ROS1 .bag 适配器。"""

    def inspect(self, path: str) -> SessionInventory:
        raise NotImplementedError

    def validate(self, path: str) -> bool:
        raise NotImplementedError
