"""MCAP info/doctor、topic inventory。"""

from zpds.adapters.base import BaseAdapter
from zpds.core.types import SessionInventory


class McapInspector(BaseAdapter):
    """MCAP 容器探测器。"""

    def inspect(self, path: str) -> SessionInventory:
        raise NotImplementedError

    def validate(self, path: str) -> bool:
        raise NotImplementedError
