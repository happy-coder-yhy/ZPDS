"""HDF5 文件探测器。"""

from zpds.adapters.base import BaseAdapter
from zpds.core.types import SessionInventory


class Hdf5Inspector(BaseAdapter):
    """HDF5 文件探测器。"""

    def inspect(self, path: str) -> SessionInventory:
        raise NotImplementedError

    def validate(self, path: str) -> bool:
        raise NotImplementedError
