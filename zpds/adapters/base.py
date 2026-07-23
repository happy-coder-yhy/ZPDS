"""BaseAdapter — 所有适配器的抽象基类。"""

from abc import ABC, abstractmethod
from zpds.core.types import SessionInventory


class BaseAdapter(ABC):
    """容器适配器基类。"""

    @abstractmethod
    def inspect(self, path: str) -> SessionInventory:
        """扫描路径/文件，返回会话清单。"""
        ...

    @abstractmethod
    def validate(self, path: str) -> bool:
        """快速校验容器完整性（header/magic/schema）。"""
        ...
