"""Storage Adapter 的公共协议与值对象。"""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable


class StorageError(RuntimeError):
    """Storage 操作失败。"""


class InvalidStorageReference(StorageError, ValueError):
    """逻辑引用格式错误或尝试越过配置根目录。"""


class ArtifactExistsError(StorageError, FileExistsError):
    """原子目录目标已存在，拒绝覆盖完整产物。"""


@dataclass(frozen=True)
class StoredFile:
    """已完整落盘的文件元数据。"""

    reference: str
    path: Path
    size_bytes: int
    sha256: str


@runtime_checkable
class StorageAdapter(Protocol):
    """可由本地文件系统或对象存储实现的最小读取协议。"""

    def open_read(self, reference: str) -> BinaryIO:
        """以二进制、只读方式打开逻辑引用。"""
        ...

    def exists(self, reference: str) -> bool:
        """判断逻辑引用是否存在。"""
        ...

    def sha256(self, reference: str) -> str:
        """流式计算逻辑引用指向文件的 SHA256。"""
        ...
