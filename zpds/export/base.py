"""BaseExporter — 训练格式导出基类。"""

from abc import ABC, abstractmethod


class BaseExporter(ABC):
    """训练格式导出基类。"""

    @abstractmethod
    def export(self, release_id: str, output_dir: str) -> str:
        """导出为训练格式，返回输出路径。"""
        ...
