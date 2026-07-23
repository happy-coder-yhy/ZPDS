"""LeRobotDataset v3 格式导出。"""

from .base import BaseExporter


class LeRobotExporter(BaseExporter):
    """导出为 LeRobotDataset v3 格式。"""

    def export(self, release_id: str, output_dir: str) -> str:
        raise NotImplementedError
