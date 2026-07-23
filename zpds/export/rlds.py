"""RLDS / Open X-Embodiment 格式导出。"""

from .base import BaseExporter


class RldsExporter(BaseExporter):
    """导出为 RLDS / Open X-Embodiment 格式。"""

    def export(self, release_id: str, output_dir: str) -> str:
        raise NotImplementedError
