"""QCCascade — 质量检查级联调度器。"""

from typing import ClassVar

from zpds.core.quality import QualityReport


class QCCascade:
    """按 stage 0–12 依次执行质量检查。"""

    stages: ClassVar[list] = []

    def run(self, session_path: str) -> QualityReport:
        """执行全级联检查，返回 QualityReport。"""
        raise NotImplementedError
