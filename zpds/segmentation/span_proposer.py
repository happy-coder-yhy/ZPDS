"""物理有效区间提议。"""

from zpds.core.types import SpanProposal


class SpanProposer:
    """有效区间提议器。"""

    def propose(self, decisions: list) -> list[SpanProposal]:
        """根据 QC 决策列表生成有效区间。"""
        raise NotImplementedError
