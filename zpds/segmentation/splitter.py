"""中间坏区间切分。"""

from zpds.core.types import SpanProposal


class Splitter:
    """坏区间切分器。"""

    def split(self, span: SpanProposal, bad_regions: list[tuple[int, int]]) -> list[SpanProposal]:
        """在坏区间处切分，返回子区间列表。"""
        raise NotImplementedError
