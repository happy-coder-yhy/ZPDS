"""首尾 idle 裁剪。"""

from zpds.core.types import SpanProposal


class Trimmer:
    """首尾裁剪器。"""

    def trim(self, span: SpanProposal, video_path: str) -> SpanProposal:
        """裁掉首尾 idle 帧，返回收紧后的区间。"""
        raise NotImplementedError
