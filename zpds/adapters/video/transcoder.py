"""CFR 转码 + sample_map 生成。"""


class VideoTranscoder:
    """变帧率→恒帧率转码器。"""

    def transcode(self, src: str, dst: str, fps: float = 30.0) -> str:
        """转码为 CFR 并返回输出路径。"""
        raise NotImplementedError

    def build_sample_map(self, src: str, dst: str) -> dict:
        """建立源帧→目标帧映射。"""
        raise NotImplementedError
