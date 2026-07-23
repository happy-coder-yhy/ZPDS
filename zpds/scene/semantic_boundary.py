"""CLIP / DINO / VideoMAE embedding 语义边界检测。"""


class SemanticBoundaryDetector:
    """语义边界检测器。"""

    def detect(self, video_path: str) -> list[int]:
        """返回语义边界帧号列表。"""
        raise NotImplementedError
