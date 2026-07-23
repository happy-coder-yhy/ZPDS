"""稀疏 hand detector。"""


class HandDetector:
    """手部检测器封装。"""

    def detect(self, frame) -> list[dict]:
        """检测手部边界框。"""
        raise NotImplementedError
