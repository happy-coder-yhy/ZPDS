"""编辑/硬切候选检测（PySceneDetect、TransNetV2）。"""


class CutDetector:
    """场景切点检测器。"""

    def detect(self, video_path: str) -> list[int]:
        """返回切点帧号列表。"""
        raise NotImplementedError
