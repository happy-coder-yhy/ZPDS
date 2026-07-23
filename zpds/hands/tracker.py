"""ByteTrack / OC-SORT 手部跟踪。"""


class HandTracker:
    """手部跟踪器。"""

    def update(self, detections: list[dict]) -> list[dict]:
        """更新跟踪状态。"""
        raise NotImplementedError
