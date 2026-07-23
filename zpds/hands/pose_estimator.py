"""WiLoR / HaWoR 手部姿态估计封装。"""


class HandPoseEstimator:
    """手部姿态估计器。"""

    def estimate(self, frame, bbox) -> dict:
        """估计手部 3D 姿态。"""
        raise NotImplementedError
