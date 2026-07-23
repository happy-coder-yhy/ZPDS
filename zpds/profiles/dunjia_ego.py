"""遁甲 ego profile。"""

from .base import BaseProfile


class DunjiaEgoProfile(BaseProfile):
    """遁甲头戴多相机 MCAP profile。"""

    def __init__(self):
        super().__init__(
            name="dunjia_ego",
            description="遁甲头戴多相机：Foxglove MCAP，3× RGB H264 + 深度 PNG + IMU",
        )
