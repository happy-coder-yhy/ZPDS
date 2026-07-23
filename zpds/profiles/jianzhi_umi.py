"""简智新创 UMI profile。"""

from .base import BaseProfile


class JianzhiUmiProfile(BaseProfile):
    """简智新创 UMI 双端夹爪 profile。"""

    def __init__(self):
        super().__init__(
            name="jianzhi_umi",
            description="简智新创 UMI 双端夹爪 teleop：MCAP，双 robot，H264 + IMU + 磁编码器 + VIO",
        )
