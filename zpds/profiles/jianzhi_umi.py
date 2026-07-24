"""简智新创 UMI profile。"""

from .base import BaseProfile


class JianzhiUmiProfile(BaseProfile):
    """简智新创 UMI 双端夹爪 profile。"""

    def __init__(self):
        super().__init__(
            name="jianzhi_umi",
            description="简智新创 UMI 双端夹爪 teleop：MCAP，双 robot，H264 + IMU + 磁编码器 + VIO",
            adapter_kind="mcap",
            required_globs=("*.mcap",),
            metadata={
                "robot_groups": ("robot0", "robot1"),
                "preserve_log_and_header_time": True,
                "forbid_interpolation_across_vio_reset": True,
                "required_topic_suffixes": (
                    "/robot0/sensor/imu",
                    "/robot1/sensor/imu",
                    "/robot0/sensor/camera0/compressed",
                    "/robot1/sensor/camera0/compressed",
                    "/robot0/sensor/magnetic_encoder",
                    "/robot1/sensor/magnetic_encoder",
                    "/robot0/vio/eef_pose",
                    "/robot1/vio/eef_pose",
                ),
            },
        )
