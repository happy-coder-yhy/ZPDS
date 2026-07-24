"""遁甲 ego profile。"""

from .base import BaseProfile


class DunjiaEgoProfile(BaseProfile):
    """遁甲头戴多相机 MCAP profile。"""

    def __init__(self):
        super().__init__(
            name="dunjia_ego",
            description="遁甲头戴多相机：Foxglove MCAP，3× RGB H264 + 深度 PNG + IMU",
            adapter_kind="mcap",
            required_globs=("*.mcap",),
            optional_globs=("*.pdf",),
            metadata={
                "preserve_log_and_publish_time": True,
                "video_encoding": "h264",
                "required_topic_suffixes": (
                    "/sensor/imu",
                    "/sensor/depth/compressed",
                ),
                "required_topic_fragments": ("/camera0/compressed",),
            },
        )
