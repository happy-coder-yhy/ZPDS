"""墨现 Guida ego profile。"""

from .base import BaseProfile


class GuidaEgoProfile(BaseProfile):
    """Guida V2 ego 采集 profile — MKV 容器 + JSONL 索引 + CSV IMU。"""

    def __init__(self):
        super().__init__(
            name="guida_ego",
            description="墨现 Guida V2 头戴 ego 采集：color/depth MKV + index.jsonl + IMU CSV",
            adapter_kind="guida",
            required_globs=("meta.json", "index.jsonl"),
            optional_globs=(
                "color_*.mkv",
                "depth_*.mkv",
                "color*.mp4",
                "depth*.mp4",
                "imu/imu_*.csv",
                "log/*",
            ),
            metadata={
                "authoritative_clock": "index_timestamp",
                "imu_pattern": "imu/imu_*.csv",
            },
        )
