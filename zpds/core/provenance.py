"""数据溯源：来源、生产者、流水线版本。"""

from dataclasses import dataclass


@dataclass
class Origin:
    """数据来源信息。"""
    source_profile: str
    device_serial: str = ""
    recording_time_local: str = ""
    raw_path: str = ""


@dataclass
class Producer:
    """处理生产者。"""
    pipeline: str = "zpds"
    version: str = "0.1.0"
    config_hash: str = ""


@dataclass
class PipelineVersion:
    """流水线版本快照。"""
    zpds_version: str = "0.1.0"
    adapters_version: str = "0.1.0"
    qc_config_version: str = "0.1.0"
    config_hashes: dict = None

    def __post_init__(self):
        if self.config_hashes is None:
            self.config_hashes = {}
