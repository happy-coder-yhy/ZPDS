"""数据溯源：来源、生产者、流水线版本。"""

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Origin:
    """数据来源信息。"""
    source_profile: str
    device_serial: str = ""
    recording_time_local: str = ""
    raw_path: str = ""


class OriginKind(str, Enum):
    """持久化值或标注的来源类别。"""

    SOURCE_RECORDED = "source_recorded"
    DETERMINISTIC_TRANSFORM = "deterministic_transform"
    MODEL_ESTIMATED = "model_estimated"
    HUMAN_ANNOTATED = "human_annotated"
    SIMULATION_GROUND_TRUTH = "simulation_ground_truth"
    UNKNOWN = "unknown"


@dataclass
class Producer:
    """处理生产者。"""

    producer_id: str = "zpds"
    name: str = "zpds"
    pipeline: str = "zpds"
    version: str = "0.1.0"
    code_commit: str = ""
    config_version: str = "0.1.0"
    config_uri: str = ""
    config_hash: str = ""


@dataclass
class ValueOrigin:
    """某个派生值、Stream 或 Annotation 的可复现来源。"""

    kind: OriginKind
    producer_id: str
    source_refs: list[str] = field(default_factory=list)
    operation: str = ""
    sample_map_uri: str = ""
    model_name: str = ""
    model_version: str = ""
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not self.source_refs:
            raise ValueError("source_refs must contain at least one source reference")
        if self.kind == OriginKind.MODEL_ESTIMATED and (
            not self.model_name or not self.model_version
        ):
            raise ValueError(
                "model_name and model_version are required for model_estimated origin"
            )


@dataclass
class PipelineVersion:
    """流水线版本快照。"""
    zpds_version: str = "0.1.0"
    adapters_version: str = "0.1.0"
    qc_config_version: str = "0.1.0"
    config_hashes: dict = field(default_factory=dict)
