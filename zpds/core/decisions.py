"""质量检查决策与严重等级枚举。"""

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    FATAL = "fatal"    # 阻断：segment 不可用
    ERROR = "error"    # 强制修复
    WARN = "warn"      # 建议修复
    INFO = "info"      # 仅记录


class DecisionType(str, Enum):
    """数据处理决策，与问题严重度相互独立。"""

    KEEP = "keep"
    KEEP_WITH_FLAG = "keep_with_flag"
    QUARANTINE = "quarantine"
    TRIM = "trim"
    SPLIT = "split"
    REJECT = "reject"


class ReasonCode(str, Enum):
    """预定义决策原因码，覆盖 Stage 0～12 的治理与 QC 场景。"""
    # Stage 0 — 文件清单
    FILE_MISSING = "file_missing"
    HASH_MISMATCH = "hash_mismatch"
    LICENSE_RESTRICTED = "license_restricted"
    PRIVACY_REVIEW_REQUIRED = "privacy_review_required"
    UNTRUSTED_INPUT = "untrusted_input"
    # Stage 1 — 结构
    CONTAINER_CORRUPT = "container_corrupt"
    INDEX_MISSING = "index_missing"
    SCHEMA_UNKNOWN = "schema_unknown"
    REQUIRED_STREAM_MISSING = "required_stream_missing"
    ANNOTATION_ORPHANED = "annotation_orphaned"
    # Stage 2 — 时间
    TIMESTAMP_GAP = "timestamp_gap"
    TIMESTAMP_REGRESSION = "timestamp_regression"
    CLOCK_MISALIGN = "clock_misalign"
    CLOCK_RESET = "clock_reset"
    FRAME_MAPPING_UNPROVEN = "frame_mapping_unproven"
    VIO_RESET = "vio_reset"
    # Stage 3 — 视觉
    BLACK_FRAME = "black_frame"
    OVEREXPOSED = "overexposed"
    BLUR_DETECTED = "blur_detected"
    FROZEN_FRAME = "frozen_frame"
    # Stage 4 — 视频时序
    DROPPED_FRAME = "dropped_frame"
    DUPLICATE_FRAME = "duplicate_frame"
    VFR_DETECTED = "vfr_detected"
    MOTION_ANOMALY = "motion_anomaly"
    # Stage 5 — 深度
    DEPTH_INVALID_RATIO = "depth_invalid_ratio"
    DEPTH_UNIT_UNKNOWN = "depth_unit_unknown"
    # Stage 6 — IMU
    IMU_GAP = "imu_gap"
    IMU_BIAS_DRIFT = "imu_bias_drift"
    IMU_SATURATION = "imu_saturation"
    # Stage 7 — 机器人
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    COMMAND_TIMEOUT = "command_timeout"
    GRIPPER_STALL = "gripper_stall"
    ACTION_SEMANTICS_UNKNOWN = "action_semantics_unknown"
    # Stage 8 — 标定
    INTRINSICS_MISSING = "intrinsics_missing"
    EXTRINSICS_INVALID = "extrinsics_invalid"
    REPROJECTION_ERROR_HIGH = "reprojection_error_high"
    CALIBRATION_FRAME_UNKNOWN = "calibration_frame_unknown"
    # Stage 9 — 手部
    HAND_ABSENT = "hand_absent"
    HAND_TRACK_LOST = "hand_track_lost"
    # Stage 10 — 语义
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"
    # Stage 11 — 去重
    NEAR_DUPLICATE = "near_duplicate"
    # Stage 12 — 交付
    DELIVERY_CHECK_FAIL = "delivery_check_fail"


@dataclass(frozen=True)
class Evidence:
    """可定位、可复核的决策证据。"""

    uri: str
    kind: str
    description: str = ""
    start_ns: int | None = None
    end_ns: int | None = None

    def __post_init__(self) -> None:
        if self.end_ns is not None and self.start_ns is None:
            raise ValueError("start_ns is required when end_ns is provided")
        if (
            self.start_ns is not None
            and self.end_ns is not None
            and self.end_ns <= self.start_ns
        ):
            raise ValueError("evidence end_ns must be greater than start_ns")


@dataclass
class Decision:
    """单条质量检查决策。"""
    stage: int
    reason: ReasonCode
    severity: Severity
    decision: DecisionType = DecisionType.KEEP_WITH_FLAG
    message: str = ""
    frame_idx: int | None = None
    timestamp_ns: int | None = None
    span_start_ns: int | None = None
    span_end_ns: int | None = None
    evidence: list[Evidence] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.stage <= 12:
            raise ValueError("stage must be between 0 and 12")
        if self.decision != DecisionType.KEEP and not self.message:
            raise ValueError("message is required for non-keep decisions")
        if self.decision != DecisionType.KEEP and not self.evidence:
            raise ValueError("evidence is required for non-keep decisions")
        if self.span_end_ns is not None and self.span_start_ns is None:
            raise ValueError("span_start_ns is required when span_end_ns is provided")
        if (
            self.span_start_ns is not None
            and self.span_end_ns is not None
            and self.span_end_ns <= self.span_start_ns
        ):
            raise ValueError("span_end_ns must be greater than span_start_ns")
