"""质量检查决策与严重等级枚举。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    FATAL = "fatal"    # 阻断：segment 不可用
    ERROR = "error"    # 强制修复
    WARN = "warn"      # 建议修复
    INFO = "info"      # 仅记录


class Disposition(str, Enum):
    """质量决策对数据的处理语义。"""

    KEEP = "keep"
    KEEP_WITH_FLAG = "keep_with_flag"
    QUARANTINE = "quarantine"
    TRIM = "trim"
    SPLIT = "split"
    REJECT = "reject"


class ReasonCode(str, Enum):
    """预定义决策原因码，覆盖 Stage 0～12。"""
    # Stage 0 — 文件清单
    FILE_MISSING = "file_missing"
    HASH_MISMATCH = "hash_mismatch"
    # Stage 1 — 结构
    CONTAINER_CORRUPT = "container_corrupt"
    INDEX_MISSING = "index_missing"
    SCHEMA_UNKNOWN = "schema_unknown"
    # Stage 2 — 时间
    TIMESTAMP_GAP = "timestamp_gap"
    TIMESTAMP_REGRESSION = "timestamp_regression"
    CLOCK_MISALIGN = "clock_misalign"
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
    FLOW_INCONSISTENT = "flow_inconsistent"
    VIDEO_TOO_SHORT = "video_too_short"
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
    # Stage 8 — 标定
    INTRINSICS_MISSING = "intrinsics_missing"
    EXTRINSICS_INVALID = "extrinsics_invalid"
    REPROJECTION_ERROR_HIGH = "reprojection_error_high"
    # Stage 9 — 手部
    HAND_ABSENT = "hand_absent"
    HAND_TRACK_LOST = "hand_track_lost"
    HAND_OCCLUDED = "hand_occluded"
    HAND_POSE_INCOMPLETE = "hand_pose_incomplete"
    NO_OPERATION = "no_operation"
    # Stage 10 — 语义
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"
    # Stage 11 — 去重
    NEAR_DUPLICATE = "near_duplicate"
    # Stage 12 — 交付
    DELIVERY_CHECK_FAIL = "delivery_check_fail"


@dataclass
class Decision:
    """单条质量检查决策。"""
    stage: int
    reason: ReasonCode
    severity: Severity
    message: str = ""
    frame_idx: int | None = None
    timestamp_ns: int | None = None
    end_frame_idx: int | None = None
    end_timestamp_ns: int | None = None
    disposition: Disposition | None = None
    detail: dict = field(default_factory=dict)
