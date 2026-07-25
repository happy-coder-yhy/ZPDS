"""
A2D 机器人专用质量检测器。

检测项:
    timeseries_structure  — 时序结构 (单调性 / NaN / 维度)
    joint_quality         — 关节状态 (position / velocity / temperature / freeze)
    action_quality        — 动作指令 (维度 / NaN / 变化 / 覆盖)
    alignment_quality     — 相机-机器人对齐 (误差 / 缺失 / 缺口)
    gap_detection         — 流缺口检测 (TS 缺口 / 相机缺失 → split / keep_with_flag)
"""

from zpds_prepare.detectors.robot.timeseries_structure import detect_timeseries_structure
from zpds_prepare.detectors.robot.joint_quality import detect_joint_quality
from zpds_prepare.detectors.robot.action_quality import detect_action_quality
from zpds_prepare.detectors.robot.alignment_quality import detect_alignment_quality
from zpds_prepare.detectors.robot.gap_detection import detect_timeseries_gaps, detect_camera_gaps

__all__ = [
    "detect_timeseries_structure",
    "detect_joint_quality",
    "detect_action_quality",
    "detect_alignment_quality",
    "detect_timeseries_gaps",
    "detect_camera_gaps",
]
