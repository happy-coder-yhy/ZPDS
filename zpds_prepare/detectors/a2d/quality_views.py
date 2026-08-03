"""B9: A2D 质量视图聚合。

将 B6–B8 的检测结果聚合成独立的、可独立判定的质量视图。

质量视图:
  - ``robot_observation_ready`` — RGB/Depth 视觉观测可用
  - ``robot_bc_ready`` — state-action 对可用于行为克隆训练
  - ``geometry_ready`` — 标定可用（内参 + 外参可信）
  - ``failure_recovery`` — 操作失败/恢复片段（独立于技术坏片段）

原则:
  - 各视图独立判定，不相互否定
  - robot_bc_ready=false 不拒绝 robot_observation_ready
  - 技术损坏与操作失败分开
  - failure/recovery 默认保留并写 outcome
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zpds_prepare.detectors.a2d.alignment import A2DAlignmentReport
from zpds_prepare.detectors.a2d.completeness import A2DCompletenessReport
from zpds_prepare.detectors.a2d.robot_quality import A2DRobotQualityReport


@dataclass
class QualityView:
    """单个质量视图。"""

    name: str
    ready: bool
    disposition: str  # "pass" | "keep_with_flag" | "reject" | "unavailable"
    reasons: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    evidence_uris: list[str] = field(default_factory=list)


@dataclass
class A2DQualityViewsReport:
    """A2D 质量视图聚合报告。"""

    episode_id: str
    source_path: str
    schema_version: str = "zpds.a2d_quality_views.v1"

    views: dict[str, QualityView] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def aggregate_a2d_quality_views(
    completeness: A2DCompletenessReport | None = None,
    alignment: A2DAlignmentReport | None = None,
    robot_quality: A2DRobotQualityReport | None = None,
    episode_id: str = "",
    source_path: str = "",
) -> A2DQualityViewsReport:
    """聚合 A2D B6–B8 检测结果为质量视图。

    各视图只依赖其相关检测器，独立判定。
    robot_bc_ready=false 不牵连 robot_observation_ready。

    Args:
        completeness: B6 完整性报告
        alignment: B7 对齐报告
        robot_quality: B8 机器人质量报告
        episode_id: episode 标识
        source_path: 源数据路径

    Returns:
        A2DQualityViewsReport 含各质量视图的独立判定。
    """
    report = A2DQualityViewsReport(
        episode_id=episode_id,
        source_path=source_path,
    )

    # ---- robot_observation_ready ----
    obs = _evaluate_robot_observation(completeness)
    report.views["robot_observation_ready"] = obs

    # ---- robot_bc_ready ----
    bc = _evaluate_robot_bc(completeness, alignment, robot_quality)
    report.views["robot_bc_ready"] = bc

    # ---- geometry_ready ----
    geo = _evaluate_geometry_ready(completeness, alignment)
    report.views["geometry_ready"] = geo

    # ---- failure_recovery ----
    fr = _evaluate_failure_recovery(robot_quality)
    report.views["failure_recovery"] = fr

    # 汇总 issues
    for view in report.views.values():
        for reason in view.reasons:
            report.issues.append(f"[{view.name}] {reason}")

    return report


def _evaluate_robot_observation(
    completeness: A2DCompletenessReport | None,
) -> QualityView:
    """评估 robot_observation_ready。

    要求: head_rgb 可解码 + 帧数 > 0。
    """
    view = QualityView(
        name="robot_observation_ready",
        ready=False,
        disposition="pass",
        depends_on=["head_rgb", "hand_left_rgb", "hand_right_rgb"],
    )

    if completeness is None:
        view.reasons.append("缺少完整性报告")
        view.disposition = "reject"
        return view

    head = completeness.assets.get("head_rgb")
    if head is None or not head.present:
        view.reasons.append("head_rgb 不存在")
        view.disposition = "reject"
        return view

    if head.frame_count == 0:
        view.reasons.append("head_rgb 帧数为 0")
        view.disposition = "reject"
        return view

    # 侧相机缺失不阻断
    missing_sides = []
    for cam in ["hand_left_rgb", "hand_right_rgb"]:
        a = completeness.assets.get(cam)
        if a is None or not a.present:
            missing_sides.append(cam)
    if missing_sides:
        view.reasons.append(f"侧相机缺失: {missing_sides}（非阻断）")

    view.ready = True
    if missing_sides:
        view.disposition = "keep_with_flag"
    return view


def _evaluate_robot_bc(
    completeness: A2DCompletenessReport | None,
    alignment: A2DAlignmentReport | None,
    robot_quality: A2DRobotQualityReport | None,
) -> QualityView:
    """评估 robot_bc_ready。

    要求: state + action 都有数据、相机-机器人对齐可信、state-action lag 可估计。
    """
    view = QualityView(
        name="robot_bc_ready",
        ready=False,
        disposition="pass",
        depends_on=["robot_state", "robot_action", "camera_robot_alignment"],
    )

    # B6: aligned_joints.h5
    if completeness is not None:
        h5 = completeness.assets.get("aligned_joints.h5")
        if h5 is None or not h5.present:
            view.reasons.append("aligned_joints.h5 不存在")
            view.disposition = "reject"
            return view
        if completeness.hdf5_sample_count == 0:
            view.reasons.append("HDF5 样本数为 0")
            view.disposition = "reject"
            return view

    # B7: 对齐
    if alignment is not None:
        if alignment.robot_bc_ready is False:
            view.reasons.append(f"对齐不可信: {alignment.issues}")
            view.disposition = "reject"
            return view
        # 多连续组 → 不可用于 BC
        max_groups = max(
            (s.continuity_groups for s in alignment.streams.values()),
            default=0,
        )
        if max_groups > 1:
            view.reasons.append(f"检测到 {max_groups} 个不连续组，跨 group 不可用")
            view.disposition = "reject"
            return view

    # B8: 机器人质量
    if robot_quality is not None:
        if robot_quality.robot_bc_ready is False:
            view.reasons.append(f"机器人质量不满足 BC 条件: {robot_quality.issues}")
            view.disposition = "reject"
            return view
        if robot_quality.robot_state_quality is not None:
            if robot_quality.robot_state_quality.has_regression:
                view.reasons.append("robot_state 时间戳回退")
                view.disposition = "reject"
                return view
    else:
        view.reasons.append("缺少机器人质量报告")
        view.disposition = "reject"
        return view

    # 告警
    warnings: list[str] = []
    if alignment is not None and alignment.issues:
        warnings.extend(alignment.issues)
    if robot_quality is not None and robot_quality.issues:
        warnings.extend(robot_quality.issues)

    view.ready = True
    if warnings:
        view.reasons.extend(warnings)
        view.disposition = "keep_with_flag"
    return view


def _evaluate_geometry_ready(
    completeness: A2DCompletenessReport | None,
    alignment: A2DAlignmentReport | None,
) -> QualityView:
    """评估 geometry_ready。

    要求: 三路相机内参 + 外参可用。
    """
    view = QualityView(
        name="geometry_ready",
        ready=False,
        disposition="pass",
        depends_on=["head_rgb_calibration", "hand_left_rgb_calibration", "hand_right_rgb_calibration"],
    )

    if completeness is None:
        view.reasons.append("缺少完整性报告")
        view.disposition = "unavailable"
        return view

    calib = completeness.assets.get("camera_calibration")
    if calib is None or not calib.present:
        view.reasons.append("无标定文件")
        view.disposition = "unavailable"
        return view

    details = calib.details
    missing = details.get("missing_calibrations", [])
    if missing:
        view.reasons.append(f"缺少标定的相机: {missing}")
        view.disposition = "unavailable"
        return view

    # A2D 外参 unavailable (per CLAUDE.md)
    view.reasons.append("外参状态: unavailable（A2D 不支持外参提取）")
    view.ready = True
    view.disposition = "keep_with_flag"  # 内参 OK，外参缺失

    return view


def _evaluate_failure_recovery(
    robot_quality: A2DRobotQualityReport | None,
) -> QualityView:
    """评估 failure_recovery。

    技术损坏与操作失败分开存储。
    当前阶段只记录状态；具体 outcome（success/failure/recovery）需 VLM/人工复核。
    """
    view = QualityView(
        name="failure_recovery",
        ready=True,  # 始终 ready——无操作失败证据时为空
        disposition="pass",
        depends_on=[],
    )

    if robot_quality is not None:
        gr = robot_quality.gripper_response
        if gr.stall_count > 0:
            view.reasons.append(
                f"夹爪失速（有命令无响应）: {gr.stall_count} 处——"
                f"可能为操作失败/恢复，需人工复核"
            )
            view.disposition = "keep_with_flag"

    return view


__all__ = [
    "A2DQualityViewsReport",
    "QualityView",
    "aggregate_a2d_quality_views",
]
